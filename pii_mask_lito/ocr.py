"""OCR engines, behind one small protocol.

This is the accuracy ceiling for every scanned input, so it must be swappable
without touching detection or rendering. Two implementations ship; neither is
imported until it is chosen, so a user who only masks native PDFs and
spreadsheets installs neither.
"""

from __future__ import annotations

import os
import shutil
from importlib.util import find_spec
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from PIL import Image

from .model import Token

# Below this Tesseract confidence a token is kept only when it still looks like
# a word. See TesseractOcr.words().
LOW_CONFIDENCE = 40


class OcrEngine(Protocol):
    def words(self, image: Image.Image, page: int = 0) -> list[Token]: ...


class PaddleOcr:
    """Default. pip-installable, no system dependency, strong on dense text."""

    name = "paddle"
    # Paddle predictors are stateful and memory-intensive. Concurrent calls on
    # a shared instance are not supported, and one model per worker can consume
    # substantial memory, so the conservative default is serial execution.
    parallel_safe = False

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR

        # 3.x preprocesses by default: it classifies document orientation and
        # unwarps the page, then reports polygons in that *corrected* image's
        # coordinates. Masks are drawn on the original render, so preprocessing
        # that transforms geometry can misalign them. PDF renders are already in
        # a stable coordinate space; disabling those transforms preserves it.
        try:
            self._engine = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except (ValueError, TypeError):
            # 2.x: no preprocessing to disable, and it wants the old arguments.
            try:
                self._engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            except (ValueError, TypeError):
                self._engine = PaddleOCR(lang=lang)

    def words(self, image: Image.Image, page: int = 0) -> list[Token]:
        import numpy as np

        array = np.array(image)
        # 2.x takes `cls` and returns [[(quad, (text, score)), ...]]; 3.x
        # dropped the argument and returns a dict of parallel lists. Both are
        # still in the wild, and the difference is a TypeError deep inside a
        # scanned page rather than anything a version check would catch early.
        try:
            result = self._engine.ocr(array, cls=True)
        except TypeError:
            result = self._engine.predict(array)
        if not result:
            return []

        w, h = image.size
        tokens = []
        for quad, text, confidence in _paddle_lines(result):
            if not text.strip() or confidence < 0.3:
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            box = (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)
            # Paddle returns lines. Split to words by looking at where the ink
            # actually is, falling back to apportioning by character count.
            words = _split_line_by_ink(image, text, box, page)
            for word in words:
                # Paddle scores a line, not a word. Every word inherits it,
                # which is the best available answer and is what vision.py
                # needs: a line Paddle was unsure of is unread ink.
                word.confidence = float(confidence)
            tokens += words
        return tokens


def _paddle_lines(result):
    """(quad, text, score) per line, from either PaddleOCR generation."""
    first = result[0]
    if isinstance(first, dict):  # 3.x: parallel lists on a result dict
        polys = first.get("rec_polys")
        if polys is None:
            polys = first.get("dt_polys", [])
        return list(zip(polys, first.get("rec_texts", []), first.get("rec_scores", [])))
    if not first:
        return []
    return [(quad, text, score) for quad, (text, score) in first]


class DocTrOcr:
    """docTR (Mindee): detection and recognition that reports *words*.

    Paddle and RapidOCR report line boxes, which this module must split into
    word geometry by inspecting ink gaps. docTR provides word boxes directly,
    avoiding that approximation on noisy or textured scans.

    Apache-2.0. Model weights download once and are cached, like paddle's.
    """

    name = "doctr"

    # The predictor holds mutable model state, so shared-instance parallelism is
    # disabled unless the upstream implementation guarantees it.
    parallel_safe = False

    def __init__(self, lang: str = "en"):
        from doctr.models import ocr_predictor

        self._model = ocr_predictor(pretrained=True)

    def words(self, image: Image.Image, page: int = 0) -> list[Token]:
        import numpy as np

        result = self._model([np.array(image.convert("RGB"))])
        tokens = []
        for doc_page in result.pages:
            for block in doc_page.blocks:
                for line_no, line in enumerate(block.lines):
                    for word in line.words:
                        text = (word.value or "").strip()
                        if not text:
                            continue
                        (x0, y0), (x1, y1) = word.geometry
                        tokens.append(
                            Token(
                                text=text,
                                page=page,
                                bbox=(float(x0), float(y0), float(x1), float(y1)),
                                confidence=float(word.confidence),
                            )
                        )
        return tokens


def _v6_model_paths():
    """PP-OCRv6 det/rec ONNX models and character dict, exported and cached.

    The ONNX models are derived from the same PP-OCRv6_medium weights Paddle
    downloads into ~/.paddlex, so a first run needs paddleocr to have fetched
    them once. Export happens once and is cached; after that paddle is not
    touched. This mirrors how paddle itself fetches weights on first use.
    """
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    cache = Path.home() / ".cache" / "pii-mask-lito" / "onnx" / "PP-OCRv6_medium"
    det = cache / "det.onnx"
    rec = cache / "rec.onnx"
    keys = cache / "dict.txt"
    if det.exists() and rec.exists() and keys.exists():
        return str(det), str(rec), str(keys)

    source = Path.home() / ".paddlex" / "official_models"
    src_det = source / "PP-OCRv6_medium_det"
    src_rec = source / "PP-OCRv6_medium_rec"
    if not src_det.exists() or not src_rec.exists():
        raise RuntimeError(
            "PP-OCRv6 models are not present. Install paddleocr and run it once "
            "so it downloads them, or set the cache up manually."
        )
    cache.mkdir(parents=True, exist_ok=True)

    # paddle2onnx lives in the same venv bin as this interpreter; PaddleX's CLI
    # finds it via PATH rather than by import.
    bin_dir = str(Path(sys.executable).parent)
    env = dict(os.environ)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    for src, name in ((src_det, "det"), (src_rec, "rec")):
        out = cache / f"{name}_tmp"
        subprocess.run(
            [sys.executable, "-m", "paddlex", "--paddle2onnx",
             "--paddle_model_dir", str(src), "--onnx_model_dir", str(out)],
            check=True, env=env, capture_output=True, text=True,
        )
        shutil.move(str(out / "inference.onnx"), str(cache / f"{name}.onnx"))

    rec_cfg = yaml.safe_load((src_rec / "inference.yml").read_text())
    keys.write_text("\n".join(rec_cfg["PostProcess"]["character_dict"]) + "\n")
    return str(det), str(rec), str(keys)


class OnnxOcr:
    """Paddle's PP-OCRv6 weights on ONNX Runtime -- paddle accuracy, no paddle.

    This is the engine-agnostic path. The weights are exported from the same
    PP-OCRv6_medium models Paddle runs (det + rec), then executed on ONNX
    Runtime, which ships an execution provider for every relevant backend:
    CUDA/TensorRT on NVIDIA, OpenVINO on Intel, DirectML on Windows, CoreML on
    Apple, plain CPU everywhere. One engine, every hardware.

    The v6 recognition model emits word spaces, which the older v4 model did
    not. Preserving those spaces is necessary for the line-to-word ink split
    and the word-boundary rules used by downstream detectors.

    Detection and recognition run inside RapidOCR's battle-tested preprocessing
    and postprocessing, pointed at the v6 models and the v6 character dict. The
    only thing changed from RapidOCR's defaults is the det normalization, which
    v6 uses in ImageNet form rather than v4's 0.5/0.5.
    """

    name = "onnx"
    # Deliberately NOT process- or thread-parallel. ONNX Runtime already fans a
    # single page out across every core through its intra-op thread pool, so one
    # page at a time already uses the machine. Spawning N worker processes, each
    # with its own intra-op pool, multiplies the thread count by N and thrashes
    # when several pages run at once. Parallelism here comes from the runtime,
    # not from an outer page pool.

    def __init__(self, lang: str = "en"):
        from rapidocr_onnxruntime import RapidOCR

        det, rec, keys = _v6_model_paths()
        self._engine = RapidOCR(
            det_model_path=det,
            det_mean=[0.485, 0.456, 0.406],
            det_std=[0.229, 0.224, 0.225],
            det_thresh=0.2,
            det_box_thresh=0.45,
            det_unclip_ratio=1.4,
            det_max_candidates=3000,
            rec_model_path=rec,
            rec_keys_path=keys,
            use_cls=False,
        )

    def words(self, image: Image.Image, page: int = 0) -> list[Token]:
        import numpy as np

        result, _ = self._engine(np.array(image.convert("RGB")))
        if not result:
            return []
        tokens = []
        for entry in result:
            quad, text, confidence = entry[0], entry[1], float(entry[2])
            if not text.strip():
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            w, h = image.size
            box = (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)
            words = _split_line_by_ink(image, text, box, page)
            for word in words:
                word.confidence = confidence
            tokens += words
        return tokens


class RapidOcr:
    """The PP-OCR models Paddle uses, on ONNX Runtime instead of PaddlePaddle.

    The recognition weights come from the same model family; only the runtime
    changes. Apache-2.0, and it ships its own models, so there is no separate
    system install like Tesseract requires.

    Like paddle it reports lines, not words, so the boxes go through the same
    ink-based splitting.
    """

    name = "rapidocr"

    # ONNX Runtime releases the GIL inside its own session, but two sessions on
    # one process still contend for the same intra-op thread pool. Left off
    # until measured, for the same reason it is off for paddle.
    parallel_safe = False

    def __init__(self, lang: str = "en"):
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def _reread(self, image: Image.Image, quad) -> str:
        """Recognition only, on one line's crop, to recover its spaces."""
        import numpy as np

        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        crop = image.crop((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        if crop.width < 2 or crop.height < 2:
            return ""
        try:
            out, _ = self._engine(np.array(crop), use_det=False, use_cls=False,
                                  use_rec=True)
        except Exception:  # noqa: BLE001
            return ""
        return out[0][0] if out else ""

    def words(self, image: Image.Image, page: int = 0) -> list[Token]:
        import numpy as np

        result, _elapsed = self._engine(np.array(image))
        if not result:
            return []

        w, h = image.size
        tokens = []
        for entry in result:
            quad, text, confidence = entry[0], entry[1], float(entry[2])
            # Re-read the line from its own crop. The whole-page path returns
            # this model's text with spaces stripped -- "CALDER,MIRA" -- while
            # downstream detectors depend on word boundaries. Recognising the
            # same crop on its own commonly restores "CALDER, MIRA".
            text = self._reread(image, quad) or text
            if not text.strip() or confidence < 0.3:
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            box = (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)
            words = _split_line_by_ink(image, text, box, page)
            for word in words:
                # A line score, inherited by every word in it -- the same
                # contract paddle has, and what vision.py reads to decide
                # whether ink was genuinely read.
                word.confidence = confidence
            tokens += words
        return tokens


class TesseractOcr:
    """Drop-in alternative. True word-level boxes; needs a system install."""

    name = "tesseract"

    # Each call shells out to Tesseract, so page-level calls can run in parallel.
    parallel_safe = True

    def __init__(self, lang: str = "eng", config: str = "--psm 6"):
        self.lang, self.config = lang, config

    def words(self, image: Image.Image, page: int = 0) -> list[Token]:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            image, lang=self.lang, config=self.config, output_type=Output.DICT
        )
        w, h = image.size
        tokens = []
        for i, text in enumerate(data["text"]):
            stripped = text.strip()
            if not stripped:
                continue
            confidence = int(data["conf"][i])
            # Retain low-confidence word-like tokens for conservative matching;
            # discard isolated marks and punctuation.
            word_like = sum(c.isalnum() for c in stripped) >= 4
            if confidence < LOW_CONFIDENCE and not word_like:
                continue
            x, y, bw, bh = (data[k][i] for k in ("left", "top", "width", "height"))
            tokens.append(
                Token(
                    text=text,
                    page=page,
                    bbox=(x / w, y / h, (x + bw) / w, (y + bh) / h),
                    line=data["line_num"][i],
                    confidence=max(confidence, 0) / 100,
                )
            )
        return tokens


# How many pages to read at once. OCR here is a subprocess, so this is bounded
# by cores rather than by anything the engine knows about itself.
def _default_workers() -> int:
    return min(8, (os.cpu_count() or 2))


def read_pages(engine, images, pages: list[int] | None = None, workers: int | None = None):
    """Tokens for a list of page images, concurrently where that is safe.

    A masking run can read pages during extraction, convergence rechecks, and
    verification. Within any one pass, page reads are independent: one page's
    tokens do not depend on another page's pixels.

    Results come back in the order the images were given, regardless of
    completion order, because everything downstream indexes by page and a
    reordered token stream would be a silent geometry bug rather than a loud one.

    `pages` is the page number to stamp on each image's tokens, and it is
    explicit because the caller does not always hand over a whole document: the
    recheck loop reads a *subset* of pages, and numbering those by their position
    in the sublist would report a finding on page 7 as a finding on page 0. It
    defaults to 0..n-1, which is right when the images are the whole document.

    An engine that says it is not parallel-safe is run serially; see each engine
    adapter for the concurrency contract it exposes.
    """
    images = list(images)
    numbers = list(range(len(images))) if pages is None else list(pages)
    if len(numbers) != len(images):
        raise ValueError(f"got {len(images)} images and {len(numbers)} page numbers")
    if getattr(engine, "parallel_safe", False) and len(images) >= 2:
        with ThreadPoolExecutor(max_workers=workers or _default_workers()) as pool:
            return list(pool.map(lambda pair: engine.words(pair[0], page=pair[1]),
                                 zip(images, numbers)))
    # An engine that serialises internally cannot be helped by threads, but a
    # process does not share whatever it serialises on. Every page is
    # independent, but each worker pays the model load once, so this only earns
    # its keep on a document long enough to amortise that cost.
    if (getattr(engine, "process_safe", False) and getattr(engine, "name", None)
            and len(images) >= _PROCESS_MIN_PAGES):
        try:
            return _read_pages_in_processes(engine.name, images, numbers, workers)
        except Exception:  # noqa: BLE001
            # A pool that cannot start is a speed problem, never a correctness
            # one: fall through and read the pages here instead.
            pass
    return [engine.words(image, page=n) for image, n in zip(images, numbers)]


# Below this many pages the model load in each worker costs more than the
# parallelism returns. Keep this conservative and benchmark changes with
# ``tests.bench`` on the public synthetic corpus.
_PROCESS_MIN_PAGES = 4
# Each worker holds a full engine in memory, so this is bounded by RAM rather
# than by cores. Four paddle engines is already several gigabytes.
_PROCESS_WORKERS = 4

_WORKER_ENGINE = None


def _worker_init(name: str) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = ENGINES[name]()


def _worker_read(job):
    image, number = job
    return _WORKER_ENGINE.words(image, page=number)


def _read_pages_in_processes(name, images, numbers, workers=None):
    """Read pages in separate processes, one engine per worker."""
    from concurrent.futures import ProcessPoolExecutor

    count = min(workers or _PROCESS_WORKERS, len(images), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=count, initializer=_worker_init,
                             initargs=(name,)) as pool:
        return list(pool.map(_worker_read, list(zip(images, numbers))))


_INK_LEVEL = 200  # 8-bit grey below this counts as ink


def _split_line_by_ink(image: Image.Image, text: str, box, page: int) -> list[Token]:
    """Word boxes from where the ink sits inside a line, not from arithmetic.

    Character-count apportionment is a guess, and on a proportional font it
    drifts far enough that masks land beside their text: verification found 21
    values still readable in a document masked this way. But the line box itself
    is accurate, and inside it the words are separated by columns of blank
    paper. Projecting the ink down onto the x-axis makes those gaps visible.

    The ink is segmented into *visual* words before the text is consulted at
    all, by cutting only at gaps wider than a fraction of the line height. That
    ordering matters, because a recognizer's idea of a word cannot be trusted:
    Some line recognizers emit no spaces. Visual whitespace can still separate
    a name from a neighboring identifier before text-based rules run.
    """
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return _split_line(text, box, page)

    words = text.split()
    width, height = image.size
    x0, y0, x1, y1 = box
    left, right = int(x0 * width), min(int(x1 * width) + 1, width)
    top, bottom = int(y0 * height), min(int(y1 * height) + 1, height)
    if right - left < 2 or bottom - top < 1:
        return _split_line(text, box, page)

    strip = np.asarray(image.convert("L").crop((left, top, right, bottom)))
    ink = (strip < _INK_LEVEL).any(axis=0)
    runs, run_start = [], None
    for i, has_ink in enumerate(ink):
        if has_ink and run_start is None:
            run_start = i
        elif not has_ink and run_start is not None:
            runs.append((run_start, i))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(ink)))
    if not runs:
        return _split_line(text, box, page)

    # Join runs separated by less than a word space. Letters inside a word sit
    # a hair apart; words sit a fraction of the line height apart. Cutting only
    # at the wide gaps turns raw ink runs into visual words without ever asking
    # the recognizer what it thought it read.
    line_height = max(bottom - top, 1)
    word_gap = max(line_height * _WORD_GAP_RATIO, 2.0)
    clusters = [list(runs[0])]
    for run in runs[1:]:
        if run[0] - clusters[-1][1] >= word_gap:
            clusters.append(list(run))
        else:
            clusters[-1][1] = run[1]

    if len(words) == len(clusters):
        pieces = words
    elif len(words) < len(clusters):
        # Glued text: apportion the characters across the visual words by how
        # much ink each one holds.
        pieces = _apportion(text.replace(" ", ""), [c[1] - c[0] for c in clusters])
    else:
        # More words than the page shows: the ink could not be separated
        # (touching glyphs, a hyphen break). Estimation is all that is left.
        return _split_line(text, box, page)

    tokens = []
    for piece, (cs, ce) in zip(pieces, clusters):
        if not piece.strip():
            continue
        # The box is the ink, not the space around it: tight in x from the
        # cluster, tight in y from the rows that cluster actually marks.
        cols = strip[:, cs:ce]
        rows = np.flatnonzero((cols < _INK_LEVEL).any(axis=1))
        gy0, gy1 = (int(rows[0]), int(rows[-1]) + 1) if rows.size else (0, strip.shape[0])
        tokens.append(
            Token(
                text=piece.strip(),
                page=page,
                bbox=((left + cs) / width, (top + gy0) / height,
                      (left + ce) / width, (top + gy1) / height),
            )
        )
    return tokens or _split_line(text, box, page)


# A gap this fraction of the line height or wider separates two words. Letter
# spacing inside a word is far below it; a column boundary is far above.
_WORD_GAP_RATIO = 0.22


def _apportion(text: str, widths: list[int]) -> list[str]:
    """Share a run of characters out across ink clusters, by ink width."""
    total = float(sum(widths)) or 1.0
    pieces, cursor = [], 0
    for index, w in enumerate(widths):
        if index == len(widths) - 1:
            pieces.append(text[cursor:])
            break
        take = int(round(len(text) * w / total))
        pieces.append(text[cursor:cursor + take])
        cursor += take
    return pieces


def _split_line(text: str, box, page: int) -> list[Token]:
    """Apportion a line box across its words, leaving no gap between them.

    Paddle reports lines, not words, so a word's box here is an estimate: the
    line's width shared out by character count. On a proportional font that
    estimate can drift far enough for a mask to land beside the intended word.

    So each box is grown to meet its neighbours, and the first and last to the
    ends of the line. The estimate is no better, but the boxes now tile the line
    completely: a value cannot fall in a gap between two of them, and a mask
    covering consecutive words covers everything between them. It costs a little
    over-masking into the surrounding whitespace, which is the affordable
    direction.
    """
    x0, y0, x1, y1 = box
    words = text.split()
    if len(words) <= 1:
        return [Token(text=text.strip(), page=page, bbox=box)]
    total = len(text)
    span = x1 - x0
    edges, cursor = [], 0
    for word in words:
        start = text.index(word, cursor)
        cursor = start + len(word)
        edges.append((word, x0 + span * start / total, x0 + span * cursor / total))

    tokens = []
    for i, (word, left, right) in enumerate(edges):
        lo = x0 if i == 0 else (edges[i - 1][2] + left) / 2
        hi = x1 if i == len(edges) - 1 else (right + edges[i + 1][1]) / 2
        tokens.append(Token(text=word, page=page, bbox=(lo, y0, hi, y1)))
    return tokens


ENGINES = {"onnx": OnnxOcr, "doctr": DocTrOcr, "rapidocr": RapidOcr,
           "paddle": PaddleOcr, "tesseract": TesseractOcr}


def available() -> list[str]:
    """Return installed engines without importing heavyweight frameworks."""
    modules = {
        "onnx": "rapidocr_onnxruntime",
        "tesseract": "pytesseract",
        "paddle": "paddleocr",
        "rapidocr": "rapidocr_onnxruntime",
        "doctr": "doctr",
    }
    found = []
    for name in ("onnx", "tesseract", "paddle", "rapidocr", "doctr"):
        if find_spec(modules[name]) is None:
            continue
        if name == "tesseract" and shutil.which("tesseract") is None:
            continue
        found.append(name)
    return found


# Framework-heavy engines are avoided in iterative rechecks when a lightweight
# installed engine can search for values already present in the lexicon.
SLOW = {"paddle", "onnx"}


def loop_engine_name(primary: str, requested: str = "auto") -> str:
    """Which engine should drive the recheck loop.

    Extraction is where an engine's accuracy is decisive: a value it fails to
    read becomes a token that never exists, and nothing downstream can mask it.
    The recheck loop is a different job -- it re-reads pages that have already
    been masked, hunting occurrences of values the lexicon now knows, which is
    mostly propagation and needs no special sharpness. It is also sixty of the
    hundred page-reads a ten-page document costs.

    So when the primary engine is an expensive one and a cheap one is installed,
    the loop runs on the cheap one. Verification is deliberately *not* included:
    it is the guarantee, and it stays on the engine that read the document.
    """
    if requested == "same":
        return primary
    if requested != "auto":
        return requested
    if primary in SLOW:
        for candidate in available():
            if candidate not in SLOW:
                return candidate
    return primary


def cheap() -> OcrEngine:
    """The first *fast* engine available, falling back to the accurate one.

    Used when OCR is a supplement over an authoritative text layer rather than
    the only reader. A native PDF already has exact token boxes from its text
    layer; OCR there only catches stray raster text and powers the read-back
    verification. A line-level OCR engine can return competing, less precise
    boxes for text the native layer already located exactly, so the supplement
    path prefers a word-level engine with a low startup cost.
    """
    failures = []
    # Tesseract is the lightweight supplemental engine. ONNX and Paddle are
    # reserved for pages whose pixels are the authoritative text source.
    for candidate in available():
        if candidate != "tesseract":
            continue
        try:
            return ENGINES[candidate]()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"  {candidate}: {type(exc).__name__}: {exc}")
    for candidate in available():
        if candidate in SLOW or candidate == "onnx":
            continue
        try:
            return ENGINES[candidate]()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"  {candidate}: {type(exc).__name__}: {exc}")
    return get("auto")


def get(name: str = "auto") -> OcrEngine:
    """Build an OCR engine, picking one that exists when asked for "auto".

    Naming a fixed default is a trap: whichever one it is, the first user
    without it gets an ImportError from deep inside a scanned page rather than
    a usable answer. "auto" tries installed engines in the documented order
    and says plainly what to install when none can be constructed.
    """
    if name == "auto":
        # Importable is not the same as usable: incompatible dependency versions
        # can fail only when an engine is constructed. Build each candidate so
        # "auto" returns a working engine instead of a late runtime error.
        failures = []
        for candidate in available():
            try:
                return ENGINES[candidate]()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"  {candidate}: {type(exc).__name__}: {exc}")
        detail = ("\n\nInstalled but unusable:\n" + "\n".join(failures)) if failures else ""
        raise RuntimeError(
            "no usable OCR engine, and this input needs one (scanned PDF "
            "or image). Install one of:\n"
            "  pip install 'pii_mask_lito[tesseract]' && brew install tesseract\n"
            "  pip install 'pii_mask_lito[paddle]'" + detail
        )
    if name not in ENGINES:
        raise ValueError(f"unknown OCR engine {name!r}; choose from {sorted(ENGINES)}")
    return ENGINES[name]()

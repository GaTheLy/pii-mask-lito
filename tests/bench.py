"""Measure OCR cost and token yield on a caller-supplied PDF.

OCR throughput varies with page density, resolution, engine version, hardware,
and process topology. This utility records the inputs and measures three useful
dimensions without embedding results from a private document in the project:

    per-page OCR cost and token yield, per engine
    what a serial pipeline costs once page count grows
    what the same work costs across cores

Run:
    python -m tests.bench tests/corpus/generated/support_case_scan.pdf
    python -m tests.bench <pdf> --pages 5 --engines tesseract paddle
    python -m tests.bench <pdf> --recall --engines tesseract paddle unlimited

Treat output as environment-specific evidence, not a universal speed claim.
Any engine or parameter change must also pass the synthetic quality corpus;
token count and elapsed time alone do not establish masking recall.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from pii_mask_lito import ocr, pdf
from pii_mask_lito.model import Token


def _render(path: str, limit: int | None, dpi: int):
    """Pages as images, timed. Rendering is a real cost and is not free."""
    start = time.perf_counter()
    pages = pdf.page_images(path, dpi)
    if limit:
        pages = pages[:limit]
    return pages, time.perf_counter() - start


def _time_engine(engine, pages) -> dict:
    """Per-page wall time and token yield, serially."""
    times, counts = [], []
    for i, image in enumerate(pages):
        start = time.perf_counter()
        tokens = engine.words(image, page=i)
        times.append(time.perf_counter() - start)
        counts.append(len(tokens))
    return {
        "total": sum(times),
        "per_page": statistics.median(times),
        "slowest": max(times),
        "tokens": sum(counts),
    }


def _time_parallel(engine, pages, workers: int) -> dict:
    """The same work across a thread pool.

    Threads rather than processes on purpose: both engines spend their time
    inside native code that releases the GIL -- tesseract in a subprocess,
    paddle in its own runtime -- and a process pool would pay to re-import and
    re-load the model in every worker, which on paddle costs more than the page.

    One engine object is shared. If that turns out not to be safe the token
    count diverges from the serial run, which is what the caller compares.
    """
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda a: engine.words(a[1], page=a[0]), enumerate(pages)))
    return {"total": time.perf_counter() - start, "tokens": sum(len(r) for r in results)}


def _load(name: str):
    start = time.perf_counter()
    engine = ocr.get(name)
    return engine, time.perf_counter() - start


def run(path: str, engines: list[str], limit: int | None, dpi: int, workers: int) -> None:
    pages, render_seconds = _render(path, limit, dpi)
    n = len(pages)
    print(f"\n{path}  {n} page(s) at {dpi} DPI")
    print(f"render          {render_seconds:6.1f}s total  {render_seconds / n:5.2f}s/page\n")

    print(f"{'engine':<14}{'load':>7}{'s/page':>9}{'slowest':>9}{'total':>8}"
          f"{'tokens':>8}{'par':>8}{'speedup':>9}")
    print("-" * 72)
    for name in engines:
        try:
            engine, load_seconds = _load(name)
        except Exception as exc:  # noqa: BLE001 - an absent engine is a row, not a crash
            print(f"{name:<14}  unavailable: {type(exc).__name__}: {str(exc)[:40]}")
            continue
        serial = _time_engine(engine, pages)
        parallel = _time_parallel(engine, pages, workers)
        speedup = serial["total"] / parallel["total"] if parallel["total"] else 0
        # A shared engine that is not thread-safe shows up here, as a token
        # count that no longer matches the serial run.
        warn = "" if parallel["tokens"] == serial["tokens"] else "  UNSAFE: token count differs"
        print(f"{name:<14}{load_seconds:6.1f}s{serial['per_page']:8.2f}s"
              f"{serial['slowest']:8.2f}s{serial['total']:7.1f}s{serial['tokens']:8d}"
              f"{parallel['total']:7.1f}s{speedup:8.1f}x{warn}")

    print(f"\nparallel column: {workers} threads over the same {n} pages, one shared engine.")
    print("A full masking run may read pages during extraction, convergence rechecks,\n"
          "and verification; measure end-to-end latency separately.")


# --------------------------------------------------------------------------
# Baidu Unlimited-OCR, as a benchmark subject
# --------------------------------------------------------------------------
#
# A 3B DeepSeek-V2 vision-language model (MIT) that parses a whole page to
# markdown in one shot, optionally tagging elements with <|det|> boxes. It is
# here to answer one question the installed engines cannot: is there text on a
# complex scanned form that neither Tesseract nor Paddle can read? Nothing unread can be
# masked, flagged or verified, so a recall ceiling is worth knowing even if the
# model itself never ships.
#
# It is NOT wired into piimask, and the reason is the same one that shapes the
# agent path: its boxes come out of a language model. This project's rule is
# that no model in the pipeline emits a coordinate, because a redaction box
# fifteen pixels off does not fail loudly -- it leaves the value readable. So
# the comparison below is on *text*, which is the thing a VLM is actually good
# at, and geometry is reported separately and treated as unproven.

# This optional benchmark is intentionally separate from the runtime pipeline.
# It requires the model's documented dependencies and suitable accelerator.
UNLIMITED_MODEL = "baidu/Unlimited-OCR"
_DET = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(\[\[.*?\]\])<\|/det\|>", re.S)
_MARKUP = re.compile(r"<\|[^|]*\|>|\[\[[\d,\s]+\]\]")


def _use_mps() -> str:
    """Route the model's hardcoded .cuda() calls at whatever this machine has.

    The published code calls .cuda() in 22 places and was tested on CUDA 12.9.
    On Apple silicon that is an immediate RuntimeError, which would report the
    model as unusable when the only thing wrong is the device name. Patching the
    two methods it actually uses is cheaper than forking the modeling file, and
    it keeps the benchmark honest: the model runs, and any slowness observed is
    the model's, not a missing backend's.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.Tensor.cuda = lambda self, *a, **k: self.to(device)
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(device)

    # And the autocast block around generation, which names "cuda" explicitly.
    # Off CUDA that context manager is inert, so bfloat16 weights run
    # unautocast and may emit an empty result. Loading fp32 instead does not help
    # either, because the image tensors are built as
    # bfloat16 regardless and the first conv rejects the mix. Pointing autocast
    # at the device that exists is what the published code meant to do.
    original = torch.autocast

    class _Autocast(original):
        def __init__(self, device_type, *a, **k):
            super().__init__(device if device_type == "cuda" else device_type, *a, **k)

    torch.autocast = _Autocast
    return device


class UnlimitedOcr:
    """Whole-page parse, adapted to the per-page `OcrEngine` shape.

    One model call per page returning markdown, so `words()` here is doing
    something categorically different from tesseract's word boxes: it is reading
    the document, not locating glyphs. Tokens come back with the model's own
    <|det|> boxes when it emits them (normalized 0-999) and with no box at all
    when it does not -- which is most of the time, since the parsing prompt is
    asked for content, not layout.
    """

    def __init__(self, model_name: str = UNLIMITED_MODEL, prompt: str | None = None,
                 base_size: int = 1024, image_size: int = 640, crop_mode: bool = True,
                 dtype: str = "auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = _use_mps()
        # The published code wraps generation in autocast("cuda"), which is a
        # no-op off CUDA -- so bfloat16 weights then run unautocast and the model
        # can emit EOS immediately. fp32 costs memory and buys an answer, which
        # is the only trade worth making
        # for a benchmark whose whole point is what the model can read.
        if dtype == "auto":
            dtype = "bfloat16"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=getattr(torch, dtype),
                _attn_implementation="eager", use_safetensors=True,
            )
            .eval()
            .to(self.device)
        )
        self.prompt = prompt or "<image>\n<|grounding|>Convert the document to markdown."
        self.base_size, self.image_size, self.crop_mode = base_size, image_size, crop_mode
        self.last_text = ""

    def words(self, image, page: int = 0) -> list[Token]:
        with tempfile.TemporaryDirectory() as work:
            path = f"{work}/page.png"
            image.save(path)
            text = self.model.infer(
                self.tokenizer, prompt=self.prompt, image_file=path, output_path=work,
                base_size=self.base_size, image_size=self.image_size,
                crop_mode=self.crop_mode, eval_mode=True, save_results=False,
            )
        self.last_text = text = text if isinstance(text, str) else str(text)
        return self._tokens(text, page)

    @staticmethod
    def _tokens(text: str, page: int) -> list[Token]:
        """Words, with a box where the model volunteered one.

        Boxes are normalized 0-999 in this model family, and every one of them
        is a number a language model generated. They are carried through so the
        benchmark can report on them; they are not evidence of anything yet.
        """
        tokens, consumed = [], []
        for label, raw in _DET.findall(text):
            consumed.append(label)
            try:
                x0, y0, x1, y1 = json.loads(raw)[0]
            except (ValueError, IndexError, TypeError):
                continue
            box = (x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000)
            for word in _MARKUP.sub(" ", label).split():
                tokens.append(Token(text=word, page=page, bbox=box))
        # Everything the model wrote outside a <|det|> block is still text it
        # read, and for a recall comparison that is the whole point.
        plain = _MARKUP.sub(" ", _DET.sub(" ", text))
        tokens += [Token(text=word, page=page) for word in plain.split()]
        return tokens


def _words(tokens) -> set[str]:
    """Comparable word forms: case and edge punctuation are OCR noise here."""
    return {w for w in (t.text.strip(".,:;|()[]#*_-").casefold() for t in tokens) if len(w) > 1}


def recall(path: str, engines: list[str], limit: int | None, dpi: int) -> None:
    """Who reads what. Text only -- no engine is asked to be right about where.

    Reported as a pairwise "found and the other missed" matrix rather than as a
    score against ground truth, because an arbitrary caller-supplied PDF does
    not have one. The comparison shows whether an engine reads tokens another
    engine never sees; use ``tests.corpus.score`` for actual accuracy scoring.
    """
    pages, _ = _render(path, limit, dpi)
    reads: dict[str, set[str]] = {}
    for name in engines:
        try:
            engine = ocr.get(name) if name in ocr.ENGINES else UnlimitedOcr()
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: unavailable -- {type(exc).__name__}: {str(exc)[:120]}")
            continue
        start = time.perf_counter()
        found: set[str] = set()
        for i, image in enumerate(pages):
            found |= _words(engine.words(image, page=i))
        reads[name] = found
        print(f"{name:<16}{len(found):5d} distinct words   {time.perf_counter()-start:7.1f}s")

    names = list(reads)
    print(f"\n{'':<16}" + "".join(f"{n[:13]:>15}" for n in names) + "   <- and this one missed it")
    for a in names:
        row = "".join(f"{len(reads[a] - reads[b]):15d}" if a != b else f"{'--':>15}" for b in names)
        print(f"{a:<16}{row}")
    if len(names) > 1:
        everyone = set.intersection(*reads.values())
        print(f"\nread by all {len(names)}: {len(everyone)} words")
        for name in names:
            only = reads[name] - set.union(*(v for k, v in reads.items() if k != name))
            sample = ", ".join(sorted(only)[:8])
            print(f"  only {name:<14}{len(only):5d}  e.g. {sample[:90]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tests.bench", description=__doc__)
    parser.add_argument("src", help="PDF to benchmark")
    parser.add_argument("--pages", type=int, default=None, help="only the first N pages")
    parser.add_argument("--dpi", type=int, default=pdf.DEFAULT_DPI)
    parser.add_argument("--engines", nargs="+", default=None,
                        help="default: every installed engine")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--recall", action="store_true",
                        help="compare what each engine reads instead of timing it; "
                             "add 'unlimited' to --engines for baidu/Unlimited-OCR")
    args = parser.parse_args(argv)
    engines = args.engines or ocr.available()
    if args.recall:
        recall(args.src, engines, args.pages, args.dpi)
    else:
        run(args.src, engines, args.pages, args.dpi, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

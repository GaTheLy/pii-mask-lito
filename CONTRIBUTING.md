# Contributing to pii-mask-lito

Thanks for helping improve a privacy-sensitive tool. Please keep changes small,
testable, and free of real personal data.

## Set up

Python 3.10 or newer is required.

```bash
git clone <your fork>
cd pii-mask-service
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m spacy download en_core_web_lg
```

Optional capabilities can be installed when a change requires them:

```bash
pip install -e '.[dev,tesseract]'  # OCR; native Tesseract is also required
pip install -e '.[dev,paddle]'     # Paddle OCR engine
pip install -e '.[dev,rapidocr]'   # ONNX-based OCR engine
pip install -e '.[dev,barcode]'    # barcode decoding
pip install -e '.[dev,vision]'     # visual detectors
```

## Test changes

```bash
pytest -q
python -m tests.corpus.build
python -m tests.corpus.gate
```

Run relevant focused tests while iterating, then run the full suite before
opening a pull request. Optional-dependency tests may skip on a minimal setup;
record relevant skips rather than treating them as coverage.

Detection, rendering, OCR, or policy changes need a regression test using
invented data. If a benchmark is available for the change, include its command,
profile, OCR engine, and before/after result in the pull request. Do not present
one environment-specific result as a universal accuracy claim.

The optional external benchmark runner is intentionally separate from the
repository. Review the source dataset's current license and terms yourself;
do not add a downloaded dataset to a pull request.

## Never commit personal or confidential data

- Do not commit real documents, identifiers, screenshots, masking reports, API
  keys, or customer-derived labels.
- Use clearly invented values in tests and examples. Do not create a fixture by
  editing or renaming a real document.
- Treat output and `--report-values` reports as sensitive. Inspect `git status`
  before committing.
- Use `tmp/` or another ignored local location for scratch artifacts.

If you need to report a missed identifier, create a minimal synthetic
reproducer that preserves only the relevant layout and value shape.

## Pull requests

Explain the problem, the policy impact, and the tests you ran. Update the
README when user-visible behavior, supported formats, policy profiles, or
security caveats change. Keep dependency changes compatible with the Apache-2.0
project license and include the dependency's license information when needed.

## Style

Follow the existing Python style and keep comments focused on durable design
reasons, not on confidential source documents. Prefer a clear false-positive or
false-negative trade-off over an undocumented heuristic.

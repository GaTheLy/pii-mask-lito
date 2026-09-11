# pii-mask-lito

`pii-mask-lito` is a local-first, format-preserving PII masking tool for PDFs,
images, text files, CSV files, DOCX files, and XLSX files. It finds supported
identifiers, replaces them with stable tags, and keeps the surrounding document
layout as intact as practical.

```bash
pii-mask-lito correspondence.pdf -o masked.pdf --report masking-report.json
```

For example, `Contact: Sam Rivera, sam.rivera@example.test` can become
`Contact: <NAME#0> <NAME#1>, <EMAIL#0>`. Tags are stable within one command, so
the same detected value receives the same tag in every input file in that run.

## What it supports

| Input | Output | Primary text source |
| --- | --- | --- |
| PDF with a text layer | PDF | PDF text geometry |
| Scanned or image-only PDF | PDF | OCR |
| PNG, JPEG | same image format | OCR |
| TXT, CSV | same format | text |
| DOCX, XLSX | same format | document or cell text |

The core detector covers common names, locations, email addresses, phone
numbers, dates, government and financial identifiers, URLs, IP addresses, and
contextual identifiers such as account, case, record, and reference IDs. Exact
coverage depends on document layout, available OCR, and the selected policy.

Optional extras add OCR engines, barcode decoding, and visual detection of
faces and signature-like regions. They are not installed by default.

## Install

Python 3.10 or later is required.

```bash
pip install pii-mask-lito

# OCR for scans and standalone images
pip install 'pii-mask-lito[tesseract]'
# Install the native Tesseract program with your operating-system package manager.
```

For a container image with Tesseract and the spaCy model included:

```bash
docker build -t pii-mask-lito .
docker run --rm -v "$PWD:/data" pii-mask-lito input.pdf -o output.pdf
```

## Use

```bash
# Default, domain-neutral policy
pii-mask-lito input.pdf -o output.pdf

# Process several files into a directory and write a value-redacted report
pii-mask-lito one.pdf two.docx -o masked/ --report masking-report.json

# Use a chosen OCR engine for image-only input
pii-mask-lito scan.png -o scan-masked.png --ocr tesseract

# Explicit US-healthcare Safe Harbor-oriented policy
pii-mask-lito input.pdf -o output.pdf --profile hipaa-safe-harbor

# Tune a policy for a known document class
pii-mask-lito input.pdf -o output.pdf --mask-organizations --min-masked-age 0
```

Run `pii-mask-lito --help` for the full option list. The tool verifies rendered
PDF output by default; `--no-verify` disables that safety check and should only
be used in controlled testing.

### Policy profiles

`general` is the default. It is intended as a conservative, domain-neutral
starting point: it masks supported common identifiers, including professionally
identified people, and masks ages from zero onward.

`hipaa-safe-harbor` is an explicit US-healthcare-oriented profile. It masks the
supported identifier categories relevant to that policy and masks ages of 90 or
older by default. It preserves explicitly identified professional people unless
you opt in with `--mask-providers`.

Neither profile is a compliance determination or a guarantee that a document is
de-identified. Organizations must choose and validate a policy appropriate to
their legal, contractual, and operational requirements.

`--entities` replaces the profile entity set entirely. `--mask-organizations`,
`--mask-providers` / `--preserve-professional-identities`, and
`--min-masked-age` make targeted adjustments without changing source code.

## How it works

The pipeline extracts positioned text, detects likely identifiers with pattern,
context, and layout rules, assigns replacement tags, and renders exact-area
masks. Native-text PDFs use their text geometry; scans and images use OCR. PDF
pages that required OCR are rechecked after masking so that visible text is
tested again from the rendered result.

Text reports are safe by default: paths and detected values are redacted.
`--report-values` deliberately includes original values for an audit workflow;
treat such a report as sensitive source data and do not commit or share it.

The optional `--agents` mode can use a local or hosted vision-language model to
suggest additional document context. It never supplies rendering coordinates;
matching and placement remain deterministic. A hosted model sends page images
and OCR-derived content to the selected provider. Do not enable it until that
data transfer is approved for your documents.

## Verification and review

Masking is an error-reduction tool, not a substitute for review. OCR can miss
small, blurred, handwritten, rotated, or unusual text. Rules can miss labels,
languages, identifiers, and layouts they do not recognize. Visual detectors and
barcode decoding require their respective optional extras. Metadata, embedded
attachments, and unsupported file features can also need separate handling.

Before releasing a masked document, review the rendered output using a process
appropriate to its sensitivity. Keep source files, output files, and any
value-bearing reports in an access-controlled workspace.

## Testing and benchmarking

The test suite uses invented fixtures only. Run it with:

```bash
pytest -q
python -m tests.corpus.build
python -m tests.corpus.gate
```

The generated corpus spans customer support, employee administration, and
retail ordering in native and degraded-scan layouts. Its exact ground-truth
boxes drive separate detection, coverage, and over-masking floors in CI.

The repository also provides a benchmark runner for a separately obtained,
publicly described synthetic PDF de-identification dataset:

```bash
python -m tests.benchmark.pdf_deid /path/to/pdf-deid-dataset --level Easy
```

That dataset is not bundled with this project. Confirm its current license,
terms, provenance, and suitability before downloading or redistributing it.
The runner reports whether known synthetic values remain OCR-readable after
masking and a coarse over-masking signal. Results vary with OCR engine,
hardware, source quality, and selected profile; do not treat one run as a
general accuracy claim.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and data-handling rules,
and [SECURITY.md](SECURITY.md) for vulnerability reporting and operational
limits.

## License

Apache-2.0. See [LICENSE](LICENSE).

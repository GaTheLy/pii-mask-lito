# Includes a native OCR engine and the default spaCy model.
FROM python:3.11-slim

# tesseract-ocr is the OCR engine; the rest are wheels' runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY pii_mask_lito ./pii_mask_lito

RUN pip install --no-cache-dir '.[tesseract]' \
    && python -m spacy download en_core_web_lg

# Mount documents here; the image itself stays unchanged.
WORKDIR /data
ENTRYPOINT ["pii_mask_lito"]
CMD ["--help"]

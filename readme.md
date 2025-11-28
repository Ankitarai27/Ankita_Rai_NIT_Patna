# Invoice Line-Item Extractor (FastAPI)

## Overview
This project provides `/extract-bill-data` POST endpoint that accepts JSON `{ "document": "<public-url>" }` and returns structured line-items and reconciled totals.

## How it works
1. Downloads the document.
2. Converts PDF pages to images (or uses image directly).
3. Runs OCR (Tesseract).
4. Extracts line items with rule-based heuristics and deduplicates them.
5. Returns JSON following the required schema.

## Requirements
- Python 3.10+
- Tesseract OCR (system install)
- Poppler (for pdf2image)

## Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

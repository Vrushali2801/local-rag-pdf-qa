import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
from config import RAW_DIR, PROCESSED_DIR, OCR_DPI


def is_scanned_page(page):
    # If the page has no text blocks, it's probably an image-only page.
    text = page.get_text().strip()
    return len(text) == 0


def ocr_page(page):
    # Convert the page to an image for OCR.
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    return text


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        if is_scanned_page(page):
            text = ocr_page(page)
        else:
            text = page.get_text("text")

        cleaned = clean_text(text)
        all_text.append(cleaned)

    return "\n".join(all_text)


def clean_text(text):
    # Basic cleaning so chunking isn’t messy.
    text = text.replace("\t", " ")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def process_all_pdfs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for filename in os.listdir(RAW_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(RAW_DIR, filename)
        print(f"Processing {filename}...")

        text = extract_text_from_pdf(path)

        out_path = os.path.join(PROCESSED_DIR, filename.replace(".pdf", ".txt"))
        with open(out_path, "w", encoding="utf8") as f:
            f.write(text)

        print(f"Saved → {out_path}")


if __name__ == "__main__":
    process_all_pdfs()

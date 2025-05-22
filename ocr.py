import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from vector_db import save_to_chroma
import os

# Configure Tesseract path (if not set in system PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_from_image(image_path):
    """Extracts text from a single image."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.medianBlur(gray_image, 3)
    extracted_text = pytesseract.image_to_string(gray_image)
    return extracted_text

def ocr_from_pdf(pdf_path):
    """Extracts text from each page of a PDF."""
    images = convert_from_path(pdf_path)
    full_text = ""
    for page_num, image in enumerate(images):
        image.save(f'temp_page_{page_num}.png', 'PNG')
        text = ocr_from_image(f'temp_page_{page_num}.png')
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        os.remove(f'temp_page_{page_num}.png')
    return full_text

def process_all_files():
    """Process all images and PDFs in their respective folders."""
    print("🔄 Processing Images...")
    for filename in os.listdir('./image'):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\n📌 Processing Image: {filename}")
            text = ocr_from_image(f'./image/{filename}')
            print("Extracted Text:\n", text)
            # After extracting text:
            doc_id = filename.replace(" ", "_")
            save_to_chroma(doc_id, text)

    print("\n🔄 Processing PDFs...")
    for filename in os.listdir('./pdf'):
        if filename.endswith('.pdf'):
            print(f"\n📌 Processing PDF: {filename}")
            text = ocr_from_pdf(f'./pdf/{filename}')
            print("Extracted Text:\n", text)
            # After extracting text:
            doc_id = filename.replace(" ", "_")
            save_to_chroma(doc_id, text)

if __name__ == "__main__":
    process_all_files()


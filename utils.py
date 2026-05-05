# utils.py
import gc
import torch
import re
import pdfplumber
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from langdetect import detect, LangDetectException

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """استخراج النص مباشرة من الـ PDF (أدق وأسرع من OCR للملفات الرقمية)"""
    import io
    text_pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_pages.append(txt.strip())
    return "\n\n".join(text_pages)

def preprocess_image(image: Image.Image) -> Image.Image:
    """تحسين جودة الصورة قبل الـ OCR لرفع الدقة بشكل ملحوظ"""
    img = image.convert("L")  # تدرج رمادي
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    # Binarization باستخدام OpenCV
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh).convert("RGB")

def detect_language(text: str) -> str:
    try:
        return "ar" if detect(text) == "ar" else "en"
    except LangDetectException:
        return "en"

def clean_text(raw: str) -> str:
    text = re.sub(r"\n\s*\n", "\n", raw)
    text = re.sub(r"[^\w\s\u0600-\u06FF\u0750-\u077F.,;:?!()-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def smart_chunk_text(text: str, max_tokens: int = 600, overlap: int = 50) -> list:
    """تقسيم النص إلى فقرات متداخلة للحفاظ على السياق"""
    sentences = re.split(r'(?<=[.!?؟])\s+', text)
    chunks, current_chunk, current_len = [], [], 0
    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_len = len(current_chunk)
        current_chunk.extend(words)
        current_len += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks if chunks else [text]

def free_gpu_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
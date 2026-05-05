# utils.py
import gc
import torch
import re
from langdetect import detect, LangDetectException

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def detect_language(text: str) -> str:
    """يكشف لغة النص لتوجيهه للموديل المناسب (AR/EN)"""
    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except LangDetectException:
        return "en"

def clean_ocr_output(raw_text: str) -> str:
    """تنظيف ذكي لمخرجات OCR دون الاعتماد على Regex عشوائي"""
    text = re.sub(r"\n\s*\n", "\n", raw_text)               # إزالة الأسطر الفارغة المتكررة
    text = re.sub(r"[^\w\s\u0600-\u06FF\u0750-\u077F.,;:?!()-]", "", text) # إزالة رموز غريبة
    text = re.sub(r"\s+", " ", text).strip()                # توحيد المسافات
    return text

def free_gpu_memory():
    """تفريغ ذاكرة GPU بعد كل مرحلة لمنع CUDA OOM على Colab"""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
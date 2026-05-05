# pipeline.py
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, NougatProcessor
)
from utils import DEVICE, detect_language, clean_ocr_output, free_gpu_memory
from PIL import Image
import streamlit as st

# ================= تحميل الموديلات (Hugging Face Transformers فقط) =================
@st.cache_resource(show_spinner=False)
def load_nougat():
    processor = NougatProcessor.from_pretrained("facebook/nougat-small")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
    model.to(DEVICE)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner=False)
def load_summarizer(lang: str):
    repo = "malmarjeh/mbart-large-50-arabic-summarization" if lang == "ar" else "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_question_generator(lang: str):
    repo = "UBC-NLP/AraT5-v2-base-1024" if lang == "ar" else "mrm8488/t5-base-finetuned-question-generation-ap"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ================= دوال الـ Pipeline =================
def run_ocr(image: Image.Image) -> str:
    processor, model = load_nougat()
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs["pixel_values"],
            max_length=1024,
            num_beams=2,
            early_stopping=True
        )
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    free_gpu_memory()
    return text.strip()

def run_summarization(text: str, lang: str) -> str:
    tokenizer, model = load_summarizer(lang)
    max_input = 1024
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input, truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    free_gpu_memory()
    return summary

def run_question_generation(text: str, lang: str) -> list:
    tokenizer, model = load_question_generator(lang)
    prefix = "generate questions:" if lang == "en" else "ولّد أسئلة:"
    input_text = f"{prefix} {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            num_beams=5,
            num_return_sequences=5,
            early_stopping=True,
            repetition_penalty=1.3
        )
    questions = [tokenizer.decode(q, skip_special_tokens=True).strip() for q in outputs]
    free_gpu_memory()
    return list(dict.fromkeys([q for q in questions if len(q) > 10]))
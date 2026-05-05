# pipeline.py
import torch
import streamlit as st
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, NougatProcessor
)
from utils import (DEVICE, detect_language, clean_text, 
                   smart_chunk_text, free_gpu_memory, preprocess_image)
from PIL import Image

@st.cache_resource(show_spinner=False)
def load_nougat():
    processor = NougatProcessor.from_pretrained("facebook/nougat-small")
    processor.image_processor.do_crop_margin = False
    processor.image_processor.do_align_long_axis = False
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
    model.to(DEVICE)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner=False)
def load_summarizer(lang: str):
    repo = "malmarjeh/mbart-large-50-arabic-summarization" if lang == "ar" else "facebook/bart-large-cnn"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
    mdl.to(DEVICE)
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_qg(lang: str):
    repo = "UBC-NLP/AraT5-v2-base-1024" if lang == "ar" else "valhalla/t5-base-qg-hl"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
    mdl.to(DEVICE)
    mdl.eval()
    return tok, mdl

def run_ocr(image: Image.Image) -> str:
    processor, model = load_nougat()
    image = preprocess_image(image)
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(inputs["pixel_values"], max_length=1024, num_beams=3, early_stopping=True)
    free_gpu_memory()
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

def run_summarization(text: str, lang: str) -> str:
    tok, mdl = load_summarizer(lang)
    chunks = smart_chunk_text(text, max_tokens=500, overlap=40)
    chunk_summaries = []
    
    for chunk in chunks:
        inputs = tok(chunk, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                               max_new_tokens=120, min_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
        chunk_summaries.append(tok.decode(out[0], skip_special_tokens=True))
        free_gpu_memory()
        
    # المرحلة الثانية: تلخيص الملخصات (Map-Reduce)
    combined = " ".join(chunk_summaries)
    inputs = tok(combined, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
    with torch.no_grad():
        final_out = mdl.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                 max_new_tokens=200, min_length=60, num_beams=5, length_penalty=2.5, early_stopping=True)
    free_gpu_memory()
    return tok.decode(final_out[0], skip_special_tokens=True)

def run_question_generation(text: str, lang: str, q_type: str = "theoretical") -> list:
    tok, mdl = load_qg(lang)
    chunks = smart_chunk_text(text, max_tokens=400, overlap=30)
    questions = []
    
    prefix_q = "generate question:" if lang == "en" else "ولّد سؤال:"
    prefix_mcq = "generate multiple choice question with answer and 3 wrong options:" if lang == "en" else "ولّد سؤال اختيار من متعدد مع الإجابة و3 خيارات خاطئة:"
    
    prompt = prefix_mcq if q_type == "mcq" else prefix_q
    
    for chunk in chunks[:3]:  # نأخذ أول 3 أجزاء لتجنب التكرار والبطء
        input_text = f"{prompt} {chunk}"
        inputs = tok(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                               max_new_tokens=150, num_beams=4, num_return_sequences=2, early_stopping=True, repetition_penalty=1.3)
        for seq in out:
            q_text = tok.decode(seq, skip_special_tokens=True).strip()
            if len(q_text) > 15:
                questions.append(q_text)
        free_gpu_memory()
        
    return list(dict.fromkeys(questions))  # إزالة التكرار مع الحفاظ على الترتيب
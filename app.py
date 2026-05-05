# app.py
import streamlit as st
from PIL import Image
from pipeline import run_ocr, run_summarization, run_question_generation
from utils import clean_ocr_output, detect_language

st.set_page_config(page_title="AI Study Companion", layout="wide", page_icon="📚")

st.markdown("""
<style>
    .main-header {font-size: 2.4rem; font-weight: 700; color: #1F4E79; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.05rem; color: #5D6D7E; text-align: center; margin-bottom: 2rem;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: 600; background-color: #2980B9; color: white;}
    .result-box {background: #F4F6F7; padding: 1rem; border-radius: 8px; border-left: 4px solid #2980B9; margin-top: 1rem;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 AI Study Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">حوّل صور الكتب إلى ملخصات وأسئلة ذكية باستخدام Hugging Face Transformers</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 ارفع صورة الصفحة (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="الصفحة المرفوعة", use_column_width=True)
    with col2:
        if st.button("▶️ بدء المعالجة", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            try:
                status.text("🔍 جاري استخراج النص (Nougat OCR)...")
                progress.progress(20)
                raw_text = run_ocr(image)
                
                status.text("🧹 جاري تنظيف النص وكشف اللغة...")
                progress.progress(40)
                clean_txt = clean_ocr_output(raw_text)
                lang = detect_language(clean_txt)
                
                status.text(f"📑 جاري التلخيص ({'عربي' if lang=='ar' else 'English'})...")
                progress.progress(60)
                summary = run_summarization(clean_txt, lang)
                
                status.text("❓ جاري توليد أسئلة المراجعة...")
                progress.progress(80)
                questions = run_question_generation(clean_txt, lang)
                
                progress.progress(100)
                status.text("✅ اكتملت المعالجة بنجاح!")
                
                st.session_state.update({
                    "clean": clean_txt, "summary": summary,
                    "questions": questions, "lang": lang
                })
            except Exception as e:
                st.error(f"⚠️ حدث خطأ: {str(e)}")
                st.stop()

if "clean" in st.session_state:
    tab1, tab2, tab3 = st.tabs(["📝 النص المستخرج", "📋 الملخص الذكي", "❓ أسئلة المراجعة"])
    with tab1:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.text_area("النص بعد التنظيف", st.session_state["clean"], height=300)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"🌍 اللغة المكتشفة: {'العربية' if st.session_state['lang']=='ar' else 'الإنجليزية'}")
        st.markdown(st.session_state["summary"])
        st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        if st.session_state["questions"]:
            for i, q in enumerate(st.session_state["questions"], 1):
                st.markdown(f"**س{i}:** {q}")
        else:
            st.warning("لم يتم توليد أسئلة كافية. جرّب صفحة تحتوي على فقرات أوضح.")
        st.markdown('</div>', unsafe_allow_html=True)
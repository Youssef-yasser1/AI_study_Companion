# app.py
import streamlit as st
from PIL import Image
from pipeline import run_ocr, run_summarization, run_question_generation
from utils import clean_text, detect_language, extract_text_from_pdf

st.set_page_config(page_title="AI Study Companion V2", layout="wide", page_icon="📚")

st.markdown("""
<style>
    .main-header {font-size: 2.4rem; font-weight: 700; color: #1F4E79; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.05rem; color: #5D6D7E; text-align: center; margin-bottom: 2rem;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: 600; background-color: #2980B9; color: white;}
    .result-box {background: #F4F6F7; padding: 1rem; border-radius: 8px; border-left: 4px solid #2980B9; margin-top: 1rem;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 AI Study Companion V2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">PDF + صور | تلخيص أكاديمي | أسئلة MCQ أو نظرية | Hugging Face Transformers</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 ارفع ملف المحاضرة (PDF أو صورة PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])
q_type = st.radio("📝 نوع الأسئلة المطلوبة:", ["نظري (Theoretical)", "اختيار من متعدد (MCQ)"], horizontal=True)

if uploaded_file:
    is_pdf = uploaded_file.name.lower().endswith(".pdf")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if is_pdf:
            st.success(f"📄 تم رفع ملف PDF: {uploaded_file.name}")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="الصفحة المرفوعة", use_column_width=True)
            
    with col2:
        if st.button("▶️ بدء المعالجة الذكية", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            try:
                # 1. استخراج النص
                status.text("📖 جاري استخراج النص...")
                progress.progress(15)
                if is_pdf:
                    raw_text = extract_text_from_pdf(uploaded_file.getvalue())
                else:
                    raw_text = run_ocr(image)
                    
                if not raw_text.strip():
                    st.error("⚠️ لم يتم استخراج نص. تأكد أن الملف يحتوي على نص واضح أو جرب صورة أعلى دقة.")
                    st.stop()
                    
                # 2. التنظيف وكشف اللغة
                status.text("🧹 تنظيف النص وتحليل اللغة...")
                progress.progress(35)
                clean_txt = clean_text(raw_text)
                lang = detect_language(clean_txt)
                
                # 3. التلخيص الأكاديمي
                status.text(f"📑 جاري التلخيص الأكاديمي ({'عربي' if lang=='ar' else 'English'})...")
                progress.progress(60)
                summary = run_summarization(clean_txt, lang)
                
                # 4. توليد الأسئلة
                q_mode = "mcq" if "MCQ" in q_type else "theoretical"
                status.text(f"❓ جاري توليد أسئلة {q_type}...")
                progress.progress(85)
                questions = run_question_generation(clean_txt, lang, q_type=q_mode)
                
                progress.progress(100)
                status.text("✅ اكتملت المعالجة بنجاح!")
                
                st.session_state.update({
                    "clean": clean_txt, "summary": summary,
                    "questions": questions, "lang": lang, "q_type": q_type
                })
            except Exception as e:
                st.error(f"⚠️ حدث خطأ: {str(e)}")
                st.stop()

if "clean" in st.session_state:
    tab1, tab2, tab3 = st.tabs(["📝 النص المستخرج", "📋 الملخص الأكاديمي", f"❓ أسئلة {st.session_state['q_type']}"])
    with tab1:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.text_area("النص المعالج", st.session_state["clean"], height=300)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"🌍 اللغة: {'العربية' if st.session_state['lang']=='ar' else 'الإنجليزية'}")
        st.markdown(st.session_state["summary"])
        st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        if st.session_state["questions"]:
            for i, q in enumerate(st.session_state["questions"], 1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.warning("لم يتم توليد أسئلة كافية. جرّب ملفاً يحتوي على فقرات أطول أو أوضح.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # زر تصدير سريع
    import json
    export_data = {
        "language": st.session_state["lang"],
        "summary": st.session_state["summary"],
        "questions": st.session_state["questions"]
    }
    st.download_button("📥 تصدير النتائج (JSON)", json.dumps(export_data, ensure_ascii=False, indent=2), 
                       file_name="study_companion_results.json", mime="application/json")
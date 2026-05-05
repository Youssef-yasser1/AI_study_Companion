# 📚 AI Study Companion
نظام ذكي لتحويل صور الكتب والمراجع إلى نصوص رقمية، ملخصات مركزة، وأسئلة مراجعة تفاعلية باستخدام نماذج Hugging Face Transformers.

## 🛠️ التقنيات المستخدمة
- **OCR**: `facebook/nougat-small` (VisionEncoderDecoderModel)
- **Summarization**: `facebook/bart-large-cnn` (EN) / `malmarjeh/mbart-large-50-arabic-summarization` (AR)
- **Question Generation**: `mrm8488/t5-base-finetuned-question-generation-ap` (EN) / `UBC-NLP/AraT5-v2-base-1024` (AR)
- **Framework**: Hugging Face `transformers`, PyTorch, Streamlit
- **Language Support**: Mixed (Arabic / English) with automatic routing

## 📦 التثبيت والتشغيل
```bash
git clone <repo-url>
cd AI_Study_Companion
pip install -r requirements.txt
streamlit run app.py

import streamlit as st
import joblib
import re
from hazm import Normalizer, WordTokenizer, Lemmatizer


# ---------------- UI Theme ----------------
st.set_page_config(page_title="SnappFood Sentiment Model", layout="centered")

st.markdown("""
<style>
/* Font + RTL */
html, body, div, h1, h2, h3, p, label {
    direction: rtl;
    text-align: center !important;
    font-family: "IRANSans", sans-serif;
}
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff7ec 0%, #ffe0d9 100%);
}
/* Titles */
h1, h2, h3 {
    color: #ff3b30;
    font-weight: bold;
}
/* Center button */
.stButton > button {
    background-color: #ff3b30;
    color: white;
    font-size: 18px;
    padding: 8px 25px;
    border-radius: 10px;
    display: block;
    margin: auto;
    cursor: pointer;
}
/* Text Area */
textarea {
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)



# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("models/model_logistic.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()



# ---------------- Text Preprocessing ----------------
normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()

def preprocess(text):
    text = normalizer.normalize(text)
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)



# ---------------- Model Prediction ----------------
def predict_sentiment(text, threshold=0.5):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])

    negative_prob = model.predict_proba(vec)[0][1]

    if negative_prob > threshold:
        return "negative", negative_prob
    else:
        return "positive", 1 - negative_prob



# ---------------- UI ----------------
st.title("🍽️ تحلیل احساسات نظرات اسنپ‌فود")

st.write("""
✨ با کمک **پردازش زبان طبیعی** و **یادگیری ماشین**  
این ابزار بررسی می‌کنه آیا متن شما **مثبت**ه یا **منفی** 👇
""")

text_input = st.text_area("✍️ متن نظر:", placeholder="مثال: غذا عالی بود ولی خیلی دیر رسید!")

if st.button("🔍 تحلیل کن"):

    if text_input.strip() == "":
        st.warning("⚠ لطفاً یک جمله وارد کنید.")

    else:
        label, confidence = predict_sentiment(text_input)

        if label == "negative":
            box_color = "#e63946"
            emoji = "😡"
            msg = "نظر منفی ❌"
        else:
            box_color = "#20c997"
            emoji = "😎"
            msg = "نظر مثبت ✨"

        # Result UI
        st.markdown(
            f"""
            <div style="padding:15px; margin-top:15px;
                        border-radius:12px; background-color:{box_color};
                        color:white; font-size:22px; font-weight:bold;">
                {emoji} {msg}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence Display
        st.write("🎯 **درجه اطمینان مدل:**")
        st.progress(confidence)
        st.info(f"📌 مدل با **{confidence*100:.2f}%** اطمینان این نتیجه را اعلام کرده است.")

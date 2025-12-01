import streamlit as st
import joblib
import re
from hazm import Normalizer, WordTokenizer, Lemmatizer


# ---------------- UI Styling ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff7ec 0%, #ffe0d9 100%);
}

h1, h2, h3, .stTextInput label {
    color: #ff3b30;
}

.stButton button {
    background-color: #ff3b30;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Load Model & Vectorizer ----------------
model = joblib.load("models/model_logistic.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# ---------------- Text Preprocessing ----------------
normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()

def preprocess(text):
    text = normalizer.normalize(text)
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    token = tokenizer.tokenize(text)
    token = [lemmatizer.lemmatize(word) for word in token]
    return " ".join(token)


# ---------------- Prediction Function ----------------
def predict_sentiment(text, threshold=0.5):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])

    prob = model.predict_proba(vec)[0][1]  # probability of negative class
    label = 1 if prob > threshold else 0

    if label == 1:
        return "❌ نظر منفی", prob
    else:
        return "✨ نظر مثبت", prob


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Persian SnappFood Sentiment Model", layout="centered")

st.title("🍽️ تحلیل احساسات نظرات اسنپ‌فود")

st.write("""
✨ این ابزار با استفاده از **پردازش زبان طبیعی** و مدل **یادگیری ماشین**
به شما کمک می‌کنه بفهمید یک نظر **مثبت** بوده یا **منفی**.

🔍 فقط یک جمله وارد کن و روی دکمه تحلیل بزن 👇
        """)



text_input = st.text_area("✍️ یک جمله بنویس...", placeholder="مثال: غذا عالی بود اما دیر رسید")

if st.button("تحلیل کن 🔍"):

    if text_input.strip() == "":
        st.warning("⚠ لطفا یک جمله وارد کن!")
    else:
        sentiment, prob = predict_sentiment(text_input)

        st.subheader("🔎 نتیجه مدل:")

        if "منفی" in sentiment:
            st.error(f"{sentiment}  | احتمال: {prob*100:.2f}%")
        else:
            st.success(f"{sentiment}  | احتمال: {prob*100:.2f}%")

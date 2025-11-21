import streamlit as st
import pandas as pd
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===============================
# Fungsi Preprocessing
# ===============================
def preprocess_text(text, stopword, stemmer):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

def get_confidence_badge(prob):
    if prob >= 90:
        return "Sangat Tinggi", "🟢"
    elif prob >= 75:
        return "Tinggi", "🟡"
    elif prob >= 60:
        return "Sedang", "🟠"
    else:
        return "Rendah", "🔴"

# ===============================
# Load Model & Tools
# ===============================
@st.cache_resource
def load_all():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("bnb_model.pkl", "rb") as f:
        model_bnb = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        model_svm = pickle.load(f)
    with open("ensemble_model.pkl", "rb") as f:
        model_ensemble = pickle.load(f)
    
    tool = {
        "stopword": StopWordRemoverFactory().create_stop_word_remover(),
        "stemmer": StemmerFactory().create_stemmer()
    }
    return vectorizer, model_bnb, model_svm, model_ensemble, tool

vectorizer, model_bnb, model_svm, model_ensemble, tools = load_all()

# ===============================
# UI STREAMLIT
# ===============================
st.title("📊 Analisis Sentimen Ulasan Restoran")

input_text = st.text_area("Masukkan teks ulasan:")
predict_btn = st.button("Prediksi")

# ===============================
# LOGIKA PREDIKSI (SUDAH DIBENARKAN)
# ===============================
if predict_btn:
    if input_text.strip() == "":
        st.warning("⚠️ Masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            try:
                stopword_remover = tools['stopword']
                stemmer = tools['stemmer']
                processed = preprocess_text(input_text, stopword_remover, stemmer)
                vec = vectorizer.transform([processed])

                pred_bnb = model_bnb.predict(vec)[0]
                pred_svm = model_svm.predict(vec)[0]
                pred_ensemble = model_ensemble.predict(vec)[0]

                prob_bnb = model_bnb.predict_proba(vec)[0]
                prob_svm = model_svm.predict_proba(vec)[0]
                prob_ensemble = model_ensemble.predict_proba(vec)[0]

            except Exception as e:
                st.error(f"Terjadi error saat memproses: {e}")
                st.stop()

        st.subheader("🎯 Hasil Analisis (Model Ensemble)")

        max_prob = max(prob_ensemble) * 100
        conf_text, conf_icon = get_confidence_badge(max_prob)

        if pred_ensemble == "positive":
            st.success("### ✅ Sentimen: POSITIF")
        else:
            st.error("### ❌ Sentimen: NEGATIF")

        st.info(f"**Tingkat Keyakinan:** {conf_icon} {conf_text} ({max_prob:.1f}%)")

        # ===========================
        # Probabilitas (Satu-satunya blok yang benar)
        # ===========================
        st.write("**📊 Probabilitas:**")
        colA, colB = st.columns(2)

        with colA:
            st.metric("Negatif", f"{prob_ensemble[0]*100:.1f}%")

        with colB:
            st.metric("Positif", f"{prob_ensemble[1]*100:.1f}%")

# Tidak ada blok duplikat di bawah ini!

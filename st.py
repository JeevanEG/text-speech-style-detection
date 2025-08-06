import os
import io
import json
import time
import pickle
import numpy as np
import pandas as pd
import requests
import sounddevice as sd
import spacy
import tensorflow as tf
from collections import Counter
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import streamlit as st

# Streamlit config
st.set_page_config(page_title="Speech Style Analyzer", page_icon="🎤", layout="wide")

# Disable file watch reloads
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = ""

# Constants
DEEPGRAM_API_KEY = "882c145300e121a320938ed34451334eebbd6308"
FILLERS = {"uh", "um", "er", "ah", "like", "you know", "so", "well"}
FUNCTION_POS = {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART", "INTJ"}

# Session state setup
for var in ["audio_data", "transcript", "features", "prediction", "confidence"]:
    if var not in st.session_state:
        st.session_state[var] = None

@st.cache_resource
def load_nlp_model():
    return spacy.load('en_core_web_sm')

def get_tree_depth(token):
    if not list(token.children):
        return 1
    return 1 + max(get_tree_depth(child) for child in token.children)

def extract_features(text, nlp):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sents = list(doc.sents)
    filler_counts = sum(word in FILLERS for word in words)
    filler_per_100w = filler_counts / (len(words) / 100) if words else 0
    sent_lens = [len([tok for tok in sent if tok.is_alpha]) for sent in sents]
    avg_sent_len = sum(sent_lens) / len(sent_lens) if sent_lens else 0
    std_sent_len = np.std(sent_lens) if len(sent_lens) > 1 else 0
    content_words = [tok for tok in doc if tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and tok.is_alpha]
    lexical_density = len(content_words) / len(words) if words else 0
    ttr = len(set(words)) / len(words) if words else 0
    function_words = [tok for tok in doc if tok.pos_ in FUNCTION_POS and tok.is_alpha]
    pos_ratio = len(content_words) / len(function_words) if function_words else 0
    word_freq = Counter(words)
    rep_count = sum(val for word, val in word_freq.items() if val > 1)
    parse_depths = [get_tree_depth(sent.root) for sent in sents if len(sent) > 0]
    max_parse_depth = max(parse_depths) if parse_depths else 0
    mean_parse_depth = np.mean(parse_depths) if parse_depths else 0
    return {
        "filler_per_100w": filler_per_100w,
        "avg_sent_len": avg_sent_len,
        "std_sent_len": std_sent_len,
        "lexical_density": lexical_density,
        "ttr": ttr,
        "content_function_ratio": pos_ratio,
        "word_repetitions": rep_count,
        "max_parse_depth": max_parse_depth,
        "mean_parse_depth": mean_parse_depth
    }

def load_scaler_with_validation():
    try:
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            if os.path.exists('scaler_stats.json'):
                with open('scaler_stats.json', 'r') as f:
                    stats = json.load(f)
                if hasattr(scaler, 'mean_') and np.allclose(scaler.mean_, stats['mean'], rtol=1e-5):
                    return scaler, "✅ Original training scaler loaded."
            return scaler, "⚠️ Scaler loaded (not validated)."
        else:
            return None, "❌ Scaler file missing."
    except Exception as e:
        return None, f"❌ Failed to load scaler: {e}"

def transcribe_audio_deepgram(audio_data):
    try:
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        params = {
            "filler_words": "true",
            "punctuate": "true",
            "language": "en-US"
        }

        if isinstance(audio_data, np.ndarray):
            buffer = io.BytesIO()
            write(buffer, 16000, (audio_data * 32767).astype(np.int16))
            audio_bytes = buffer.getvalue()
        else:
            audio_bytes = audio_data.read()

        response = requests.post(url, headers=headers, params=params, data=audio_bytes)
        if response.status_code == 200:
            result = response.json()
            return result["results"]["channels"][0]["alternatives"][0]["transcript"]
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def predict_speech_style(text, model, scaler, nlp):
    features = extract_features(text, nlp)
    features_df = pd.DataFrame([features])
    X_scaled = scaler.transform(features_df)
    X = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    prediction = model.predict(X)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]
    label = "Spontaneous Speech" if predicted_class == 1 else "Read Speech"
    return label, confidence, features

def record_audio(duration, sample_rate=16000):
    st.info(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True)
    st.success("✅ Recording complete!")
    return audio.flatten()

def convert_numpy_to_wav_bytes(audio_array, sample_rate=16000):
    buffer = io.BytesIO()
    write(buffer, sample_rate, (audio_array * 32767).astype(np.int16))
    return buffer.getvalue()

def main():
    st.title("🎤 Speech Style Analyzer")

    nlp = load_nlp_model()
    scaler, scaler_status = load_scaler_with_validation()

    st.subheader("📊 Model Status")
    st.info(scaler_status)

    if not os.path.exists("speech_style_lstm_model.h5"):
        st.error("Model file 'speech_style_lstm_model.h5' not found.")
        return
    else:
        model = load_model("speech_style_lstm_model.h5")

    st.subheader("🎵 Audio Input")
    method = st.radio("Select Input Method", ["📁 Upload", "🎙️ Record"])

    if method == "📁 Upload":
        uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac", "m4a"])
        if uploaded:
            st.audio(uploaded)
            st.session_state.audio_data = uploaded

    elif method == "🎙️ Record":
        duration = st.slider("Duration (sec)", 5, 30, 10)
        if st.button("🔴 Record"):
            audio = record_audio(duration)
            st.audio(convert_numpy_to_wav_bytes(audio))
            st.session_state.audio_data = audio

    if st.session_state.audio_data is not None and st.button("🎯 Analyze Speech Style"):
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio_deepgram(st.session_state.audio_data)
            if transcript:
                st.session_state.transcript = transcript
            else:
                st.error("❌ Transcription failed.")
                return

        st.subheader("📝 Transcript")
        st.text_area("Text", st.session_state.transcript, height=150)

        with st.spinner("Analyzing..."):
            prediction, confidence, features = predict_speech_style(st.session_state.transcript, model, scaler, nlp)
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.features = features

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Prediction")
            st.success(f"{prediction} ({confidence:.2%})")
        with col2:
            st.subheader("📊 Features")
            for key, val in features.items():
                st.metric(key.replace("_", " ").title(), f"{val:.2f}" if isinstance(val, float) else val)

if __name__ == "__main__":
    main()

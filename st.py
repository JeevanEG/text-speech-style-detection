# app.py
"""
Interview Integrity Monitor - Streamlit app

Features:
- Continuous monitoring loop (configurable interval)
- Deepgram transcription (api key entered in sidebar)
- Linguistic feature extraction (spaCy) + LSTM classifier (Keras)
- Results log (table), downloadable CSV, live read-confidence chart
- Visual alerts for repeated high-confidence "Read Speech"
- Robust error handling & stateful session management
"""

import os
import io
import json
import time
import math
import pickle
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import sounddevice as sd
from scipy.io.wavfile import write

import streamlit as st
import spacy
from collections import Counter

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ------------------------------
# --------------- CONFIGURATION
# ------------------------------
SAMPLE_RATE = 16000  # Hz
DEFAULT_INTERVAL = 30  # seconds default recording chunk
MIN_INTERVAL = 15
MAX_INTERVAL = 60

# Filler words & POS sets used by feature extractor
FILLERS = {"uh", "um", "er", "ah", "like", "you know", "so", "well"}
FUNCTION_POS = {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART", "INTJ"}

# ------------------------------
# --------------- SESSION STATE
# ------------------------------
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False
if "processing_chunk" not in st.session_state:
    st.session_state.processing_chunk = False
if "results_history" not in st.session_state:
    st.session_state.results_history = []  # list of dicts
if "chunk_counter" not in st.session_state:
    st.session_state.chunk_counter = 0
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "nlp" not in st.session_state:
    st.session_state.nlp = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "alert_state" not in st.session_state:
    st.session_state.alert_state = {"level": "normal", "message": ""}

# ------------------------------
# --------------- HELPERS
# ------------------------------
@st.cache_resource
def load_spacy_model():
    # load spacy once per session
    return spacy.load("en_core_web_sm")

def get_tree_depth(token):
    """Recursively compute dependency-tree depth for a token."""
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(get_tree_depth(child) for child in children)

def extract_features(text, nlp):
    """
    Extract a set of lightweight linguistic features from transcript text.
    Returns a dict of numeric features used by the LSTM (same keys expected by scaler).
    """
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sents = list(doc.sents)

    filler_counts = sum(1 for w in words if w in FILLERS)
    filler_per_100w = filler_counts / (len(words) / 100) if words else 0

    sent_lens = [len([tok for tok in sent if tok.is_alpha]) for sent in sents]
    avg_sent_len = (sum(sent_lens) / len(sent_lens)) if sent_lens else 0
    std_sent_len = np.std(sent_lens) if len(sent_lens) > 1 else 0

    content_words = [tok for tok in doc if tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and tok.is_alpha]
    lexical_density = len(content_words) / len(words) if words else 0

    ttr = len(set(words)) / len(words) if words else 0

    function_words = [tok for tok in doc if tok.pos_ in FUNCTION_POS and tok.is_alpha]
    pos_ratio = (len(content_words) / len(function_words)) if function_words else 0

    word_freq = Counter(words)
    rep_count = sum(val for word, val in word_freq.items() if val > 1)

    parse_depths = [get_tree_depth(sent.root) for sent in sents if len(sent) > 0]
    max_parse_depth = max(parse_depths) if parse_depths else 0
    mean_parse_depth = float(np.mean(parse_depths)) if parse_depths else 0

    return {
        "filler_per_100w": float(filler_per_100w),
        "avg_sent_len": float(avg_sent_len),
        "std_sent_len": float(std_sent_len),
        "lexical_density": float(lexical_density),
        "ttr": float(ttr),
        "content_function_ratio": float(pos_ratio),
        "word_repetitions": float(rep_count),
        "max_parse_depth": float(max_parse_depth),
        "mean_parse_depth": float(mean_parse_depth),
    }

def load_scaler_with_validation():
    """
    Loads scaler.pkl and optionally validates with scaler_stats.json.
    Returns (scaler_or_None, status_message).
    """
    try:
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            if os.path.exists("scaler_stats.json"):
                with open("scaler_stats.json", "r") as f:
                    stats = json.load(f)
                if hasattr(scaler, "mean_") and np.allclose(scaler.mean_, stats.get("mean", scaler.mean_), rtol=1e-5):
                    return scaler, "✅ Training scaler loaded & validated."
                else:
                    return scaler, "⚠️ Scaler loaded (stats mismatch)."
            return scaler, "✅ Scaler loaded."
        else:
            return None, "❌ scaler.pkl missing. Feature normalization will fail."
    except Exception as e:
        return None, f"❌ Failed to load scaler: {e}"

def convert_numpy_to_wav_bytes(audio_array, sample_rate=SAMPLE_RATE):
    """
    Convert float numpy audio (-1..1) to WAV bytes (16-bit).
    """
    buffer = io.BytesIO()
    int16 = (audio_array * 32767).astype(np.int16)
    write(buffer, sample_rate, int16)
    buffer.seek(0)
    return buffer.read()

def record_audio_block(duration, sample_rate=SAMPLE_RATE):
    """
    Record audio via sounddevice for `duration` seconds. Returns numpy float32 array (-1..1).
    Raises exceptions on audio device errors.
    """
    # sounddevice returns float32 in range -1..1 by default
    try:
        frames = int(duration * sample_rate)
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", blocking=True)
        return audio.flatten()
    except Exception as e:
        raise RuntimeError(f"Audio recording failed: {e}")

def transcribe_audio_deepgram(audio_bytes, api_key):
    """
    Send wav bytes to Deepgram listen v1 endpoint. Returns transcript string on success.
    Raises RuntimeError on network/api issues, including 401 for invalid key.
    """
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav"
    }
    params = {
        "filler_words": "true",
        "punctuate": "true",
        "language": "en-US"
    }
    try:
        resp = requests.post(url, headers=headers, params=params, data=audio_bytes, timeout=30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error during transcription: {e}")

    if resp.status_code == 401:
        raise RuntimeError("Deepgram API error: Invalid or unauthorized API key (401).")
    if resp.status_code >= 400:
        raise RuntimeError(f"Deepgram API error (status {resp.status_code}): {resp.text[:300]}")

    try:
        result = resp.json()
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript
    except Exception as e:
        raise RuntimeError(f"Failed to parse Deepgram response: {e}")

def predict_speech_style(text, model, scaler, nlp):
    """
    Extract features, scale, reshape for LSTM input, and return label, confidence, and features dict.
    Expects binary classification with model output shape (1,2) with softmax.
    """
    # guard against empty transcript
    if not text or not text.strip():
        return "No Speech Detected", 0.0, {}

    features = extract_features(text, nlp)
    features_df = pd.DataFrame([features])

    if scaler is None:
        raise RuntimeError("Scaler not available - cannot scale features.")

    try:
        X_scaled = scaler.transform(features_df)  # shape (1, n_features)
    except Exception as e:
        raise RuntimeError(f"Scaler transform failed: {e}")

    # reshape for LSTM: (batch, time_steps, features). We use time_steps=1
    X = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    preds = model.predict(X, verbose=0)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds, axis=1)[0])

    label = "Spontaneous Speech" if predicted_class == 1 else "Read Speech"
    return label, confidence, features

def append_result(transcript, prediction, confidence, features):
    """
    Append a result row to session_state.results_history
    """
    st.session_state.chunk_counter += 1
    row = {
        "Chunk #": st.session_state.chunk_counter,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction": prediction,
        "Confidence": float(confidence),
        "Transcript": transcript,
    }
    # attach features separately if needed
    row.update({f: features.get(f, None) for f in [
        "filler_per_100w","avg_sent_len","std_sent_len","lexical_density","ttr",
        "content_function_ratio","word_repetitions","max_parse_depth","mean_parse_depth"
    ]})
    st.session_state.results_history.append(row)

def check_alerts():
    """
    Check for consecutive high-confidence 'Read Speech' and update alert_state.
    Example rule: If two consecutive chunks are Read Speech with confidence > 0.8 => High Alert.
    """
    hist = st.session_state.results_history
    if len(hist) >= 2:
        last = hist[-1]
        prev = hist[-2]
        if last["Prediction"] == "Read Speech" and prev["Prediction"] == "Read Speech" and last["Confidence"] > 0.8 and prev["Confidence"] > 0.8:
            st.session_state.alert_state = {"level": "high", "message": "🔴 High Alert: Reading detected (2 consecutive high-confidence Read Speech)."}
            return
    # fallback normal
    st.session_state.alert_state = {"level": "normal", "message": "🟢 Monitoring nominal."}

# ------------------------------
# --------------- UI: SIDEBAR
# ------------------------------
st.set_page_config(page_title="Interview Integrity Monitor", page_icon="🔎🎤", layout="wide")
st.title("Interview Integrity Monitor — Read vs Spontaneous (Real-time)")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Deepgram API Key", type="password", help="Enter your Deepgram API key (not stored to disk).")
    interval = st.slider("Recording interval (seconds)", MIN_INTERVAL, MAX_INTERVAL, DEFAULT_INTERVAL)
    st.markdown("---")
    st.write("Model & scaler files must be present in the app directory:")
    st.write("- `speech_style_lstm_model.h5` (Keras LSTM model)")
    st.write("- `scaler.pkl` (scikit-learn StandardScaler)")
    st.write("- Optional: `scaler_stats.json` for validation")
    st.markdown("---")
    st.write("Notes:")
    st.write("• Use 'Start Monitoring' to begin continuous chunked recording and analysis.")
    st.write("• After stopping, you can download the session CSV.")

# ------------------------------
# --------------- LOAD MODELS (once)
# ------------------------------
model_load_col = st.container()
with model_load_col:
    if not st.session_state.model_loaded:
        st.info("Loading spaCy & model/scaler... This happens once per session.")
        try:
            # load spacy
            st.session_state.nlp = load_spacy_model()
            # load scaler
            scaler, scaler_status = load_scaler_with_validation()
            st.session_state.scaler = scaler
            # load keras model
            if os.path.exists("speech_style_lstm_model.h5"):
                st.session_state.model = load_model("speech_style_lstm_model.h5")
                st.session_state.model_loaded = True
                st.success("Model & resources loaded.")
            else:
                st.error("Model file `speech_style_lstm_model.h5` not found in working directory.")
                st.session_state.model_loaded = False
            # show scaler status
            st.info(scaler_status)
        except Exception as e:
            st.error(f"Failed to initialize models: {e}")
            st.session_state.model_loaded = False

# ------------------------------
# --------------- MAIN CONTROLS
# ------------------------------
controls = st.container()
with controls:
    col1, col2 = st.columns([1, 1])
    with col1:
        start_clicked = st.button("▶️ Start Monitoring", disabled=st.session_state.monitoring_active or not st.session_state.model_loaded or not api_key)
    with col2:
        stop_clicked = st.button("⏹️ Stop Monitoring", disabled=not st.session_state.monitoring_active)

    # start/stop logic
    if start_clicked:
        st.session_state.monitoring_active = True
        st.session_state.processing_chunk = False  # allow first chunk
        st.success("Monitoring started.")
        # reset counters/history for a fresh session if empty
        # (if you want to append across sessions, remove clearing)
        st.session_state.results_history = []
        st.session_state.chunk_counter = 0
        st.session_state.alert_state = {"level": "normal", "message": "🟡 Monitoring started."}
        # rerun to update UI
        st.experimental_rerun()

    if stop_clicked:
        st.session_state.monitoring_active = False
        st.session_state.processing_chunk = False
        st.success("Monitoring stopped.")
        # After stopping, leave data in results_history, allow download
        # No automatic rerun necessary — UI updates below.

# ------------------------------
# --------------- LIVE DISPLAY AREAS
# ------------------------------
left_col, right_col = st.columns([3, 2])

# left column: results table + transcript area
with left_col:
    st.subheader("Results Log")
    log_placeholder = st.empty()
    # show DataFrame of history
    if st.session_state.results_history:
        df = pd.DataFrame(st.session_state.results_history).sort_values("Chunk #")
        # show with st.dataframe for interactivity
        log_placeholder.dataframe(df, use_container_width=True)
    else:
        log_placeholder.info("No chunks yet. Start monitoring to see results.")

    st.subheader("Transcript / Details")
    details_placeholder = st.empty()

# right column: chart + alert + stats
with right_col:
    st.subheader("Session Summary")
    alert_box = st.empty()
    chart_box = st.empty()
    controls_box = st.empty()

# ------------------------------
# --------------- MONITORING LOOP (single-chunk-per-run pattern)
# ------------------------------
# Important pattern:
# - We process exactly one chunk per Streamlit run when monitoring_active==True.
# - After finishing a chunk, we call st.experimental_rerun() so the UI can react to button clicks (Start/Stop).
# - Recording itself blocks for `interval` seconds (that's the "real-time" chunking).
if st.session_state.monitoring_active:
    # disable Start button by UI state (done above)
    st.session_state.processing_chunk = True

    # show a small status box about active monitoring
    st.info(f"Monitoring active — recording chunks every {interval} s. Press ⏹️ Stop to end session.")

    try:
        # RECORD CHUNK
        with st.spinner(f"Recording chunk #{st.session_state.chunk_counter + 1} for {interval} seconds..."):
            try:
                audio_np = record_audio_block(interval, SAMPLE_RATE)
            except Exception as e:
                # fatal recording error: stop monitoring
                st.error(f"Recording failed: {e}")
                st.session_state.monitoring_active = False
                st.session_state.processing_chunk = False
                st.experimental_rerun()

        # convert to wav bytes for Deepgram
        try:
            wav_bytes = convert_numpy_to_wav_bytes(audio_np, SAMPLE_RATE)
        except Exception as e:
            st.error(f"Audio conversion failed: {e}")
            st.session_state.processing_chunk = False
            st.experimental_rerun()

        # TRANSCRIBE
        transcript = ""
        try:
            if not api_key:
                raise RuntimeError("No Deepgram API key provided in sidebar.")
            transcript = transcribe_audio_deepgram(wav_bytes, api_key)
            if transcript is None:
                transcript = ""
        except Exception as e:
            # gracefully handle transcription errors but append a log entry
            err_msg = f"[Transcription error: {e}]"
            st.warning(err_msg)
            transcript = err_msg

        # PREDICT
        try:
            if transcript and not transcript.startswith("[Transcription error"):
                label, confidence, features = predict_speech_style(transcript, st.session_state.model, st.session_state.scaler, st.session_state.nlp)
            else:
                # empty or error transcript is treated as "No Speech Detected"
                label, confidence, features = "No Speech Detected", 0.0, {}
        except Exception as e:
            st.error(f"Prediction error: {e}")
            label, confidence, features = "Prediction Error", 0.0, {}

        # append to session history
        append_result(transcript, label, confidence, features)

        # update alert checks
        check_alerts()

        # Update UI displays (table, transcript area, chart, alert)
        # We'll allow the normal flow at the top re-render to render history.
        # But also show latest chunk details now.
        details_placeholder.markdown("**Latest chunk**")
        details_placeholder.text_area("Transcript (latest)", value=transcript, height=200)
        last_row = st.session_state.results_history[-1] if st.session_state.results_history else None

        # display alert box with color
        if st.session_state.alert_state["level"] == "high":
            alert_box.error(st.session_state.alert_state["message"])
        else:
            alert_box.success(st.session_state.alert_state["message"])

        # chart: plot Read Speech confidence over time (we extract the 'Read Speech' confidence)
        hist_df = pd.DataFrame(st.session_state.results_history)
        # ensure numeric
        if not hist_df.empty:
            # make a numeric list for confidence (use Confidence directly)
            chart_series = hist_df[["Chunk #", "Confidence"]].set_index("Chunk #")
            chart_box.line_chart(chart_series)

        # after finishing one chunk, allow another chunk by rerunning the app.
        # This approach keeps the UI responsive to Stop button presses.
        st.session_state.processing_chunk = False
        # Re-run script to allow user to click Stop button or continue (Start remains disabled).
        st.experimental_rerun()

    except Exception as e:
        st.error(f"Unexpected monitoring error: {e}")
        st.session_state.monitoring_active = False
        st.session_state.processing_chunk = False
        st.experimental_rerun()
else:
    # Monitoring not active: show summary / download if available
    if st.session_state.results_history:
        df = pd.DataFrame(st.session_state.results_history)
        log_placeholder.dataframe(df, use_container_width=True)
        # show last transcript in details
        details_placeholder.markdown("**Last Transcript**")
        details_placeholder.text_area("Transcript (latest)", value=df.iloc[-1]["Transcript"], height=200)

        # show chart
        hist_df = df[["Chunk #", "Confidence"]].set_index("Chunk #")
        chart_box.line_chart(hist_df)

        # show alert
        if st.session_state.alert_state["level"] == "high":
            alert_box.error(st.session_state.alert_state["message"])
        else:
            alert_box.info(st.session_state.alert_state["message"])

        # Download button for CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results (CSV)", data=csv, file_name="monitor_results.csv", mime="text/csv")
    else:
        chart_box.info("No session data yet. Start monitoring to collect results.")
        alert_box.info(st.session_state.alert_state.get("message", "Idle."))

# ------------------------------
# --------------- FINAL NOTES
# ------------------------------
st.markdown("---")
st.write("Technical notes:")
st.write("- The app records blocking chunks of `interval` seconds — recording time equals wall-clock monitoring time.")
st.write("- The 'single-chunk-per-run' pattern lets Streamlit stay responsive to the Stop button between chunks.")
st.write("- If you wish to run the recorder on a separate thread (more advanced), ensure careful thread-safe updates to `st.session_state`.")
st.write("- If microphone is not available or permission denied the app will display an error and stop monitoring.")

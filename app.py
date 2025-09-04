import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

REPO_ID = "hilyashfae/transformers"
MAX_LEN = 100

id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT",
}
label2id = {v: k for k, v in id2label.items()}

def preprocess(text: str) -> str:
    text = str(text)
    new_text = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1:
            t = ""
        if t.startswith("http"):
            t = ""
        t = t.replace("#", "")
        t = t.replace("rt", "")
        if t:
            new_text.append(t.lower())
    return " ".join(new_text).strip()


@st.cache_resource
def load_model_and_tokenizer():
    local_dir = snapshot_download(repo_id=REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
    
    # Try TensorFlow first, fall back to PyTorch
    try:
        from transformers import TFAutoModelForSequenceClassification
        model = TFAutoModelForSequenceClassification.from_pretrained(local_dir)
        st.success("Loaded TensorFlow model")
    except:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        st.success("Loaded PyTorch model (TensorFlow not available)")
    
    if getattr(model.config, "id2label", None):
        cfg_map = {int(k): v for k, v in model.config.id2label.items()}
        if set(cfg_map.keys()) == set(id2label.keys()):
            for k in id2label:
                id2label[k] = cfg_map[k]
    return tokenizer, model

def encode_texts(tokenizer, texts, max_len):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf",
    )

def predict_single(tokenizer, model, text, max_len):
    x = preprocess(text)
    enc = encode_texts(tokenizer, [x], max_len)
    outputs = model(enc)
    logits = outputs.logits.numpy()[0]
    probs = tf.nn.softmax(logits).numpy()
    pred_id = int(np.argmax(probs))
    return pred_id, probs

def predict_batch(tokenizer, model, texts, max_len):
    clean = [preprocess(t) for t in texts]
    enc = encode_texts(tokenizer, clean, max_len)
    outputs = model(enc)
    logits = outputs.logits.numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()
    pred_ids = probs.argmax(axis=1).astype(int)
    return pred_ids, probs


st.title("Emotion Mining: Sentence-Based Emotion Prediction")

tokenizer, model = load_model_and_tokenizer()

tab1, tab2 = st.tabs(["Single Text", "Batch CSV"])

with tab1:
    txt = st.text_area("Enter your text here:", height=150)

    if st.button("Predict"):
        if not txt.strip():
            st.warning("The text is still empty.")
        else:
            with st.spinner("Processing..."):
                prog = st.progress(0)
                for p in [20, 45, 70, 90]:
                    time.sleep(0.08)
                    prog.progress(p)

                pred_id, probs = predict_single(tokenizer, model, txt, MAX_LEN)
                prog.progress(100)
                time.sleep(0.05)

            with st.container(border=True):

                top_label = id2label[pred_id]
                st.subheader(f"PREDICT RESULT: **{top_label}**")

                labels = [id2label[i] for i in range(len(probs))]
                probs = np.array(probs, dtype=float)
                perc = (probs * 100).round(2)

                c1, c2 = st.columns(2)
                for i, (lbl, pc) in enumerate(zip(labels, perc)):
                    target_col = c1 if i % 2 == 0 else c2
                    with target_col:
                        st.write(f"**{lbl}** — {pc:.2f}%")
                        st.progress(int(pc))

                st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

                dfp = pd.DataFrame({"prob": probs}, index=labels)
                st.bar_chart(dfp, use_container_width=True)

with tab2:
    file = st.file_uploader("Upload test.csv (id,text)", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        if not {"id", "text"}.issubset(df.columns):
            st.error("CSV must have 'id' and 'text' columns")
        else:
            with st.spinner("Processing..."):
                pred_ids, _ = predict_batch(tokenizer, model, df["text"].tolist(), MAX_LEN)
            subm = pd.DataFrame({"id": df["id"], "label": pred_ids})
            st.dataframe(subm.head())

            st.download_button("Download submission.csv", subm.to_csv(index=False), "submission.csv", "text/csv")

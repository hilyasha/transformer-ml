import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    try:
        local_dir = snapshot_download(repo_id=REPO_ID)
        tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(local_dir, from_tf=True)
        except:
            model = AutoModelForSequenceClassification.from_pretrained(local_dir)

        if hasattr(model.config, "id2label") and model.config.id2label:
            global id2label, label2id
            cfg_map = {int(k): v for k, v in model.config.id2label.items()}
            id2label = cfg_map
            label2id = {v: k for k, v in cfg_map.items()}
            
        return tokenizer, model
        
    except Exception as e:
        st.error(f" Failed to load model: {str(e)}")
        st.info("Please check that the model repository exists and is accessible")
        raise

def encode_texts(tokenizer, texts, max_len):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",  # PyTorch tensors
    )

def predict_single(tokenizer, model, text, max_len):
    x = preprocess(text)
    enc = encode_texts(tokenizer, [x], max_len)
    
    with torch.no_grad():
        outputs = model(**enc)
    
    logits = outputs.logits.numpy()[0]
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
    pred_id = int(np.argmax(probs))
    return pred_id, probs

def predict_batch(tokenizer, model, texts, max_len):
    clean = [preprocess(t) for t in texts]
    enc = encode_texts(tokenizer, clean, max_len)
    
    with torch.no_grad():
        outputs = model(**enc)
    
    logits = outputs.logits.numpy()
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    pred_ids = probs.argmax(axis=1).astype(int)
    return pred_ids, probs


st.title("Emotion Mining: Sentence-Based Emotion Prediction")

try:
    tokenizer, model = load_model_and_tokenizer()
    
    tab1, tab2 = st.tabs(["Single Text", "Batch CSV"])

    with tab1:
        txt = st.text_area("Enter your text here:", height=150, placeholder="Type your text here...")

        if st.button("Predict Emotion", type="primary"):
            if not txt.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing emotions..."):
                    prog = st.progress(0)
                    for p in [20, 45, 70, 90]:
                        time.sleep(0.08)
                        prog.progress(p)

                    pred_id, probs = predict_single(tokenizer, model, txt, MAX_LEN)
                    prog.progress(100)
                    time.sleep(0.05)

                with st.container(border=True):
                    top_label = id2label[pred_id]
                    st.subheader(f" PREDICTED EMOTION: **{top_label}**")

                    labels = [id2label[i] for i in range(len(probs))]
                    perc = (probs * 100).round(2)

                    st.write("**Confidence Scores:**")
                    cols = st.columns(2)
                    for i, (lbl, pc) in enumerate(zip(labels, perc)):
                        col_idx = i % 2
                        with cols[col_idx]:
                            st.metric(label=lbl, value=f"{pc:.1f}%")
                            st.progress(float(pc/100))

                    chart_data = pd.DataFrame({
                        "Emotion": labels,
                        "Confidence": perc
                    })
                    st.bar_chart(chart_data.set_index("Emotion"))

    with tab2:
        st.write("Upload a CSV file with 'id' and 'text' columns")
        file = st.file_uploader("Choose CSV file", type="csv")
        
        if file is not None:
            df = pd.read_csv(file)
            if not {"id", "text"}.issubset(df.columns):
                st.error("CSV must contain 'id' and 'text' columns")
            else:
                with st.spinner("Processing batch predictions..."):
                    pred_ids, all_probs = predict_batch(tokenizer, model, df["text"].tolist(), MAX_LEN)
                
                results = pd.DataFrame({
                    "id": df["id"],
                    "predicted_label": pred_ids,
                    "predicted_emotion": [id2label[pid] for pid in pred_ids]
                })
                
                st.success(f"Processed {len(results)} samples")
                st.dataframe(results.head(10))
                
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download predictions as CSV",
                    csv,
                    "emotion_predictions.csv",
                    "text/csv",
                    key='download-csv'
                )

except Exception as e:
    st.error("Failed to initialize the application. Please check the model configuration.")
    st.code(f"Error: {str(e)}")

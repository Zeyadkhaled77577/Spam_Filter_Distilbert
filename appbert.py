import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="📩 Spam Filter App (BERT)", layout="wide")

# =============================
# Hugging Face repo ID
# =============================
REPO_ID = "ZeyadKhaled/Distilbert"  # Your Hugging Face model repo

# =============================
# Load model & tokenizer (cached)
# =============================
@st.cache_resource
def load_model():
    try:
        with st.spinner("⏳ Downloading and loading model... please wait (first run may take 1-2 min)"):
            tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
            model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model, tokenizer, device = load_model()

# =============================
# Prediction helper
# =============================
def predict_messages(messages, threshold=0.5):
    if not messages:
        return [], []

    inputs = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

    preds = ["Spam" if p >= threshold else "Ham" for p in probs]
    return preds, probs

# =============================
# Streamlit UI
# =============================
st.title("📩 Unified Spam Filter App — BERT Edition")
st.write(
    "Paste one message or many messages (one per line) OR upload a CSV with a 'text' column. "
    "Then press **Predict**."
)

# Sidebar: threshold
threshold = st.sidebar.slider("Set Spam Threshold", 0.0, 1.0, 0.5, 0.01)

# Inputs
user_input = st.text_area(
    "Enter message(s) (one per line for multiple):",
    placeholder="Example:\nHello, are we meeting today?\nCongratulations! You've won a prize..."
)
uploaded_file = st.file_uploader("Or upload CSV with a 'text' column", type=["csv"])

# =============================
# Prediction Logic
# =============================
if st.button("Predict", key="predict_btn"):
    messages = []

    # Case 1: CSV uploaded
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Can't read uploaded CSV: {e}")
        else:
            if "text" in df.columns:
                messages = df["text"].astype(str).tolist()
            else:
                first_col = df.columns[0]
                st.warning(
                    f"CSV does not have a 'text' column — using first column '{first_col}' as text."
                )
                messages = df[first_col].astype(str).tolist()

    # Case 2: Textarea input
    elif user_input.strip():
        messages = [line.strip() for line in user_input.splitlines() if line.strip()]

    # No valid input
    if not messages:
        st.warning("⚠️ Please paste at least one message in the textarea or upload a CSV file.")
    else:
        preds, probs = predict_messages(messages, threshold=threshold)
        result_df = pd.DataFrame({
            "Message": messages,
            "Prediction": preds,
            "Spam_Score": probs
        })

        # Show results
        if len(messages) == 1:
            prob = float(probs[0])
            label = "🚨 Spam" if prob >= threshold else "✅ Ham"
            st.metric(label="Prediction", value=label)
            st.progress(min(prob, 1.0))  # progress bar (caps at 1.0)
            st.write(f"**Spam Score:** {prob:.2f}")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Prediction (CSV)",
                result_df.to_csv(index=False).encode("utf-8"),
                "single_prediction.csv",
                "text/csv"
            )
        else:
            st.success(f"✅ Classified {len(messages)} messages.")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Predictions (CSV)",
                result_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )

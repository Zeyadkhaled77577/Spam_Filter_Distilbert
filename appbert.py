import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io

# =============================
# Hugging Face repo ID
# =============================
REPO_ID = "ZeyadKhaled/Distilbert"  # Your Hugging Face model repo

# =============================
# Load model & tokenizer
# =============================
@torch.no_grad()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# =============================
# Prediction helper
# =============================
def predict_messages(messages, threshold=0.5):
    if not messages:
        return pd.DataFrame(columns=["Message", "Prediction", "Spam_Score"]), None

    if isinstance(messages, str):
        messages = [messages]

    # If uploaded CSV
    if isinstance(messages, pd.DataFrame):
        if "text" in messages.columns:
            messages = messages["text"].astype(str).tolist()
        else:
            messages = messages.iloc[:, 0].astype(str).tolist()

    # Tokenize and predict
    inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

    preds = ["Spam" if p >= threshold else "Ham" for p in probs]

    result_df = pd.DataFrame({
        "Message": messages,
        "Prediction": preds,
        "Spam_Score": probs
    })

    # CSV bytes for download
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    return result_df, csv_bytes

# =============================
# Gradio interface
# =============================
def gradio_interface(single_text, file, threshold):
    # Handle CSV upload
    df = None
    if file is not None:
        df = pd.read_csv(file.name)
    return predict_messages(single_text if df is None else df, threshold)

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 📩 Unified Spam Filter App — BERT Edition")
    gr.Markdown(
        "Paste one message or many messages (one per line) OR upload a CSV with a 'text' column. "
        "Then press **Predict**."
    )

    with gr.Row():
        with gr.Column():
            single_text = gr.Textbox(
                label="Enter message(s) (one per line for multiple)",
                placeholder="Example:\nHello, are we meeting today?\nCongratulations! You've won a prize..."
            )
            file = gr.File(label="Or upload CSV with a 'text' column", file_types=['.csv'])
            threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Set Spam Threshold")
            predict_btn = gr.Button("Predict")
        with gr.Column():
            result_df = gr.Dataframe(headers=["Message", "Prediction", "Spam_Score"])
            download_btn = gr.File(label="Download Predictions (CSV)")

    predict_btn.click(
        fn=gradio_interface,
        inputs=[single_text, file, threshold],
        outputs=[result_df, download_btn]
    )

if __name__ == "__main__":
    demo.launch()

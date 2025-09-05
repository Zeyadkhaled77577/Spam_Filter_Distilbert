
# 📩 Unified Spam Filter App — BERT Edition

[![Hugging Face](https://img.shields.io/badge/HuggingFace-DistilBERT-purple)](https://huggingface.co/ZeyadKhaled/Distilbert)

A Streamlit web application that classifies messages as Spam or Ham using a fine-tuned DistilBERT model hosted on Hugging Face.

------------------------------------------------------------

## Folder Structure

Deep_Learning/
```
  ├─ appbert.py                 # Streamlit app
  ├─ requirements.txt           # Python dependencies
  ├─ spam-filtering-distilbert.ipynb  # Notebook for training/analysis
  └─ model/                     # Hugging Face model (not required locally after upload)
```
------------------------------------------------------------

## Features

- Detect spam in single messages or multiple messages at once.
- Upload a CSV file with a `text` column for batch predictions.
- Adjustable spam threshold using the sidebar slider.
- Download prediction results as a CSV.
- Model hosted on Hugging Face Hub, so no large files are required locally.

------------------------------------------------------------

## Demo

You can run the app locally or deploy it online via Streamlit Cloud.

------------------------------------------------------------

## Installation

# 1. Clone the repository
```
  git clone https://github.com/YOUR_USERNAME/spam-filtering-app.git
  cd spam-filtering-app
```
# 2. Install dependencies
```
  pip install -r requirements.txt
```
------------------------------------------------------------

## Usage

# Run the Streamlit app
```
  streamlit run appbert.py
```
- Paste messages in the text area or upload a CSV with a `text` column.
- Adjust the spam threshold from the sidebar.
- Click Predict to see results.
- Download predictions as CSV if needed.

------------------------------------------------------------

## Model

- Model: DistilBERT fine-tuned for spam classification
- Hugging Face repo: https://huggingface.co/ZeyadKhaled/Distilbert
- The app automatically loads the model from Hugging Face, no local model download needed.

------------------------------------------------------------

## Requirements
```
  streamlit==1.36.0
  torch==2.3.0
  transformers==4.42.0
  pandas==2.2.2
  safetensors==0.4.3
  numpy==1.26.4
  huggingface-hub>=0.17.0
```
------------------------------------------------------------

## Future Work

- **Ensemble Learning:** Combine DistilBERT with other models (e.g., LSTM, RoBERTa) for improved spam detection accuracy.
- **Bilingual Support:** Extend the model to classify messages in multiple languages.
- **UI Enhancements:** Improve the Streamlit interface with visualizations and batch analytics.
- **Deployment Optimization:** Fully deploy the app on Streamlit Cloud or other cloud platforms with auto model loading from Hugging Face.

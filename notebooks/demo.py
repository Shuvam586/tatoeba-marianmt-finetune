import torch
import pathlib
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

MODEL_PATH = pathlib.Path().resolve().parent

@st.cache_resource
def load_model(path: str):
    tokenizer = MarianTokenizer.from_pretrained(path)
    model = MarianMTModel.from_pretrained(path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model(MODEL_PATH)

def translate(text: str, num_beams: int = 5, max_length: int = 128) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, num_beams=num_beams, max_length=max_length)

    return tokenizer.decode(out[0], skip_special_tokens=True)

st.set_page_config(page_title="finetuned-tatoeba-en2fr", page_icon="🇫🇷", layout="centered")
st.title("🇬🇧 ➡️ 🇫🇷  finetuned-tatoeba-en2fr")
st.caption(f"device: **{device}**")

st.subheader("English")

text = st.text_area(label="", height=100, placeholder="Type text here...", label_visibility="collapsed")

st.subheader("French")
output_box = st.empty()

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("translating..."):
            try:
                result = translate(text.strip())
                output_box.markdown(result)
            except Exception as e:
                st.exception(e)

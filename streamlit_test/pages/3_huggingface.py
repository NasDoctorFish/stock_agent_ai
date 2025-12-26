import streamlit as st
from llama_cpp import Llama

st.title("Phi-3 Medium 14B Chatbot")

MODEL = "models/phi-3-medium-128k-instruct.Q4_K_M.gguf"

@st.cache_resource
def load_llm():
    return Llama(
        model_path=MODEL,
        n_ctx=4096,
        n_gpu_layers=40,
        temperature=0.2,
    )

llm = load_llm()

user = st.text_input("Ask me anything:")

if user:
    out = llm(user, max_tokens=500)
    st.write(out["choices"][0]["text"])

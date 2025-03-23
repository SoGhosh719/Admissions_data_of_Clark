import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

st.set_page_config(page_title="Clark GenAI Admissions Assistant", layout="wide")

# Load CSV data
@st.cache_data
def load_data():
    df = pd.read_csv("Admissions Data of Clark.csv")
    df.fillna("", inplace=True)
    return df

df = load_data()

# Convert rows to text
docs = df.apply(lambda row: f"{row['Category']} - {row['Subcategory']} | {row['Label']}: {row['Value']} | {row['Details']}", axis=1).tolist()

# Embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
doc_embeddings = embedder.encode(docs, convert_to_tensor=False)

# FAISS index
dimension = doc_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Generation model
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=150)

generator = load_generator()

# UI
st.title("🎓 Clark GenAI Admissions Assistant")
st.markdown("Ask anything about applying to Clark University — I'll try my best to help!")

user_input = st.chat_input("E.g. When is the regular decision deadline?")

if user_input:
    st.chat_message("user").write(user_input)

    # Embed question
    question_vec = embedder.encode([user_input])
    _, top_k = index.search(np.array(question_vec), k=3)
    context = "\n".join([docs[i] for i in top_k[0]])

    # Prompt the model
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_input}\nAnswer:"
    try:
        with st.spinner("Let me think..."):
            result = generator(prompt)[0]['generated_text']
            response = result.split("Answer:")[-1].strip()
            st.chat_message("assistant").write(response)
    except:
        st.warning("⚠️ The model couldn't generate a response. Here's the top relevant info I found:")
        for i in top_k[0]:
            row = df.iloc[i]
            st.markdown(f"**{row['Label']}** — {row['Value']}  
            _{row['Details']}_")


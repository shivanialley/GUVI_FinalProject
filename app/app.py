import streamlit as st
from src.rag_pipeline import rag_answer

st.set_page_config("NQ RAG Chatbot")
st.title("📘 Natural Questions RAG Chatbot")

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    answer, sources = rag_answer(query)
    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 📚 Retrieved Context")
    for s in sources:
        st.info(s[:400])

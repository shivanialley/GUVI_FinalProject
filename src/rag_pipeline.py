from transformers import pipeline
from retriever import retrieve

llm = pipeline("text-generation", model="gpt2")

def rag_answer(query):
    contexts = retrieve(query)
    prompt = f"""
    Context:
    {contexts}

    Question: {query}
    Answer:
    """
    output = llm(prompt, max_length=300)
    return output[0]["generated_text"], contexts

from src.data_loader import load_nq_dataset
from src.preprocessing import chunk_text
from src.embed_store import build_vector_store

docs = load_nq_dataset(r"C:\Users\Dell\Documents\guvi\Project\final_project2\data\simplified-nq-train.jsonl",limit=500)
chunks = []

for doc in docs[:500]:   
    chunks.extend(chunk_text(doc))

build_vector_store(chunks)

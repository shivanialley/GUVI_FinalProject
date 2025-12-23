import faiss, pickle
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/vector_store/faiss.index")
chunks = pickle.load(open("embeddings/vector_store/chunks.pkl", "rb"))

def retrieve(query, top_k=3):
    q_emb = model.encode([query])
    distances, indices = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in indices[0]]

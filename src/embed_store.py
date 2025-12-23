from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, "embeddings/vector_store/faiss.index")
    pickle.dump(chunks, open("embeddings/vector_store/chunks.pkl", "wb"))

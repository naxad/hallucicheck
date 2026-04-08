# verifier/services/retrieval.py
import numpy as np
import faiss

def build_faiss_index(embeddings):
    emb = np.asarray(embeddings).astype("float32")

    # cosine similarity via inner product + L2 normalization
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def retrieve_top_k(index, query_embedding, k=5):
    q = np.asarray(query_embedding).astype("float32")

    # query_embedding is shape (1, dim)
    faiss.normalize_L2(q)

    D, I = index.search(q, k)   # D and I are shape (1, k)

    # ✅ return 1D python lists
    return D[0].tolist(), I[0].tolist()
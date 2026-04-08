from .chunking import chunk_text
from .embeddings import embed_texts
from .retrieval import build_faiss_index, retrieve_top_k

def retrieve_context(extracted_text: str, question: str, top_k: int = 5) -> list[str]:
    """
    Retrieve top-k chunks for answering the question (RAG context).
    """
    chunks = chunk_text(extracted_text)
    if not chunks:
      return []

    chunk_emb = embed_texts(chunks)
    query_emb = embed_texts([question])

    index = build_faiss_index(chunk_emb)
    scores, idxs = retrieve_top_k(index, query_emb, k=min(top_k, len(chunks)))

    context_chunks = []
    for idx in idxs:
        if idx == -1:
            continue
        context_chunks.append(chunks[idx])

    return context_chunks
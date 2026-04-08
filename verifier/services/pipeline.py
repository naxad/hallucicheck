import time
from .pdf_extract import extract_pdf_text
from .chunking import chunk_text
from .embeddings import embed_texts
from .retrieval import build_faiss_index, retrieve_top_k
from .nli import nli_scores
from .aggregation import aggregate_verdict


def prepare_document_chunks(pdf_paths: list[str], documents=None):
    """
    Extract text from multiple PDFs and return:
    - combined extracted text
    - structured chunk records with source tracking
    """
    all_texts = []
    chunk_records = []

    for i, pdf_path in enumerate(pdf_paths):
        extracted = extract_pdf_text(pdf_path)

        if extracted and len(extracted.strip()) >= 50:
            all_texts.append(extracted)
            chunks = chunk_text(extracted)

            source_name = None
            if documents and i < len(documents):
                source_name = documents[i].title
            else:
                source_name = f"Document {i+1}"

            for chunk in chunks:
                chunk_records.append({
                    "chunk_text": chunk,
                    "source_name": source_name,
                })

    combined_text = "\n\n".join(all_texts)
    return combined_text, chunk_records


def retrieve_evidence_chunks(chunk_records: list[dict], question: str, top_k: int = 5):
    """
    Retrieve top-k evidence chunks for a question using embeddings + FAISS.
    Each chunk keeps its source document name.
    """
    if not chunk_records:
        return []

    chunk_texts = [c["chunk_text"] for c in chunk_records]
    query_text = f"Question: {question}"

    chunk_emb = embed_texts(chunk_texts)
    query_emb = embed_texts([query_text])

    index = build_faiss_index(chunk_emb)
    scores, idxs = retrieve_top_k(index, query_emb, k=min(top_k, len(chunk_texts)))

    evidence_rows = []
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        if idx == -1:
            continue

        evidence_rows.append(
            {
                "rank": rank,
                "chunk_text": chunk_records[idx]["chunk_text"],
                "source_name": chunk_records[idx]["source_name"],
                "similarity_score": float(score),
            }
        )

    return evidence_rows



def score_answer_against_evidence(evidence_rows: list[dict], answer: str):
    """
    Run NLI scoring for each retrieved chunk against the answer.
    """
    scored_rows = []

    for row in evidence_rows:
        nli = nli_scores(row["chunk_text"], answer)

        scored_rows.append(
            {
                "rank": row["rank"],
                "chunk_text": row["chunk_text"],
                "source_name": row["source_name"],
                "similarity_score": row["similarity_score"],
                "entailment": nli["entailment"],
                "neutral": nli["neutral"],
                "contradiction": nli["contradiction"],
            }
        )

    return scored_rows


def run_pipeline(pdf_paths: list[str], question: str, answer: str, top_k: int = 5, documents=None):
    """
    Full multi-document verification pipeline.
    Returns:
      - extracted_text_combined
      - all_chunks
      - scored evidence rows
      - verdict
      - confidence
      - runtime_ms
    """
    t0 = time.time()

    extracted_text, chunk_records = prepare_document_chunks(pdf_paths, documents)

    if not extracted_text.strip() or not chunk_records:
        runtime_ms = int((time.time() - t0) * 1000)
        return extracted_text, [], [], "Unsupported", 0.0, runtime_ms

    evidence_rows = retrieve_evidence_chunks(chunk_records, question, top_k=top_k)
    scored_rows = score_answer_against_evidence(evidence_rows, answer)

    verdict, confidence = aggregate_verdict(scored_rows)
    runtime_ms = int((time.time() - t0) * 1000)

    return extracted_text, chunk_records, scored_rows, verdict, confidence, runtime_ms
def aggregate_verdict(chunk_results: list[dict]) -> tuple[str, float]:
    """
    chunk_results: list of dicts containing entailment/neutral/contradiction per chunk.
    Returns (verdict, confidence).
    """
    if not chunk_results:
        return "Unsupported", 0.0

    best_entail = max(r["entailment"] for r in chunk_results)
    best_contra = max(r["contradiction"] for r in chunk_results)

    # Simple, defensible thresholds for MVP (you can tune later using evaluation set)
    if best_entail >= 0.75 and best_contra <= 0.20:
        verdict = "Supported"
        confidence = best_entail
    elif best_entail >= 0.45 and best_contra <= 0.35:
        verdict = "Partially Supported"
        confidence = best_entail
    else:
        verdict = "Unsupported"
        confidence = 1.0 - best_entail

    return verdict, round(float(confidence), 2)
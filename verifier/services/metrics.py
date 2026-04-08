def compute_run_metrics(chunks):
    """
    Compute summary metrics from a queryset or list of EvidenceChunk objects.
    Returns a dictionary of values for the dashboard and storage.
    """
    chunks = list(chunks)

    if not chunks:
        return {
            "best_entailment": 0.0,
            "avg_entailment": 0.0,
            "avg_contradiction": 0.0,
            "best_similarity": 0.0,
            "risk_percentage": 100,
            "evidence_count": 0,
            "support_count": 0,
            "contradiction_count": 0,
        }

    entailments = [(c.entailment or 0.0) for c in chunks]
    contradictions = [(c.contradiction or 0.0) for c in chunks]
    similarities = [(c.similarity_score or 0.0) for c in chunks]

    best_entailment = max(entailments)
    avg_entailment = sum(entailments) / len(entailments)
    avg_contradiction = sum(contradictions) / len(contradictions)
    best_similarity = max(similarities)

    support_count = sum(1 for c in chunks if (c.entailment or 0.0) >= 0.70)
    contradiction_count = sum(1 for c in chunks if (c.contradiction or 0.0) >= 0.40)
    evidence_count = len(chunks)

    risk_score = (
        (1.0 - best_entailment) * 0.5 +
        avg_contradiction * 0.3 +
        (1.0 - best_similarity) * 0.2
    )

    risk_percentage = max(0, min(100, round(risk_score * 100)))

    return {
        "best_entailment": round(best_entailment, 3),
        "avg_entailment": round(avg_entailment, 3),
        "avg_contradiction": round(avg_contradiction, 3),
        "best_similarity": round(best_similarity, 3),
        "risk_percentage": risk_percentage,
        "evidence_count": evidence_count,
        "support_count": support_count,
        "contradiction_count": contradiction_count,
    }


def compute_trust_score(metrics: dict) -> dict:
    """
    Combine key evaluation signals into a final trust score (0-100)
    and a final verdict label.
    """

    entailment = metrics.get("best_entailment", 0.0)
    similarity = metrics.get("best_similarity", 0.0)

    ragas_faithfulness = metrics.get("ragas_faithfulness")
    deepeval_faithfulness = metrics.get("deepeval_faithfulness")

    # fallback values if external frameworks are missing
    ragas_faithfulness = ragas_faithfulness if ragas_faithfulness is not None else 0.5
    deepeval_faithfulness = deepeval_faithfulness if deepeval_faithfulness is not None else 0.5

    score = (
        entailment * 0.40 +
        similarity * 0.20 +
        ragas_faithfulness * 0.20 +
        deepeval_faithfulness * 0.20
    )

    trust_score = round(score * 100)

    if trust_score >= 75 and entailment >= 0.60:
        final_verdict = "SUPPORTED"
    elif trust_score >= 45:
        final_verdict = "PARTIAL"
    else:
        final_verdict = "UNSUPPORTED"

    return {
        "trust_score": trust_score,
        "final_verdict": final_verdict,
    }


def generate_explanation(metrics: dict) -> str:
    best_entailment = metrics.get("best_entailment") or 0.0
    avg_contradiction = metrics.get("avg_contradiction") or 0.0
    best_similarity = metrics.get("best_similarity") or 0.0
    ragas_faithfulness = metrics.get("ragas_faithfulness")
    ragas_relevancy = metrics.get("ragas_answer_relevancy")
    deepeval_faithfulness = metrics.get("deepeval_faithfulness")
    deepeval_relevancy = metrics.get("deepeval_answer_relevancy")
    trust_score = metrics.get("trust_score")
    final_verdict = metrics.get("final_verdict")

    parts = []

    if final_verdict:
        parts.append(f"The final system verdict is {final_verdict.lower()}.")

    if trust_score is not None:
        parts.append(f"The overall trust score is {trust_score}%.")

    if best_entailment >= 0.7:
        parts.append(
            "At least one retrieved evidence chunk strongly supports the answer."
        )
    elif best_entailment >= 0.4:
        parts.append(
            "The answer has moderate evidence support, but the support is not decisive."
        )
    else:
        parts.append(
            "The retrieved evidence provides weak direct support for the answer."
        )

    if avg_contradiction >= 0.4:
        parts.append(
            "Contradictory evidence was detected, which increases the likelihood of hallucination."
        )
    elif avg_contradiction > 0.0:
        parts.append(
            "Some contradiction signal is present, although it is not dominant."
        )

    if best_similarity >= 0.7:
        parts.append(
            "The answer is semantically close to the retrieved document evidence."
        )
    elif best_similarity >= 0.4:
        parts.append(
            "The answer is moderately related to the retrieved evidence."
        )
    else:
        parts.append(
            "The answer has weak semantic alignment with the retrieved evidence."
        )

    if ragas_faithfulness is not None:
        if ragas_faithfulness >= 0.7:
            parts.append("RAGAS indicates strong faithfulness to the retrieved context.")
        elif ragas_faithfulness >= 0.4:
            parts.append("RAGAS indicates partial faithfulness to the retrieved context.")
        else:
            parts.append("RAGAS indicates weak faithfulness to the retrieved context.")

    if ragas_relevancy is not None:
        if ragas_relevancy >= 0.7:
            parts.append("RAGAS indicates that the answer is highly relevant to the question.")
        elif ragas_relevancy >= 0.4:
            parts.append("RAGAS indicates moderate question relevancy.")
        else:
            parts.append("RAGAS indicates weak question relevancy.")

    if deepeval_faithfulness is not None:
        if deepeval_faithfulness >= 0.7:
            parts.append("DeepEval indicates strong faithfulness.")
        elif deepeval_faithfulness >= 0.4:
            parts.append("DeepEval indicates moderate faithfulness.")
        else:
            parts.append("DeepEval indicates weak faithfulness.")

    if deepeval_relevancy is not None:
        if deepeval_relevancy >= 0.7:
            parts.append("DeepEval indicates strong answer relevancy.")
        elif deepeval_relevancy >= 0.4:
            parts.append("DeepEval indicates moderate answer relevancy.")
        else:
            parts.append("DeepEval indicates weak answer relevancy.")

    return " ".join(parts)



def detect_model_disagreement(comparison_runs: list[dict]) -> str | None:
    """
    Detect disagreement between model runs in comparison mode.

    comparison_runs is a list of dicts like:
    {
        "run": run,
        "chunks": chunks,
        "metrics": metrics,
    }

    Returns:
        - a human-readable disagreement message
        - or None if there is no significant disagreement
    """

    if len(comparison_runs) < 2:
        return None

    final_verdicts = []
    trust_scores = []

    for item in comparison_runs:
        metrics = item.get("metrics", {})
        verdict = metrics.get("final_verdict")
        trust_score = metrics.get("trust_score")

        if verdict:
            final_verdicts.append(verdict)

        if trust_score is not None:
            trust_scores.append(trust_score)

    # Case 1: direct verdict disagreement
    unique_verdicts = set(final_verdicts)
    if len(unique_verdicts) > 1:
        return "Model disagreement detected: different models produced different final verdicts for the same question and evidence."

    # Case 2: same verdict, but trust scores differ a lot
    if len(trust_scores) >= 2:
        score_gap = max(trust_scores) - min(trust_scores)
        if score_gap >= 25:
            return "Confidence disagreement detected: models reached similar verdicts, but their trust scores differ significantly."

    return None
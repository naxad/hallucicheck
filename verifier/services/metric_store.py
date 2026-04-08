from verifier.models import RunMetric


def save_run_metrics(run, metrics: dict):
    """
    Create or update stored metrics for a run.
    """
    obj, _ = RunMetric.objects.update_or_create(
        run=run,
        defaults={
            "best_entailment": metrics.get("best_entailment", 0.0),
            "avg_entailment": metrics.get("avg_entailment", 0.0),
            "avg_contradiction": metrics.get("avg_contradiction", 0.0),
            "best_similarity": metrics.get("best_similarity", 0.0),
            "risk_percentage": metrics.get("risk_percentage", 100),
            "evidence_count": metrics.get("evidence_count", 0),
            "support_count": metrics.get("support_count", 0),
            "contradiction_count": metrics.get("contradiction_count", 0),
            "ragas_faithfulness": metrics.get("ragas_faithfulness"),
            "ragas_answer_relevancy": metrics.get("ragas_answer_relevancy"),
            "deepeval_faithfulness": metrics.get("deepeval_faithfulness"),
            "deepeval_answer_relevancy": metrics.get("deepeval_answer_relevancy"),
            "trust_score": metrics.get("trust_score"),
            "final_verdict": metrics.get("final_verdict"),
            "explanation": metrics.get("explanation"),
        }
    )
    return obj
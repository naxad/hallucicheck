from django.db import models
import uuid


class Document(models.Model):
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to="documents/")
    extracted_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or f"Document {self.id}"


class EvaluationRun(models.Model):
    VERDICT_SUPPORTED = "SUPPORTED"
    VERDICT_PARTIAL = "PARTIAL"
    VERDICT_UNSUPPORTED = "UNSUPPORTED"

    VERDICT_CHOICES = [
        (VERDICT_SUPPORTED, "Supported"),
        (VERDICT_PARTIAL, "Partially Supported"),
        (VERDICT_UNSUPPORTED, "Unsupported"),
    ]

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="runs")
    question = models.TextField()
    answer = models.TextField()

    verdict = models.CharField(max_length=20, choices=VERDICT_CHOICES, blank=True)
    confidence = models.FloatField(null=True, blank=True)

    # ✅ New fields for model comparison
    provider = models.CharField(max_length=50, blank=True)
    model_name = models.CharField(max_length=100, blank=True)
    comparison_id = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)

    created_at = models.DateTimeField(auto_now_add=True)
    runtime_ms = models.IntegerField(null=True, blank=True)

    def __str__(self):
        label = self.model_name or "manual"
        return f"Run {self.id} ({label})"


class EvidenceChunk(models.Model):
    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="chunks")
    rank = models.IntegerField()
    chunk_text = models.TextField()
    source_name=models.CharField(max_length=255, blank=True)
    similarity_score = models.FloatField()

    entailment = models.FloatField(null=True, blank=True)
    neutral = models.FloatField(null=True, blank=True)
    contradiction = models.FloatField(null=True, blank=True)

    is_best = models.BooleanField(default=False)

    def __str__(self):
        return f"Chunk r{self.rank} (run {self.run_id})"


class NLIResult(models.Model):
    chunk = models.OneToOneField(EvidenceChunk, on_delete=models.CASCADE, related_name="nli")
    entailment = models.FloatField(default=0.0)
    neutral = models.FloatField(default=0.0)
    contradiction = models.FloatField(default=0.0)


class RunMetric(models.Model):
    run = models.OneToOneField(EvaluationRun, on_delete=models.CASCADE, related_name="metric")

    #pure metrics
    best_entailment = models.FloatField(default=0.0)
    avg_entailment = models.FloatField(default=0.0)
    avg_contradiction = models.FloatField(default=0.0)
    best_similarity = models.FloatField(default=0.0)

    risk_percentage = models.IntegerField(default=100)

    evidence_count = models.IntegerField(default=0)
    support_count = models.IntegerField(default=0)
    contradiction_count = models.IntegerField(default=0)

    #ragas metrics
    ragas_faithfulness = models.FloatField(null=True, blank=True)
    ragas_answer_relevancy = models.FloatField(null=True, blank=True)
    
    #deepeval metrics
    deepeval_faithfulness = models.FloatField(null=True, blank=True)
    deepeval_answer_relevancy = models.FloatField(null=True, blank=True)


    #trust score and final verdict
    trust_score = models.IntegerField(null=True, blank=True)
    final_verdict = models.CharField(max_length=20, blank=True)

    explanation = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Metrics for run {self.run_id}"
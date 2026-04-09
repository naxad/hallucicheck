from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
import uuid

from .models import Document, EvaluationRun, EvidenceChunk

from .services.llm_answer import generate_answer
from .services.metrics import compute_run_metrics, compute_trust_score, generate_explanation, detect_model_disagreement
from .services.highlighting import highlight_chunk_sentences
from .services.metric_store import save_run_metrics
from .services.ragas_eval import evaluate_with_ragas
from django.db.models import Q

from .services.deepeval_eval import evaluate_with_deepeval

from django.contrib.auth.decorators import login_required
import math

@login_required
def home(request):
    run_id = request.GET.get("run")
    comparison_id = request.GET.get("comparison")

    # -------------------------
    # COMPARISON MODE
    # -------------------------
    if comparison_id:
        runs = EvaluationRun.objects.filter(comparison_id=comparison_id).order_by("created_at")
        comparison_runs = []

        for run in runs:
            chunks_qs = EvidenceChunk.objects.filter(run=run).order_by("rank")

            chunks = sorted(
                chunks_qs,
                key=lambda c: (
                    -(c.entailment or 0.0),
                    -(c.similarity_score or 0.0),
                    c.rank,
                )
            )

            for c in chunks:
                c.highlighted_html = highlight_chunk_sentences(c.chunk_text, run.answer)

            metrics = compute_run_metrics(chunks)
            if hasattr(run, "metric"):
                metrics = {
                    "best_entailment": run.metric.best_entailment,
                    "avg_entailment": run.metric.avg_entailment,
                    "avg_contradiction": run.metric.avg_contradiction,
                    "best_similarity": run.metric.best_similarity,
                    "risk_percentage": run.metric.risk_percentage,
                    "evidence_count": run.metric.evidence_count,
                    "support_count": run.metric.support_count,
                    "contradiction_count": run.metric.contradiction_count,
                    "ragas_faithfulness": run.metric.ragas_faithfulness,
                    "ragas_answer_relevancy": run.metric.ragas_answer_relevancy,
                    "deepeval_faithfulness": run.metric.deepeval_faithfulness,
                    "deepeval_answer_relevancy": run.metric.deepeval_answer_relevancy,
                    "trust_score": run.metric.trust_score,
                    "final_verdict": run.metric.final_verdict,
                    "explanation": run.metric.explanation,
                }

            comparison_runs.append({
                "run": run,
                "chunks": chunks,
                "metrics": metrics,
            })

        disagreement = detect_model_disagreement(comparison_runs)

        return render(
            request,
            "verifier/home.html",
            {"comparison_runs": comparison_runs, "disagreement": disagreement},
        )

    # -------------------------
    # SINGLE RUN MODE
    # -------------------------
    if run_id:
        run = EvaluationRun.objects.filter(id=run_id).first()

        if run:
            chunks_qs = EvidenceChunk.objects.filter(run=run).order_by("rank")

            chunks = sorted(
                chunks_qs,
                key=lambda c: (
                    -(c.entailment or 0.0),
                    -(c.similarity_score or 0.0),
                    c.rank,
                )
            )

            for c in chunks:
                c.highlighted_html = highlight_chunk_sentences(c.chunk_text, run.answer)

            metrics = compute_run_metrics(chunks)
            if hasattr(run, "metric"):
                metrics = {
                    "best_entailment": run.metric.best_entailment,
                    "avg_entailment": run.metric.avg_entailment,
                    "avg_contradiction": run.metric.avg_contradiction,
                    "best_similarity": run.metric.best_similarity,
                    "risk_percentage": run.metric.risk_percentage,
                    "evidence_count": run.metric.evidence_count,
                    "support_count": run.metric.support_count,
                    "contradiction_count": run.metric.contradiction_count,
                    "ragas_faithfulness": run.metric.ragas_faithfulness,
                    "ragas_answer_relevancy": run.metric.ragas_answer_relevancy,
                    "deepeval_faithfulness": run.metric.deepeval_faithfulness,
                    "deepeval_answer_relevancy": run.metric.deepeval_answer_relevancy,
                    "trust_score": run.metric.trust_score,
                    "final_verdict": run.metric.final_verdict,
                    "explanation": run.metric.explanation,
                }
        else:
            chunks = []
            metrics = None

        return render(
            request,
            "verifier/home.html",
            {
                "run": run,
                "chunks": chunks,
                "metrics": metrics,
            },
        )

    # -------------------------
    # DEFAULT
    # -------------------------
    return render(request, "verifier/home.html")

@login_required
def dashboard(request):
    query = request.GET.get("q", "").strip()
    provider = request.GET.get("provider", "").strip()
    verdict = request.GET.get("verdict", "").strip()
    sort = request.GET.get("sort", "newest").strip()

    runs = EvaluationRun.objects.select_related("metric", "document").all()

    if query:
        runs = runs.filter(
            Q(question__icontains=query) |
            Q(answer__icontains=query) |
            Q(model_name__icontains=query) |
            Q(provider__icontains=query)
        )

    if provider:
        runs = runs.filter(provider__iexact=provider)

    if verdict:
        runs = runs.filter(verdict__iexact=verdict)

    if sort == "oldest":
        runs = runs.order_by("created_at")
    elif sort == "risk_desc":
        runs = sorted(
            runs,
            key=lambda r: getattr(r.metric, "risk_percentage", -1) if hasattr(r, "metric") else -1,
            reverse=True
        )
    elif sort == "confidence_desc":
        runs = runs.order_by("-confidence", "-created_at")
    else:
        runs = runs.order_by("-created_at")

    dashboard_rows = []
    for run in runs:
        metric = getattr(run, "metric", None)

        dashboard_rows.append({
            "run": run,
            "risk_percentage": metric.risk_percentage if metric else None,
            "ragas_faithfulness": metric.ragas_faithfulness if metric else None,
            "ragas_answer_relevancy": metric.ragas_answer_relevancy if metric else None,
            "deepeval_faithfulness": metric.deepeval_faithfulness if metric else None,
            "deepeval_answer_relevancy": metric.deepeval_answer_relevancy if metric else None,
            "best_entailment": metric.best_entailment if metric else None,
            "best_similarity": metric.best_similarity if metric else None,
            "evidence_count": metric.evidence_count if metric else None,
        })

    context = {
        "dashboard_rows": dashboard_rows,
        "query": query,
        "provider_filter": provider,
        "verdict_filter": verdict,
        "sort_filter": sort,
    }
    return render(request, "verifier/dashboard.html", context)



@login_required
@require_http_methods(["POST"])
def run_check(request):
    
    from .services.pipeline import (run_pipeline, prepare_document_chunks, retrieve_evidence_chunks, )

    doc_files = request.FILES.getlist("document_files")
    question = request.POST.get("question", "").strip()
    answer = request.POST.get("answer", "").strip()
    mode = request.POST.get("mode", "manual").strip().lower()
    selected_models = request.POST.getlist("selected_models")

    if not doc_files or not question:
        return redirect("/")

    if mode == "manual" and not answer:
        return redirect("/")

    # 1) Save all uploaded documents
    documents = []
    for uploaded_file in doc_files:
        doc = Document.objects.create(title=uploaded_file.name, file=uploaded_file)
        documents.append(doc)

    primary_doc = documents[0]
    pdf_paths = [doc.file.path for doc in documents]

    # Shared comparison group for all runs triggered in one submission
    comparison_id = uuid.uuid4()

    # -------------------------
    # MODE 1: MANUAL
    # -------------------------
    if mode == "manual":
        run = EvaluationRun.objects.create(
            document=primary_doc,
            question=question,
            answer=answer,
            provider="manual",
            model_name="manual-input",
            comparison_id=comparison_id,
        )

        try:
            extracted_text, chunk_records, evidence_rows, verdict, confidence, runtime_ms = run_pipeline(
                pdf_paths=pdf_paths,
                question=question,
                answer=answer,
                top_k=5,
                documents=documents,
            )
        except Exception as e:
            run.verdict = EvaluationRun.VERDICT_UNSUPPORTED
            run.confidence = 0.0
            run.save()

            EvidenceChunk.objects.create(
                run=run,
                rank=1,
                chunk_text=f"Pipeline error: {str(e)}",
                source_name="System",
                similarity_score=0.0,
                entailment=0.0,
                neutral=0.0,
                contradiction=0.0,
                is_best=True,
            )
            return redirect(f"/?run={run.id}")

        primary_doc.extracted_text = extracted_text
        primary_doc.save()

        verdict_norm = verdict.strip().lower()
        if verdict_norm == "supported":
            run.verdict = EvaluationRun.VERDICT_SUPPORTED
        elif verdict_norm in ("partially supported", "partial"):
            run.verdict = EvaluationRun.VERDICT_PARTIAL
        else:
            run.verdict = EvaluationRun.VERDICT_UNSUPPORTED

        run.confidence = float(confidence)
        run.runtime_ms = runtime_ms
        run.save()

        EvidenceChunk.objects.filter(run=run).delete()

        if not evidence_rows:
            EvidenceChunk.objects.create(
                run=run,
                rank=1,
                chunk_text="No usable text evidence was retrieved. This usually happens if the PDF is scanned/image-based or the extracted text is too short. Try a digital (selectable-text) PDF.",
                source_name="System",
                similarity_score=0.0,
                entailment=0.0,
                neutral=0.0,
                contradiction=0.0,
                is_best=True,
            )
            return redirect(f"/?run={run.id}")

        best_row = max(evidence_rows, key=lambda r: r["entailment"])
        best_rank = best_row["rank"]

        for row in evidence_rows:
            EvidenceChunk.objects.create(
                run=run,
                rank=row["rank"],
                chunk_text=row["chunk_text"],
                source_name=row.get("source_name", ""),
                similarity_score=row["similarity_score"],
                entailment=row["entailment"],
                neutral=row["neutral"],
                contradiction=row["contradiction"],
                is_best=(row["rank"] == best_rank),
            )

        saved_chunks = EvidenceChunk.objects.filter(run=run)
        metrics = compute_run_metrics(saved_chunks)

        contexts = [c.chunk_text for c in saved_chunks]
        ragas_scores = evaluate_with_ragas(
            question=run.question,
            answer=run.answer,
            contexts=contexts,
        )

        deepeval_scores = evaluate_with_deepeval(
            question=run.question,
            answer=run.answer,
            contexts=contexts,
        )

        metrics.update(ragas_scores)
        metrics.update(deepeval_scores)

        for key, value in list(metrics.items()):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                metrics[key] = None

        trust = compute_trust_score(metrics)
        metrics.update(trust)
        explanation = generate_explanation(metrics)
        metrics["explanation"] = explanation

        save_run_metrics(run, metrics)
        

        return redirect(f"/?run={run.id}")

    # -------------------------
    # MODE 2: CHAT + VERIFY
    # -------------------------
    if not selected_models:
        return redirect("/")

    combined_text, chunk_records = prepare_document_chunks(pdf_paths, documents=documents)
    context_rows = retrieve_evidence_chunks(chunk_records, question, top_k=5)
    context_chunks = [row["chunk_text"] for row in context_rows]

    run_ids = []

    for model_spec in selected_models:
        provider, model_name = model_spec.split(":", 1)

        try:
            generated_answer = generate_answer(
                question=question,
                context_chunks=context_chunks,
                provider=provider,
                model=model_name,
            )
        except Exception as e:
            run = EvaluationRun.objects.create(
                document=primary_doc,
                question=question,
                answer=f"[Generation failed] {str(e)}",
                provider=provider,
                model_name=model_name,
                comparison_id=comparison_id,
                verdict=EvaluationRun.VERDICT_UNSUPPORTED,
                confidence=0.0,
                runtime_ms=0,
            )

            EvidenceChunk.objects.create(
                run=run,
                rank=1,
                chunk_text=f"Generation error for {provider}/{model_name}: {str(e)}",
                source_name="System",
                similarity_score=0.0,
                entailment=0.0,
                neutral=0.0,
                contradiction=0.0,
                is_best=True,
            )

            run_ids.append(run.id)
            continue

        run = EvaluationRun.objects.create(
            document=primary_doc,
            question=question,
            answer=generated_answer,
            provider=provider,
            model_name=model_name,
            comparison_id=comparison_id,
        )

        try:
            extracted_text, chunk_records, evidence_rows, verdict, confidence, runtime_ms = run_pipeline(
                pdf_paths=pdf_paths,
                question=question,
                answer=generated_answer,
                top_k=5,
                documents=documents,
            )
        except Exception as e:
            run.verdict = EvaluationRun.VERDICT_UNSUPPORTED
            run.confidence = 0.0
            run.save()

            EvidenceChunk.objects.create(
                run=run,
                rank=1,
                chunk_text=f"Pipeline error: {str(e)}",
                source_name="System",
                similarity_score=0.0,
                entailment=0.0,
                neutral=0.0,
                contradiction=0.0,
                is_best=True,
            )
            run_ids.append(run.id)
            continue

        primary_doc.extracted_text = extracted_text
        primary_doc.save()

        verdict_norm = verdict.strip().lower()
        if verdict_norm == "supported":
            run.verdict = EvaluationRun.VERDICT_SUPPORTED
        elif verdict_norm in ("partially supported", "partial"):
            run.verdict = EvaluationRun.VERDICT_PARTIAL
        else:
            run.verdict = EvaluationRun.VERDICT_UNSUPPORTED

        run.confidence = float(confidence)
        run.runtime_ms = runtime_ms
        run.save()

        EvidenceChunk.objects.filter(run=run).delete()

        if not evidence_rows:
            EvidenceChunk.objects.create(
                run=run,
                rank=1,
                chunk_text="No usable text evidence was retrieved. This usually happens if the PDF is scanned/image-based or the extracted text is too short. Try a digital (selectable-text) PDF.",
                source_name="System",
                similarity_score=0.0,
                entailment=0.0,
                neutral=0.0,
                contradiction=0.0,
                is_best=True,
            )
            run_ids.append(run.id)
            continue

        best_row = max(evidence_rows, key=lambda r: r["entailment"])
        best_rank = best_row["rank"]

        for row in evidence_rows:
            EvidenceChunk.objects.create(
                run=run,
                rank=row["rank"],
                chunk_text=row["chunk_text"],
                source_name=row.get("source_name", ""),
                similarity_score=row["similarity_score"],
                entailment=row["entailment"],
                neutral=row["neutral"],
                contradiction=row["contradiction"],
                is_best=(row["rank"] == best_rank),
            )

        saved_chunks = EvidenceChunk.objects.filter(run=run)
        metrics = compute_run_metrics(saved_chunks)

        contexts = [c.chunk_text for c in saved_chunks]
        ragas_scores = evaluate_with_ragas(
            question=run.question,
            answer=run.answer,
            contexts=contexts,
        )

        deepeval_scores = evaluate_with_deepeval(
            question=run.question,
            answer=run.answer,
            contexts=contexts,
        )

        metrics.update(ragas_scores)
        metrics.update(deepeval_scores)
        for key, value in list(metrics.items()):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                metrics[key] = None
        trust = compute_trust_score(metrics)
        metrics.update(trust)
        explanation = generate_explanation(metrics)
        metrics["explanation"] = explanation
        save_run_metrics(run, metrics)
        

        run_ids.append(run.id)

    return redirect(f"/?comparison={comparison_id}")
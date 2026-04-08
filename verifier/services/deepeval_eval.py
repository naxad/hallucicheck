def evaluate_with_deepeval(question: str, answer: str, contexts: list[str]) -> dict:
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

        if not contexts:
            return {
                "deepeval_faithfulness": None,
                "deepeval_answer_relevancy": None,
            }

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
        )

        faithfulness_metric = FaithfulnessMetric(
            threshold=0.5,
            async_mode=False,
            verbose_mode=False,
        )

        answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.5,
            async_mode=False,
            verbose_mode=False,
        )

        faithfulness_metric.measure(test_case)
        answer_relevancy_metric.measure(test_case)

        print("DEEPEVAL FAITHFULNESS SCORE:", faithfulness_metric.score)
        print("DEEPEVAL ANSWER RELEVANCY SCORE:", answer_relevancy_metric.score)

        return {
            "deepeval_faithfulness": float(faithfulness_metric.score) if faithfulness_metric.score is not None else None,
            "deepeval_answer_relevancy": float(answer_relevancy_metric.score) if answer_relevancy_metric.score is not None else None,
        }

    except Exception as e:
        print("DEEPEVAL EVALUATION ERROR:", repr(e))
        return {
            "deepeval_faithfulness": None,
            "deepeval_answer_relevancy": None,
        }
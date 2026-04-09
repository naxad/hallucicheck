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

        faithfulness_score = None
        answer_relevancy_score = None

        try:
            faithfulness_metric.measure(test_case)
            if faithfulness_metric.score is not None:
                faithfulness_score = float(faithfulness_metric.score)
        except Exception as e:
            print("DEEPEVAL FAITHFULNESS ERROR:", repr(e))

        try:
            answer_relevancy_metric.measure(test_case)
            if answer_relevancy_metric.score is not None:
                answer_relevancy_score = float(answer_relevancy_metric.score)
        except Exception as e:
            print("DEEPEVAL ANSWER RELEVANCY ERROR:", repr(e))

        print("DEEPEVAL FAITHFULNESS SCORE:", faithfulness_score)
        print("DEEPEVAL ANSWER RELEVANCY SCORE:", answer_relevancy_score)

        return {
            "deepeval_faithfulness": faithfulness_score,
            "deepeval_answer_relevancy": answer_relevancy_score,
        }

    except Exception as e:
        print("DEEPEVAL EVALUATION ERROR:", repr(e))
        return {
            "deepeval_faithfulness": None,
            "deepeval_answer_relevancy": None,
        }
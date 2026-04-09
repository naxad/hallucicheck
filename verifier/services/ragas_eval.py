import math

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _clean_score(value):
    if value is None:
        return None
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def evaluate_with_ragas(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Run RAGAS evaluation for one QA pair.
    Returns None values if evaluation fails gracefully.
    """
    if not contexts:
        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
        }

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    dataset = Dataset.from_dict(data)

    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
            show_progress=False,
        )

        row = result.to_pandas().iloc[0].to_dict()
        print("RAGAS RESULT row:", row)

        return {
            "ragas_faithfulness": _clean_score(row.get("faithfulness")),
            "ragas_answer_relevancy": _clean_score(
                row.get("answer_relevancy") or row.get("response_relevancy")
            ),
        }

    except Exception as e:
        print("RAGAS EVALUATION ERROR:", str(e))
        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
        }
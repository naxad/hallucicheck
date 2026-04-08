import re
import html
from .nli import nli_scores


def split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter.
    Good enough for MVP and avoids adding heavy NLP dependencies.
    """
    text = (text or "").strip()
    if not text:
        return []

    # split after ., !, ? followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def highlight_chunk_sentences(chunk_text: str, answer: str) -> str:
    """
    Returns HTML-safe highlighted chunk text.
    Each sentence is evaluated independently with NLI.
    """
    sentences = split_into_sentences(chunk_text)

    if not sentences:
        return html.escape(chunk_text)

    rendered = []

    for sentence in sentences:
        scores = nli_scores(sentence, answer)

        entailment = scores["entailment"]
        contradiction = scores["contradiction"]

        if entailment >= 0.70:
            cls = "sent-supported"
        elif contradiction >= 0.40:
            cls = "sent-contradiction"
        else:
            cls = "sent-neutral"

        rendered.append(
            f'<span class="{cls}">{html.escape(sentence)}</span>'
        )

    return " ".join(rendered)
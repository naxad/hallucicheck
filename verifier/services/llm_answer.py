import os
from openai import OpenAI
from anthropic import Anthropic
from google import genai


def generate_answer(question: str, context_chunks: list[str], provider: str, model: str | None = None) -> str:
    """
    Provider-agnostic answer generation.
    provider: "openai", "anthropic", or "gemini"
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "Your answer must always be a complete, self-contained sentence. "
        "Do not answer with only 'yes', 'no', or very short fragments. "
        "If the answer is not in the context, say exactly: "
        "\"I do not know based on the provided document.\""
    )

    user_prompt = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n"
    )

    provider = provider.strip().lower()

    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.output_text or "").strip()

    elif provider == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.content[0].text.strip()

    elif provider == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = model or os.getenv("GEMINI_MODEL", "gemini-3-flash")

        response = client.models.generate_content(
            model=model_name,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )
        return (response.text or "").strip()

    else:
        raise ValueError(f"Unsupported provider: {provider}")
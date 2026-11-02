import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"[LLM] Error initializing Groq client: {e}")


def get_active_model_name() -> str:
    """
    Selects verified active chat models from the user's specific Groq account.
    """
    if not client:
        return "openai/gpt-oss-120b"

    # Exact verified chat models authorized for this account
    preferred_candidates = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b"
    ]

    try:
        available_models = [m.id for m in client.models.list().data]
        for candidate in preferred_candidates:
            if candidate in available_models:
                return candidate
    except Exception as e:
        print(f"[LLM] Could not query models dynamically: {e}")

    return "openai/gpt-oss-120b"


def generate_rag_response(query: str, top_paper_title: str, top_paper_abstract: str) -> str:
    """
    Invokes authorized Groq models with greedy decoding (temperature=0.0).
    Enforces strict grounding on the retrieved academic abstract.
    """
    strict_prompt = f"""You are an objective academic research assistant.
Answer the user's research question STRICTLY and EXCLUSIVELY based on the provided research paper abstract below.
Do not extrapolate, assume, or inject external knowledge.
If the abstract does not contain sufficient details to answer the question, state exactly:
"Based on the retrieved context, this paper does not contain sufficient information to answer the query."

Paper Title: {top_paper_title}
Paper Abstract: {top_paper_abstract}

User Question: {query}

Concise Academic Answer:"""

    if not client:
        return f"Grounded on '{top_paper_title}': {top_paper_abstract[:220]}... [API key missing]"

    active_model = get_active_model_name()

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": strict_prompt}
            ],
            model=active_model,
            temperature=0.0,
            max_tokens=250,
        )
        if chat_completion.choices and chat_completion.choices[0].message.content:
            return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Groq API call with model '{active_model}' failed: {e}")

    return (
        f"Grounded on '{top_paper_title}': {top_paper_abstract[:250]}... "
        f"[Notice: Extractive summary active]"
    )
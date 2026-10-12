import re
from typing import List, Optional

from groq import Groq

# Small, fast, non-reasoning model: planning searches doesn't need a big model,
# and reasoning models can burn a short token budget before emitting any text.
PLANNER_MODEL = "llama-3.1-8b-instant"

SYSTEM = (
    "You are a query planner for a codebase search engine.\n"
    "Rewrite the user's latest question into 2 or 3 short, distinct search queries that would "
    "locate the relevant functions, classes, config or logic in source code. Use identifiers, "
    "file names and technical keywords rather than full sentences. If the question is a follow-up, "
    "use the conversation to resolve what 'it', 'that' or 'this' refer to.\n"
    "Return ONLY the queries, one per line, with no numbering, bullets or commentary."
)


def decompose_query(client: Groq, question: str, history: Optional[List[dict]] = None) -> List[str]:
    """Returns the original question plus up to 3 targeted sub-queries."""
    context = ""
    if history:
        recent = [m for m in history if m["role"] == "user"][-2:]
        if recent:
            context = "Earlier questions:\n" + "\n".join(f"- {m['content'][:200]}" for m in recent) + "\n\n"

    try:
        completion = client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{context}Latest question: {question}"},
            ],
            max_tokens=120,
            temperature=0.1,
        )
        raw = (completion.choices[0].message.content or "").strip()
        subs = [re.sub(r"^[\s\-*\d.)]+", "", ln).strip() for ln in raw.split("\n")]
        subs = [s for s in subs if s]
    except Exception:
        subs = []

    queries = [question]
    for s in subs:
        if s.lower() not in (q.lower() for q in queries):
            queries.append(s)
    return queries[:4]

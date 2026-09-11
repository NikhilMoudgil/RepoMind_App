from typing import List
from groq import Groq

def decompose_query(user_prompt: str, client: Groq) -> List[str]:
    """Breaks a complex user question into 2-3 specific code search queries."""
    system_prompt = (
        "You are a technical query planner for a codebase search engine.\n"
        "Break down the user's question into 2 to 3 distinct, highly focused search queries "
        "designed to locate specific functions, variable declarations, or logic in code.\n"
        "Format requirements:\n"
        "- Return ONLY the search queries, one per line.\n"
        "- Do not include bullet points, numbering, or explanations."
    )

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.2
        )
        
        raw_output = completion.choices[0].message.content.strip()
        sub_queries = [line.strip() for line in raw_output.split("\n") if line.strip()]
        
        # Always include original prompt as fallback
        if user_prompt not in sub_queries:
            sub_queries.append(user_prompt)
            
        return sub_queries[:3]
    except Exception:
        return [user_prompt]
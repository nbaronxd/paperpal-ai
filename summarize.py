import os
from typing import List, Dict, Any
from google import genai

DEFAULT_MODEL = "gemini-2.0-flash-lite"

SUMMARY_PROMPT = """
You are a research paper summarizer.
Summarize the paper using ONLY the provided text.

Output MUST contain these headings exactly and in this order:

1. Research Question
2. Hypothesis
3. Experimental Setup
4. Main Results
5. Conclusion

Rules:
- Be specific (include datasets, metrics, baselines if present).
- If something is not stated in the text, write:
  "Not clearly stated in the paper text provided."
- Keep each section concise (3-8 sentences).
- Do not add any extra headings or sections.
"""

CHAT_SYSTEM = """
You are PaperPal, a careful assistant that answers using ONLY the provided paper text.
If the answer is not in the paper text, say you can't find it in the provided document.
When useful, mention page numbers in brackets like [PAGE 3].
Be concise and precise.
"""

def _get_settings() -> tuple[str, str, str]:
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    project = os.getenv("GCP_PROJECT", "")
    region = os.getenv("GCP_REGION", "us-east1")
    return model, project, region

def _get_client() -> genai.Client:
    _, project, region = _get_settings()
    if not project:
        raise RuntimeError("Missing env var GCP_PROJECT")
    return genai.Client(vertexai=True, project=project, location=region)

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED DUE TO LENGTH]"

def summarize_text(paper_text: str) -> str:
    model, _, _ = _get_settings()
    client = _get_client()

    paper_text = _truncate(paper_text, max_chars=80_000)

    full_prompt = f"{SUMMARY_PROMPT}\n\nPAPER TEXT:\n{paper_text}"

    resp = client.models.generate_content(
        model=model,
        contents=full_prompt
    )
    return resp.text or ""

def chat_with_paper(paper_text: str, user_message: str, history: List[Dict[str, Any]] | None = None) -> str:
    model, _, _ = _get_settings()
    client = _get_client()

    paper_text = _truncate(paper_text, max_chars=80_000)
    history = history or []

    # Build a simple text transcript
    transcript_lines = [f"SYSTEM:\n{CHAT_SYSTEM}\n\nPAPER TEXT:\n{paper_text}\n"]
    for turn in history[-8:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        transcript_lines.append(f"{role.upper()}:\n{content}\n")
    transcript_lines.append(f"USER:\n{user_message}\n")

    resp = client.models.generate_content(
        model=model,
        contents="\n".join(transcript_lines)
    )
    return resp.text or ""
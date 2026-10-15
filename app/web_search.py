"""Optional Tavily web search and source-grounded web summarization."""
from __future__ import annotations
from typing import Any
import requests

from config import TAVILY_API_KEY, GROQ_API_KEY, GROQ_MODEL

TAVILY_ENDPOINT = "https://api.tavily.com/search"


def web_search_configured() -> bool:
    return bool(TAVILY_API_KEY)


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return normalized Tavily results. Returns [] if no key/results are available."""
    if not TAVILY_API_KEY:
        return []
    response = requests.post(
        TAVILY_ENDPOINT,
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": max(1, min(int(max_results), 8)),
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = []
    for item in payload.get("results", []):
        title = str(item.get("title") or "Untitled source")
        url = str(item.get("url") or "")
        content = str(item.get("content") or "").strip()
        if not url or not content:
            continue
        results.append({
            "kind": "web",
            "title": title,
            "url": url,
            "text": content,
            "score": item.get("score"),
            "published_date": item.get("published_date"),
        })
    return results


def summarize_web_results(question: str, results: list[dict[str, Any]], extra_context: str = "") -> str:
    """Summarize web search snippets using only retrieved results and link each result."""
    if not results:
        return "No web results were available. Configure TAVILY_API_KEY to enable sourced web research."
    if not GROQ_API_KEY:
        return "Web sources were found, but GROQ_API_KEY is not configured to summarize them."

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    source_context = []
    for i, result in enumerate(results, 1):
        source_context.append(
            f"[WEB {i}] {result['title']}\nURL: {result['url']}\n"
            f"Snippet: {result['text'][:1800]}"
        )
    joined_sources = "\n\n".join(source_context)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_completion_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": (
                    "You provide cautious, source-grounded crop disease information. "
                    "Use only supplied web snippets. Do not invent pesticides, dosages, "
                    "chemical rates, or regulatory recommendations. Distinguish general "
                    "integrated pest-management guidance from product-specific directions. "
                    "If snippets are insufficient, say so. Cite claims inline as [WEB 1], [WEB 2], etc. "
                    "For suspected plant disease, advise confirming with a local agricultural extension "
                    "service before applying treatments."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nAdditional context: {extra_context}\n\n"
                    f"Retrieved web sources:\n{joined_sources}\n\n"
                    "Give a concise answer, with separate treatment/management and prevention sections "
                    "where relevant. Do not claim that visual AI output is a confirmed diagnosis."
                ),
            },
        ],
    )
    answer = completion.choices[0].message.content or "No summary was generated."
    answer += "\n\n**Web sources**\n"
    answer += "\n".join(f"- [WEB {i}: {r['title']}]({r['url']})" for i, r in enumerate(results, 1))
    return answer

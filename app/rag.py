from __future__ import annotations
from typing import List, Tuple, Dict, Any
import json
import re

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import VECTOR_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL
from web_search import search_web

SYSTEM_PROMPT = """You are AyurVana, a careful, source-grounded assistant for Aṣṭāṅga Saṃgraha and related questions.

BOOK-FIRST RULES
1. Start with the retrieved Aṣṭāṅga Saṃgraha excerpts. Treat them as the primary source.
2. For claims attributed to the book, use only the supplied book excerpts. Cite the exact PDF page number(s) given in the source labels; do not invent citations.
3. Preserve source terminology. Do not silently correct uncertain OCR. If a plant name or term is garbled or ambiguous, say so.
4. Clearly separate: (a) what the classical text says, (b) a cautious explanation, and (c) additional information found on the web.
5. A passage that is merely on the same broad topic is not enough to establish a specific plant-disease indication. Do not infer cures, dosages, preparations, or safety from general Dosha principles.
6. Never present a traditional indication as proof of modern clinical efficacy. For health topics, do not replace qualified professional advice.
7. If the book excerpts are incomplete and web results are supplied, answer the missing part only from those web results and cite them as [WEB 1], [WEB 2], etc. Never invent outside facts or sources.
8. If neither source supports the answer, say that the available sources do not establish it.
9. Be concise but helpful. Prefer a short answer and a clearly labeled Sources section.
"""

BOOK_ONLY_PATTERNS = re.compile(
    r"\b(according to (the )?(book|text)|what does (the )?(book|text) say|"
    r"in the (book|text)|from the (book|text)|ashtanga (sangraha|hridaya)|"
    r"aṣṭāṅga saṃgraha|classical text states|quote the text)\b",
    re.IGNORECASE,
)


def _clean_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"sufficient": False, "reason": "Could not reliably judge source coverage."}


class AyurVanaRAG:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or Streamlit Secrets.")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Vector collection '{COLLECTION_NAME}' was not found. Run `python app\\build_index.py --reset` first."
            ) from exc
        self.groq = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def retrieve(
        self,
        question: str,
        k: int = 6,
        candidate_k: int = 24,
        max_per_page: int = 2,
    ) -> List[dict[str, Any]]:
        count = self.collection.count()
        if count <= 0:
            return []
        n_results = min(max(candidate_k, k), count)
        query_embedding = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        candidates = []
        seen_text = set()
        page_counts: Dict[Any, int] = {}
        for doc, meta, distance in zip(docs, metas, distances):
            meta = meta or {}
            key = _clean_key(doc or "")
            if not key or key in seen_text:
                continue
            page = meta.get("page", "unknown")
            if page_counts.get(page, 0) >= max_per_page:
                continue
            seen_text.add(key)
            page_counts[page] = page_counts.get(page, 0) + 1
            candidates.append({
                "kind": "book",
                "text": doc,
                "page": page,
                "chunk": meta.get("chunk", "unknown"),
                "distance": float(distance),
                "metadata": meta,
            })
            if len(candidates) >= k:
                break
        return candidates

    def _resolve_followup(self, question: str, history: list[dict] | None) -> str:
        """Rewrite terse follow-up questions into standalone retrieval queries."""
        if not history:
            return question
        needs_rewrite = (
            len(question.split()) <= 9
            or re.search(r"\b(it|this|that|these|those|they|them|their|its|there|same|what about|how about|and what|also)\b", question, re.I)
        )
        if not needs_rewrite:
            return question
        recent = self._history_messages(history)
        if not recent:
            return question
        try:
            response = self.groq.chat.completions.create(
                model=self.model,
                temperature=0,
                max_completion_tokens=120,
                messages=[
                    {"role": "system", "content": (
                        "Rewrite the latest user question as one concise, standalone search query using conversation context. "
                        "Do not answer it. Preserve the user's intent and important names. Output only the rewritten query."
                    )},
                    *recent[-6:],
                    {"role": "user", "content": f"Latest question to rewrite: {question}"},
                ],
            )
            rewritten = (response.choices[0].message.content or "").strip().strip('"')
            if 2 <= len(rewritten) <= 500:
                return rewritten
        except Exception:
            pass
        # Safe fallback: don't fail the chat if query rewriting fails.
        return question

    def _judge_book_sufficiency(self, question: str, sources: list[dict], history: list[dict] | None = None) -> dict:
        if not sources:
            return {"sufficient": False, "reason": "No book passages were retrieved."}
        excerpts = "\n\n".join(
            f"[PDF page {s['page']}, chunk {s['chunk']}]\n{s['text'][:1600]}"
            for s in sources[:6]
        )
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {str(m.get('content', ''))[:500]}"
            for m in (history or [])[-4:]
        )
        prompt = f"""Decide whether the supplied excerpts from a classical Ayurveda book contain enough direct evidence to answer the user's current question without making unsupported inferences.
A topic-related passage is NOT sufficient if it does not answer the specific question. For broad list requests, a tiny partial list may be insufficient. Do not judge OCR garbling as reliable evidence.
Return valid JSON only with keys: sufficient (boolean), reason (short string).

Recent conversation:
{history_text}

Question: {question}

Book excerpts:
{excerpts}
"""
        try:
            response = self.groq.chat.completions.create(
                model=self.model,
                temperature=0,
                max_completion_tokens=250,
                messages=[
                    {"role": "system", "content": "You are a strict evidence sufficiency evaluator. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception:
            # Some model/API combinations may not accept JSON response format.
            response = self.groq.chat.completions.create(
                model=self.model,
                temperature=0,
                max_completion_tokens=250,
                messages=[
                    {"role": "system", "content": "You are a strict evidence sufficiency evaluator. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
        parsed = _extract_json(response.choices[0].message.content or "")
        return {
            "sufficient": bool(parsed.get("sufficient", False)),
            "reason": str(parsed.get("reason", "")),
        }

    def _history_messages(self, history: list[dict] | None) -> list[dict]:
        messages = []
        for item in (history or [])[-8:]:
            role = item.get("role")
            content = str(item.get("content", ""))
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content[:1800]})
        return messages

    def _generate_from_book(
        self,
        question: str,
        sources: list[dict],
        history: list[dict] | None = None,
        web_results: list[dict] | None = None,
        book_sufficient: bool = True,
    ) -> str:
        book_context = "\n\n".join(
            f"[BOOK SOURCE {i} | PDF page {s['page']} | chunk {s['chunk']}]\n{s['text']}"
            for i, s in enumerate(sources, 1)
        ) or "No book passages were retrieved."
        web_context = "\n\n".join(
            f"[WEB {i} | {w['title']} | {w['url']}]\n{w['text'][:1800]}"
            for i, w in enumerate(web_results or [], 1)
        )
        if web_results:
            instruction = (
                "First state what the retrieved classical book excerpts support. Then add a clearly separated "
                "'Additional information from the web' section only for relevant details not established by the book. "
                "Cite book claims as [BOOK SOURCE n] or page references, and web claims as [WEB n]."
            )
        elif not book_sufficient:
            instruction = (
                "The book excerpts have been judged insufficient for the specific question. State what, if anything, "
                "they do support, then explicitly state which part cannot be answered from the book. Do not fill gaps "
                "from general model knowledge."
            )
        else:
            instruction = (
                "Answer from the book excerpts only. Separate direct statements from interpretation. Cite the actual "
                "PDF page(s) shown in the source labels."
            )
        user_prompt = f"""Current question: {question}

Book sufficiency: {'sufficient' if book_sufficient else 'insufficient/partial'}

Book excerpts:
{book_context}

Additional web search results (if present):
{web_context or 'None supplied.'}

Task: {instruction}

Do not claim the book says anything not supported by its passages. Since this is a scanned OCR source, flag suspicious or inconsistent terms rather than silently correcting them. Do not describe a traditional plant indication as proven medical efficacy. Keep the answer clear and reasonably concise. Finish with a Sources section containing only sources actually used; include web links when provided."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._history_messages(history))
        messages.append({"role": "user", "content": user_prompt})
        response = self.groq.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.05,
            max_completion_tokens=1800,
        )
        answer = response.choices[0].message.content or "I could not generate an answer from the available sources."
        if web_results:
            # Add robust links even if the model neglects one of the citation markers.
            answer += "\n\n**Web sources**\n" + "\n".join(
                f"- [WEB {i}: {w['title']}]({w['url']})" for i, w in enumerate(web_results, 1)
            )
        return answer

    def answer(
        self,
        question: str,
        k: int = 6,
        candidate_k: int = 24,
        history: list[dict] | None = None,
        allow_web_fallback: bool = True,
    ) -> tuple[str, list[dict]]:
        retrieval_query = self._resolve_followup(question, history)
        sources = self.retrieve(retrieval_query, k=k, candidate_k=candidate_k)
        explicit_book_only = bool(BOOK_ONLY_PATTERNS.search(question))
        sufficiency = self._judge_book_sufficiency(question, sources, history)
        web_results = []
        book_sufficient = sufficiency["sufficient"]
        if not book_sufficient and allow_web_fallback and not explicit_book_only:
            # Search the same topic, not a random general web query.
            web_query = retrieval_query.strip()
            if not re.search(r"\b(plant|disease|crop|ayurveda|ayurvedic|dosha|herb|medicine|treatment|prevention)\b", web_query, re.I):
                web_query += " related to Ayurveda and traditional medicine"
            try:
                web_results = search_web(web_query, max_results=5)
            except Exception as exc:
                web_results = []
                web_error = f"Web search failed: {exc}"
            else:
                web_error = ""
        else:
            web_error = ""
        answer = self._generate_from_book(
            question,
            sources,
            history=history,
            web_results=web_results,
            book_sufficient=book_sufficient,
        )
        if not book_sufficient and allow_web_fallback and not explicit_book_only and not web_results:
            if web_error:
                answer += f"\n\n*Note: The book excerpts were insufficient and web fallback could not be completed ({web_error}).*"
            else:
                answer += (
                    "\n\n*Note: The book excerpts were insufficient. To enable web fallback, configure "
                    "TAVILY_API_KEY in your local .env file or Streamlit Secrets.*"
                )
        return answer, sources + web_results

    def answer_from_web(self, question: str, max_results: int = 5, extra_context: str = "") -> tuple[str, list[dict]]:
        results = search_web(question, max_results=max_results)
        answer = self._generate_web_only(question, results, extra_context=extra_context)
        return answer, results

    def _generate_web_only(self, question: str, results: list[dict], extra_context: str = "") -> str:
        if not results:
            return "No web results were available. Configure TAVILY_API_KEY to enable web-sourced guidance."
        context = "\n\n".join(
            f"[WEB {i} | {item['title']} | {item['url']}]\n{item['text'][:1800]}"
            for i, item in enumerate(results, 1)
        )
        response = self.groq.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_completion_tokens=1300,
            messages=[
                {"role": "system", "content": (
                    "You summarize retrieved web sources cautiously. Use only given sources; do not invent treatment "
                    "products, dosages, or legal directions. Say when evidence is insufficient. For crop disease "
                    "management, favor integrated pest management and authoritative agriculture/extension guidance. "
                    "Cite claims as [WEB 1], [WEB 2], etc."
                )},
                {"role": "user", "content": (
                    f"Question: {question}\nContext: {extra_context}\n\nSources:\n{context}\n\n"
                    "Provide a concise summary with management/treatment and prevention where relevant. "
                    "State that visual model output is not a definitive agronomic diagnosis."
                )},
            ],
        )
        answer = response.choices[0].message.content or "No summary was generated."
        answer += "\n\n**Web sources**\n" + "\n".join(
            f"- [WEB {i}: {w['title']}]({w['url']})" for i, w in enumerate(results, 1)
        )
        return answer

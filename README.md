# AyurVana RAG — Code First

A Python RAG pipeline for the uploaded *Aṣṭāṅga Saṃgraha* (Vol. 1 / Sūtra Sthāna).

## Current pipeline
PDF → OCR → page text → chunks → embeddings → Chroma → retrieval → Groq → source-grounded answer

## Important design choices
- No Flowise or other no/low-code RAG platform.
- The answer is restricted to retrieved source excerpts.
- Retrieval uses a candidate pool and page-diversity filtering to reduce duplicate chunks from the same page.
- The LLM is instructed not to silently repair uncertain OCR.
- Source page/chunk metadata is retained for citations.

## Run
```powershell
python app\query.py "What are the qualities of Vata?"
```

For more retrieval candidates:
```powershell
python app\query.py "What are the qualities of Vata?" --k 6 --candidate-k 18
```

## Next upgrades
1. Better OCR cleanup and page quality evaluation.
2. Chapter/section metadata detection.
3. Query expansion for Sanskrit/common plant names and synonyms.
4. Hybrid lexical + semantic retrieval.
5. Reranking.
6. Plant-identification API/model integration.

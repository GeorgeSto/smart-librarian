from __future__ import annotations
from typing import List, Dict, Any, Optional
import json, re
from .ollama_client import OllamaClient
from .tools import get_summary_by_title

# Functie pentru extragerea unui JSON dintr-un text
def _json_loose(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

# Functie pentru recomandari folosind Ollama
# k = numarul de recomandari dorit
def recommend_with_ollama(
    *, query: str, k: int, theme: Optional[str], collection, books: List[Dict[str, str]]
) -> Dict[str, Any]:
    oc = OllamaClient()
    total_books = len(books)
    k = max(1, min(k, total_books)) 

    
    qtext = query if not theme else f"{query}. Themes: {theme}"
    qvec = oc.embed_query(qtext)
    
    res = collection.query(query_embeddings=[qvec], n_results=k)
    base = []
    # Extragere metadate
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        base.append({
            "title": md["title"],
            "themes": md.get("themes", ""),
            "summary": res["documents"][0][i],
        })

    # Filtrare candidati
    candidates = base
    if theme:
        filtered = [c for c in base if theme.lower() in c["themes"].lower()]
        if filtered:
            candidates = filtered

        if len(candidates) < k:
            seen = {c["title"].lower() for c in candidates}
            for c in base:
                if c["title"].lower() not in seen:
                    candidates.append(c)
                    seen.add(c["title"].lower())
                if len(candidates) >= k:
                    break

    candidates = candidates[:k]

    # Formatare mesaje pentru chat
    lines = "\n".join([f'- "{c["title"]}" (teme: {c["themes"]})' for c in candidates])
    messages = [
        {"role": "system", "content": "Esti Smart Librarian. Raspunzi în romana, concis."},
        {"role": "user", "content": (
            f"Intrebarea utilizatorului: {query}\n"
            f"Candidati (alege-le pe toate și da un motiv scurt pentru fiecare):\n{lines}\n\n"
            'Returneaza doar JSON: {"items":[{"title":"<titlu>","reason":"<motiv scurt>"} ...]}'
        )}
    ]
    content, _ = oc.chat(messages, temperature=0.2)
    data = {}
    try:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            data = json.loads(m.group(0))
    except Exception:
        data = {}
    # Extrage motivele din raspuns
    reasons_map = {item["title"]: item.get("reason","") for item in (data.get("items") or []) if "title" in item}
    # Asociaza motivele cu candidatii
    recommendations: List[Dict[str, str]] = []
    for c in candidates:
        title = c["title"]
        summary_full = get_summary_by_title(title) or c["summary"]
        reason = reasons_map.get(title, "")
        recommendations.append({
            "title": title,
            "summary": summary_full,
            "themes": [t.strip() for t in c["themes"].split(",")] if c["themes"] else [],
            "reason": reason
        })
    # Formatare raspuns
    if k == 1:
        t = recommendations[0]["title"]
        ans = f"Iti recomand **{t}**. {('Motiv: '+ recommendations[0]['reason']) if recommendations[0]['reason'] else ''}\n\nRezumat:\n{recommendations[0]['summary']}"
    else:
        bullets = "\n".join([f"- **{r['title']}** — {r['reason']}".rstrip(" —") for r in recommendations])
        ans = f"Iata {k} recomandari pentru tine:\n{bullets}"

    return {
        "answer": ans,
        "recommendations": recommendations,
    }
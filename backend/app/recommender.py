
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json, re, os
from openai import OpenAI
from .tools import OPENAI_TOOLS, get_summary_by_title

def _json_in_text(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
# Functie pentru recomandari folosind OpenAI si functii
def recommend_with_tools(
    *,
    query: str,
    k: int,
    theme: Optional[str],
    client: OpenAI,
    collection,                 
    books: List[Dict[str, str]] 
) -> Dict[str, Any]:
    total = len(books)
    topn = max(5, min(8, total))

    
    qtext = query if not theme else f"{query}. Themes: {theme}"
    qvec = client.embeddings.create(model="text-embedding-3-small", input=qtext).data[0].embedding
    res = collection.query(query_embeddings=[qvec], n_results=topn)

    candidates = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        candidates.append({
            "title": md["title"],
            "themes": md.get("themes", ""),
            "summary": res["documents"][0][i],  
        })
    # Formatare titluri
    titles_line = "\n".join([f'- "{c["title"]}" (teme: {c["themes"]})' for c in candidates])

    # Formatare mesaje pentru chat
    messages = [
        {"role": "system", "content": "Esti Smart Librarian. Raspunzi în romana, concis."},
        {"role": "user", "content": (
            f"Intrebarea utilizatorului: {query}\n"
            f"Tema: {theme or '-'}\n"
            f"Candidati:\n{titles_line}\n\n"
            "Alege o carte potrivita și apeleaza functia `get_summary_by_title` cu titlul ales. "
            "Dupa ce primesti rezumatul complet, compune raspunsul final cu recomandarea și rezumatul."
        )},
    ]
    # Configurare model chat
    chat_model = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"

    first = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        tools=OPENAI_TOOLS,   
        tool_choice="auto",
        temperature=0.2,
    )
    # Extrage mesajele din raspuns
    msg1 = first.choices[0].message
    tool_msgs = []
    selected_title = None

    # Extrage argumentele din apelurile de functie
    if msg1.tool_calls:
        for tc in msg1.tool_calls:
            if tc.function.name == "get_summary_by_title":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                selected_title = (args.get("title") or "").strip()

                # Extrage rezumatul complet
                full = get_summary_by_title(selected_title) or ""
                if not full:
                    for c in candidates:
                        if c["title"].lower() == selected_title.lower():
                            full = c["summary"]
                            break

                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": full or "NOT_FOUND",
                })
    # Formatare mesaje finale
    final_msgs = messages + [msg1] + tool_msgs
    second = client.chat.completions.create(
        model=chat_model,
        messages=final_msgs,
        temperature=0.2,
    )
    answer = second.choices[0].message.content or ""
    # Extrage titlul selectat
    if not selected_title:
        for c in candidates:
            if c["title"].lower() in (answer.lower()):
                selected_title = c["title"]
                break
    if not selected_title and candidates:
        selected_title = candidates[0]["title"]
    # Extrage rezumatul complet
    summary_full = get_summary_by_title(selected_title) if selected_title else ""
    if not summary_full and selected_title:
        for c in candidates:
            if c["title"].lower() == selected_title.lower():
                summary_full = c["summary"]
                break
    # Extrage temele
    themes_list: List[str] = []
    for c in candidates:
        if c["title"].lower() == (selected_title or "").lower():
            themes_list = [t.strip() for t in (c["themes"] or "").split(",")] if c["themes"] else []
            break

    selected = {
        "title": selected_title,
        "summary": summary_full,
        "themes": themes_list,
        "reason": ""  
    }

    recommendations = [selected]

    return {
        "answer": answer,
        "recommendations": recommendations,
        "used_tool": bool(msg1.tool_calls),
    }

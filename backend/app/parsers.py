from pathlib import Path
import json
from typing import List, Dict

# Functie pentru parsarea fisierului cu carti pentru a-l transforma intr-o lista curata
# normalizez structura , elimin duplicatele si returneaza o lista de carti
def parse_book_summaries(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of book objects")
    books, seen = [], set()
    for item in data:
        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()
        themes = item.get("themes", [])
        if not title or not summary or title.lower() in seen:
            continue
        seen.add(title.lower())
        books.append({"title": title, "summary": summary, "themes": themes})
    return books

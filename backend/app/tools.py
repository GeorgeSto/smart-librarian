from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Dict
import json

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "book_summaries.json"

# incarc rezumatele cartilor intr-un dictionar titlu -> rezumat
@lru_cache(maxsize=1)
def _load_summaries() -> Dict[str, str]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {
        (item.get("title") or "").strip(): (item.get("summary") or "").strip()
        for item in data if item.get("title") and item.get("summary")
    }

# functie pentru obtinerea rezumatului dupa titlu
def get_summary_by_title(title: str) -> str:
    title = (title or "").strip()
    books = _load_summaries()
    if title in books:
        return books[title]
    low = title.lower()
    for k, v in books.items():
        if k.lower() == low:
            return v
    return ""

# lista de functii disponibile pentru OpenAI
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returneaza rezumatul complet pentru un titlu exact de carte.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Titlul exact al cărtii."
                    }
                },
                "required": ["title"],
                "additionalProperties": False
            }
        }
    }
]
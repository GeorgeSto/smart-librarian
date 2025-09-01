from __future__ import annotations
import os, re, unicodedata
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

SAFETY_ON = (os.getenv("SAFETY_ON") or "1").strip().lower() not in ("0", "false", "off", "")
BLOCKLIST_PATH = Path(os.getenv("BLOCKLIST_PATH") or Path(__file__).resolve().parent.parent / "data" / "blocklist.txt")

# normalizez textul pentru verificarea termenilor din blocklist
def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    # scot diacriticele
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # pastrez doar litere/cifre ca separatori de cuvinte
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# incarc lista de termeni interzisi
def _load_blocklist() -> list[str]:
    if not BLOCKLIST_PATH.exists():
        return []
    items = []
    for line in BLOCKLIST_PATH.read_text(encoding="utf-8").splitlines():
        t = line.strip().lower()
        if not t or t.startswith("#"):
            continue
        items.append(_norm(t))
    return sorted(set(items))

BLOCKED = _load_blocklist()
# verific daca textul contine termeni interzisi
def check_inappropriate(text: str) -> tuple[bool, list[str]]:
    norm = _norm(text)
    hits: list[str] = []
    for term in BLOCKED:
        if not term:
            continue
        if re.search(rf"\b{re.escape(term)}\b", norm):
            hits.append(term)
    return (len(hits) > 0, hits)

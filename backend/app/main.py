
import os, asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from .safety import SAFETY_ON, check_inappropriate
from .recommender_ollama import recommend_with_ollama
from .ollama_client import OllamaClient
from .parsers import parse_book_summaries
from .ingest_chroma import build_chroma
from .recommender import recommend_with_tools
from .tools import get_summary_by_title

# definirea cailor pentru date
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "book_summaries.json"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_store"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


SAFETY_ON = (os.getenv("SAFETY_FILTER", "on").lower() in ("on", "true", "1"))



load_dotenv(dotenv_path=ENV_PATH, override=True)

# aleg providerul
PROVIDER = (os.getenv("EMBEDDINGS_PROVIDER") or "ollama").lower()

# incercare pentru OpenAI
OA = None
if PROVIDER == "openai":
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY lipsește. Verifică backend/.env")
    OA = OpenAI(
        api_key=key,
        organization=os.getenv("OPENAI_ORG_ID") or None,
        project=os.getenv("OPENAI_PROJECT_ID") or None,
    )


app = FastAPI(title="Smart Librarian")

# configurare CORS pentru frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class Book(BaseModel):
    title: str
    summary: str

# request pentru recomandari
class RecommendRequest(BaseModel):
    query: str
    k: int = 5
    theme: Optional[str] = None

BOOKS: list[Book] = []
COLLECTION = None
INIT_TASK: asyncio.Task | None = None
INIT_STATE = {"status": "idle", "error": None}
PARSED: list[dict] = []


# functie pentru embedderi
def embed_query(text: str) -> list[float]:
    if PROVIDER == "ollama":
        return OllamaClient().embed_query(text)
    elif PROVIDER == "openai":
        r = OA.embeddings.create(model="text-embedding-3-small", input=text)
        return r.data[0].embedding
    else:
        from sentence_transformers import SentenceTransformer
        global _q_model
        try:
            _q_model
        except NameError:
            _q_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _q_model.encode(text).tolist()


# initializare Chroma in fundal
async def _init_chroma_bg():
    global COLLECTION, INIT_STATE
    INIT_STATE.update(status="initializing", error=None)
    try:
        COLLECTION = await asyncio.to_thread(build_chroma, CHROMA_DIR, PARSED)
        INIT_STATE.update(status="ready", error=None)
    except Exception as e:
        COLLECTION = None
        INIT_STATE.update(status="failed", error=str(e)[:300])

@app.on_event("startup")
async def startup():
    global BOOKS, PARSED, INIT_TASK
    print(f"[Startup] Provider embeddings: {PROVIDER}")
    if PROVIDER == "openai":
        key = os.getenv("OPENAI_API_KEY")
        print("OPENAI_API_KEY:", f"{(key or '')[:6]}...{(key or '')[-4:]}" if key else "absent")

    PARSED = parse_book_summaries(DATA_PATH)
    BOOKS = [Book(title=b["title"], summary=b["summary"]) for b in PARSED]
    INIT_TASK = asyncio.create_task(_init_chroma_bg())


@app.get("/")
def root():
    return {"message": "Smart Librarian", "try": ["/health", "/books", "/search?q=...&theme=...&k=5"], "docs": "/docs"}

# Endpoint pentru verificarea starii
@app.get("/health")
async def health():
    state = INIT_STATE.get("status", "idle")
    chroma = (
        "ready" if COLLECTION else
        "initializing" if state == "initializing" else
        "failed" if state == "failed" else
        "not-started"
    )
    return {
        "status": "ok",
        "books_loaded": len(BOOKS),
        "chroma": chroma,
        "error": INIT_STATE.get("error") if chroma == "failed" else None
    }

# Lista toate cartile
@app.get("/books", response_model=list[Book])
def list_books():
    return BOOKS

# Detalii despre o carte
@app.get("/books/{title}", response_model=Book)
def get_book(title: str):
    for b in BOOKS:
        if b.title.lower() == title.lower():
            return b
    raise HTTPException(status_code=404, detail="Book not found")

# Cautare semantica
@app.get("/search")
def semantic_search(
    q: str = Query(..., description="cautare după context"),
    theme: Optional[str] = Query(None, description="filtru după tema"),
    k: int = Query(5, ge=1, le=10, description="top-k rezultate")
):
    # verificare continut inadecvat
    if SAFETY_ON:
        to_check = " ".join([q or "", theme or ""])
        bad, hits = check_inappropriate(to_check)
        if bad:
            return {
                "safe": False,
                "results": [],
                "message": "Te rog folosește un limbaj adecvat.",
                "blocked_terms": hits,
            }
    # verificare stare Chroma
    if not COLLECTION:
        raise HTTPException(503, "Chroma initializing, try again in a few seconds")

    # generare vector de interogare
    query_text = q if not theme else f"{q}. Themes: {theme}"
    qvec = embed_query(query_text)

    raw_k = max(k, 8) if theme else k
    res = COLLECTION.query(query_embeddings=[qvec], n_results=raw_k)

    # filtrare rezultate
    results = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        themes_str = md.get("themes", "")
        if theme and theme.lower() not in themes_str.lower():
            continue
        dist = res.get("distances", [[None]])[0][i]
        sim = (1 - dist) if dist is not None else None
        results.append({
            "id": res["ids"][0][i],
            "title": md["title"],
            "themes": [t.strip() for t in themes_str.split(",")] if themes_str else [],
            "similarity": sim,
            "snippet": res["documents"][0][i][:180] + "..."
        })
    results.sort(key=lambda r: (-(r["similarity"] or 0)))
    return {"query": {"q": q, "theme": theme, "k": k}, "results": results[:k]}

@app.post("/admin/rebuild")
async def admin_rebuild():
    import shutil
    global INIT_TASK, COLLECTION, INIT_STATE
    if INIT_TASK and not INIT_TASK.done():
        return {"status": "initializing"}
    COLLECTION = None
    INIT_STATE.update(status="initializing", error=None)
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    INIT_TASK = asyncio.create_task(_init_chroma_bg())
    return {"status": "started"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if SAFETY_ON:
        to_check = " ".join([req.query or "", req.theme or ""])
        bad, hits = check_inappropriate(to_check)
        if bad:
            return {
                "safe": False,
                "answer": (
                    "Un limbaj respectuos te rog, reformulează întrebarea "
                ),
                "blocked_terms": hits,
            }

    
    if not COLLECTION:
        raise HTTPException(503, "Chroma initializing or failed")
    books_plain = [{"title": b.title, "summary": b.summary} for b in BOOKS]

    if PROVIDER == "ollama":
        result = recommend_with_ollama(
            query=req.query, k=req.k, theme=req.theme,
            collection=COLLECTION, books=books_plain
        )
    else:
        result = recommend_with_tools(
            query=req.query, k=req.k, theme=req.theme,
            client=OA, collection=COLLECTION, books=books_plain,
        )
    return {"safe": True, "query": req.query, "theme": req.theme, **result}

# Request pentru chat
class OllamaChatReq(BaseModel):
    messages: list[dict]
    temperature: float = 0.4

@app.get("/ollama/health")
def ollama_health():
    return OllamaClient().health()


class OllamaEmbedReq(BaseModel):
    texts: list[str]

@app.post("/ollama/embeddings")
def ollama_embeddings(req: OllamaEmbedReq):
    vecs = OllamaClient().embed_texts(req.texts)
    return {"count": len(vecs), "dim": len(vecs[0]) if vecs else None}

@app.get("/tools/summary")
def tool_summary(title: str):
    s = get_summary_by_title(title)
    if not s:
        raise HTTPException(404, f"Nu am gasit rezumat pentru: {title}")
    return {"title": title, "summary": s}

@app.get("/safety/check")
def safety_check(q: str):
    bad, hits = check_inappropriate(q)
    return {"input": q, "bad": bad, "hits": hits}

class OllamaChatReq(BaseModel):
    messages: list[dict]
    temperature: float = 0.4

@app.post("/ollama/chat")
def ollama_chat(req: OllamaChatReq):
    if SAFETY_ON:
        whole = " ".join([(m.get("content", "") or "") for m in req.messages])
        bad, hits = check_inappropriate(whole)
        if bad:
            return {
                "safe": False,
                "content": (
                    "Limbaj nepotrivit."
                    "Te rog reformulează mesajul."
                ),
                "blocked_terms": hits,
            }
        
    content, raw = OllamaClient().chat(req.messages, temperature=req.temperature)
    return {"safe": True, "content": content, "raw": raw}



from pathlib import Path
from typing import List, Dict, Any
import os
import chromadb
from openai import OpenAI
from .ollama_client import OllamaClient

# convertesc datele unui document in metadate pentru Chroma
def _to_metadata(d: Dict[str, Any]) -> Dict[str, str]:
    themes = d.get("themes", [])
    themes = ", ".join(themes) if isinstance(themes, list) else (str(themes) if themes is not None else "")
    return {"title": d["title"], "themes": themes}

# functie pentru embedderi
def _embed_texts_openai(texts: List[str]) -> List[List[float]]:
    client = OpenAI(
        api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        organization=os.getenv("OPENAI_ORG_ID") or None,
        project=os.getenv("OPENAI_PROJECT_ID") or None,
    )
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


def _embed_texts_ollama(texts: List[str]) -> List[List[float]]:
    return OllamaClient().embed_texts(texts)

_local_model = None
def _embed_texts_local(texts: List[str]) -> List[List[float]]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return [_local_model.encode(t).tolist() for t in texts]

# aleg providerul
def _embed_texts(texts: List[str]) -> List[List[float]]:
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or "ollama").lower()
    if provider == "ollama":
        return _embed_texts_ollama(texts)
    if provider == "openai":
        return _embed_texts_openai(texts)
    return _embed_texts_local(texts)

# construiesc baza de date Chroma
def build_chroma(persist_dir: Path, docs: List[Dict[str, str]]):
    persist_dir.mkdir(parents=True, exist_ok=True)
    client_db = chromadb.PersistentClient(path=str(persist_dir))
    collection = client_db.get_or_create_collection(name="books", metadata={"hnsw:space": "cosine"})
    ids = [f"book-{i}" for i, _ in enumerate(docs, start=1)]
    texts = [d["summary"] for d in docs]
    metadatas = [_to_metadata(d) for d in docs]
    vectors = _embed_texts(texts)
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
    return collection


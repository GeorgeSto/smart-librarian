import os, httpx
from typing import List, Tuple, Any, Dict

# Client pentru interacțiunea cu API-ul Ollama
class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.chat_model = chat_model or os.getenv("OLLAMA_CHAT_MODEL") or "llama3.1:8b"
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text:latest"

        t = float(os.getenv("OLLAMA_HTTP_TIMEOUT") or (timeout or 300))
        self.http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10.0, read=t, write=t, pool=t),
        )

    # Request pentru chat
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.4, format: str | None = None) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.chat_model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        if format:
            payload["format"] = format
        r = self.http.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        return content, data

    # Request pentru generarea embedding-urilor
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vecs: List[List[float]] = []
        for t in texts:
            r = self.http.post("/api/embeddings", json={"model": self.embed_model, "prompt": t})
            r.raise_for_status()
            vecs.append(r.json()["embedding"])
        return vecs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def models(self) -> Dict[str, Any]:
        r = self.http.get("/api/tags")
        r.raise_for_status()
        return r.json()

    # Verificare stare API
    def health(self) -> Dict[str, Any]:
        try:
            self.models()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

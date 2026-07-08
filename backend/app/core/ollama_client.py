"""
Thin async wrapper around the Ollama HTTP API.
Handles: chat completions, embeddings, streaming, model health check.
"""
import httpx
import json
from typing import AsyncIterator
from config.settings import get_settings

settings = get_settings()


class OllamaClient:
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.timeout = httpx.Timeout(120.0, connect=10.0)

    async def health_check(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"status": "ok", "models": models}

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        stream: bool = False,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if not data.get("done"):
                            yield data["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        payload = {"model": settings.ollama_embed_model, "prompt": text}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/api/embeddings", json=payload
            )
            r.raise_for_status()
            return r.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# Singleton
_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client

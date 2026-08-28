"""SchemeSetu — one LLM seam, many providers. How the project stays cost-free.

Every LLM call in this project goes through `complete(system, user)` on a
provider object. That one-method seam (decisions.md 012) now pays its second
dividend: which provider answers is *configuration*, not code. Free options —
none of these require a card:

  provider    key env var        get a key at
  ---------   ----------------   -------------------------------------------
  groq        GROQ_API_KEY       console.groq.com  (fast Llama models)
  gemini      GEMINI_API_KEY     aistudio.google.com
  cerebras    CEREBRAS_API_KEY   cloud.cerebras.ai
  ollama      (none — local)     ollama.com  (runs small models on your Mac)
  anthropic   ANTHROPIC_API_KEY  console.anthropic.com  (paid, PRD default)

WHY ONE TINY CLIENT COVERS FOUR PROVIDERS: most hosted LLM APIs speak the
same wire format — OpenAI's `/chat/completions` JSON shape — because it became
the industry's de-facto lingua franca. So `OpenAICompatLLM` below is ~20 lines
of httpx (which ships with the anthropic SDK: zero new dependencies), and
Groq, Gemini, Cerebras and Ollama are just different base URLs. Anthropic's
native API has its own (richer) format, so it keeps its own class.

Selection order: the LLM_PROVIDER env var wins if set; otherwise the first
provider with a key present (anthropic → groq → gemini → cerebras); otherwise
a local Ollama server if one is listening. LLM_MODEL overrides the default
model id — provider catalogs drift over time, and the fix is one env var, not
a code change.

Caveat that belongs in your interview answer: free tiers are rate-limited
(requests/minute and /day). Fine for development and small eval runs; a real
deployment budgets for paid calls or self-hosts. Cost-free is a constraint we
engineered around, not a free lunch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from naive.rag import GEN_MODEL  # noqa: E402  (the anthropic-provider default)

PROVIDERS: dict[str, dict] = {
    "anthropic": {"key_env": "ANTHROPIC_API_KEY", "model": GEN_MODEL},
    # Model ids verified against the live catalogs on 2026-08-29 — they WILL
    # drift again; when a provider 404s, list its models (see decisions 013)
    # and update here, or override instantly with LLM_MODEL=...
    "groq": {"key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b",
             "base_url": "https://api.groq.com/openai/v1"},
    "gemini": {"key_env": "GEMINI_API_KEY", "model": "gemini-3.6-flash",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "cerebras": {"key_env": "CEREBRAS_API_KEY", "model": "gpt-oss-120b",  # unverified (no key)
                 "base_url": "https://api.cerebras.ai/v1"},
    "ollama": {"key_env": None, "model": "llama3.2",
               "base_url": "http://localhost:11434/v1"},
}


class OpenAICompatLLM:
    """Minimal client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str | None, model: str):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        import time
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Floor the token budget: reasoning models (e.g. gpt-oss) spend tokens
        # thinking BEFORE the visible answer — a tiny cap yields empty content.
        payload = {"model": self.model, "max_tokens": max(max_tokens, 512),
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        for attempt in range(4):
            resp = httpx.post(f"{self.base_url}/chat/completions",
                              headers=headers, timeout=120, json=payload)
            if resp.status_code == 429 and attempt < 3:
                # HTTP 429 comes in two species. A per-minute meter says
                # "slow down" — waiting works. A DAILY quota says "come back
                # tomorrow" — retrying is hopeless and stalls the caller for
                # minutes before its fallback can engage. Distinguish them:
                # a long Retry-After or an explicit quota message = terminal.
                wait = float(resp.headers.get("retry-after", 5 * (attempt + 1)))
                if wait > 30 or "quota" in resp.text.lower():
                    break  # terminal — raise below so callers can fall back
                time.sleep(wait)
                continue
            break
        if resp.status_code >= 400:
            # Status codes alone don't debug anything — surface the body.
            raise RuntimeError(
                f"{self.base_url} returned {resp.status_code}: {resp.text[:400]}")
        message = resp.json()["choices"][0]["message"]
        return (message.get("content") or "").strip()


class ClaudeLLM:
    """Native Anthropic client — the PRD default when a key is present."""

    def __init__(self, model: str = GEN_MODEL):
        import anthropic
        self._client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in response.content if b.type == "text").strip()


def available_provider() -> str | None:
    """The provider that would be used, or None if nothing is configured."""
    forced = os.environ.get("LLM_PROVIDER")
    if forced:
        return forced if forced in PROVIDERS else None
    for name in ("anthropic", "groq", "gemini", "cerebras"):
        if os.environ.get(PROVIDERS[name]["key_env"]):
            return name
    try:  # last resort: a local Ollama server needs no key at all
        import httpx
        httpx.get(PROVIDERS["ollama"]["base_url"] + "/models", timeout=1)
        return "ollama"
    except Exception:
        return None


def get_llm():
    """Build the configured provider, or exit with setup guidance."""
    name = available_provider()
    if name is None:
        sys.exit(
            "No LLM provider configured. Free options (no card needed):\n"
            "  Groq:   create a key at console.groq.com, then\n"
            "          export GROQ_API_KEY=...   (add to ~/.zshrc to persist)\n"
            "  Gemini: aistudio.google.com  ->  export GEMINI_API_KEY=...\n"
            "  Local:  install Ollama (ollama.com), run `ollama pull llama3.2`\n"
            "Or the PRD default (paid): export ANTHROPIC_API_KEY=...\n"
            "Force a specific one with LLM_PROVIDER=groq|gemini|cerebras|ollama|anthropic")
    cfg = PROVIDERS[name]
    model = os.environ.get("LLM_MODEL", cfg["model"])
    if name == "anthropic":
        return ClaudeLLM(model=model)
    key = os.environ.get(cfg["key_env"]) if cfg["key_env"] else None
    return OpenAICompatLLM(cfg["base_url"], key, model)


if __name__ == "__main__":  # tiny smoke test:  python agent/llm.py
    provider = available_provider()
    print(f"provider: {provider or 'none configured'}")
    if provider:
        print("reply:", get_llm().complete(
            "Reply with exactly one short sentence.", "Say hello.", max_tokens=30))

"""Language-model clients: an OpenAI-compatible HTTP client with a disk cache and a budget,
and an offline stand-in for tests and demos."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from timefence.core.errors import BudgetExceeded

Messages = list[dict[str, str]]

PLACEHOLDER_KEY = "your-deepseek-api-key"  # the value shipped in .env.example


def api_key_from_env(name: str) -> str:
    """Return the API key held in environment variable `name`.

    Raises if the variable is unset or still holds the placeholder from `.env.example`, so a
    run stops before any work instead of failing on its first model call.
    """
    key = os.environ.get(name, "").strip()
    if not key:
        raise RuntimeError(
            f"environment variable {name} is not set; set it to your API key or put it in .env "
            "(never put the key in code or committed files)"
        )
    if key == PLACEHOLDER_KEY:
        raise RuntimeError(
            f"{name} is still the placeholder from .env.example; replace it with your real key"
        )
    return key


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


@dataclass
class Budget:
    """Hard caps on model usage for one run. Cached replies are free."""

    max_calls: int = 300
    max_tokens: int = 3_000_000
    calls: int = 0
    tokens: int = 0

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"reached the limit of {self.max_calls} model calls")
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"reached the limit of {self.max_tokens} tokens")

    def charge(self, reply: LLMReply) -> None:
        if not reply.cached:
            self.calls += 1
            self.tokens += reply.prompt_tokens + reply.completion_tokens


class LLM(Protocol):
    budget: Budget
    name: str

    def complete(self, messages: Messages, temperature: float = 0.7) -> LLMReply:
        """Return the model's reply. Replies are expected to be a JSON object."""
        ...


@dataclass
class OpenAICompatibleLLM:
    """Any chat-completions endpoint that speaks the OpenAI protocol. DeepSeek by default.

    Every request and reply is cached under `cache_dir`, keyed by a hash of the request.
    Re-running an experiment therefore replays the same replies: results are reproducible
    and cost nothing the second time. The API key is read from the environment only.
    """

    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    # DeepSeek's thinking switch, which its API turns on by default. "disabled" keeps replies
    # fast and lets temperature take effect; None leaves the field out for other providers.
    thinking: str | None = "disabled"
    cache_dir: Path = Path(".cache/llm")
    budget: Budget = field(default_factory=Budget)
    timeout: float = 120.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self._client = None

    @property
    def name(self) -> str:
        mode = f", thinking {self.thinking}" if self.thinking else ""
        return f"{self.model}{mode} @ {self.base_url}"

    def _connect(self):
        if self._client is None:
            key = api_key_from_env(self.api_key_env)
            from openai import OpenAI

            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def complete(self, messages: Messages, temperature: float = 0.7) -> LLMReply:
        request = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.thinking is not None:
            request["thinking"] = self.thinking
        key = hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            reply = LLMReply(
                saved["text"], saved["prompt_tokens"], saved["completion_tokens"], cached=True
            )
            self.budget.charge(reply)
            return reply

        self.budget.check()
        options: dict = {}
        if self.thinking is not None:
            options["extra_body"] = {"thinking": {"type": self.thinking}}
        if self.thinking != "enabled":
            # DeepSeek advises against JSON mode together with thinking. The prompts ask for
            # JSON either way, and only `content` is read: the reasoning is a separate field.
            options["response_format"] = {"type": "json_object"}
        response = self._connect().chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, **options
        )
        usage = response.usage
        reply = LLMReply(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"request": request, **reply.__dict__}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        self.budget.charge(reply)
        return reply


@dataclass
class ScriptedLLM:
    """Replies come from a Python function instead of a model. For tests and offline demos.

    Nothing it produces is a language model's work; runs that use it say so in their report.
    """

    respond: Callable[[Messages], str]
    budget: Budget = field(default_factory=Budget)
    name: str = "scripted stand-in (not a language model)"
    log: list[Messages] = field(default_factory=list)

    def complete(self, messages: Messages, temperature: float = 0.7) -> LLMReply:
        self.budget.check()
        self.log.append(messages)
        text = self.respond(messages)
        reply = LLMReply(text, prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
                         completion_tokens=len(text) // 4)  # fmt: skip
        self.budget.charge(reply)
        return reply


def parse_json(text: str) -> dict:
    """Parse a model reply as a JSON object, tolerating a Markdown code fence around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value

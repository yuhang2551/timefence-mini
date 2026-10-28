"""Language-model clients behind one small interface.

K2 Horizon, DeepSeek and any other OpenAI-compatible endpoint go through `OpenAICompatibleLLM`;
Claude goes through the official Anthropic SDK (`AnthropicLLM`). Both cache every reply on
disk and count usage against a budget. `ScriptedLLM` is an offline stand-in for tests and demos.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from timefence.core.errors import BudgetExceeded

Messages = list[dict[str, str]]

PLACEHOLDER_KEY = "your-deepseek-api-key"  # what earlier versions shipped in .env.example


@dataclass(frozen=True)
class Provider:
    """Defaults for one model provider.

    `knowledge_bound` is the first day the registered models (the default model and those in
    `same_bound`) cannot know about: the day after their published knowledge cutoff, or the
    release date when no cutoff is published. The holdout window starts there unless a run sets
    another date.
    """

    model: str
    api_key_env: str
    knowledge_bound: str
    cutoff_published: bool  # False: no published cutoff, so the bound is the release date
    same_bound: tuple[str, ...] = ()  # other models with the same knowledge bound
    api: str = "openai"  # "openai": an OpenAI-compatible endpoint; "anthropic": the Anthropic SDK
    base_url: str | None = None
    base_url_env: str | None = None  # an environment variable that overrides `base_url`
    model_env: str | None = None  # for hosts that serve the same model under another name
    thinking: str | None = None  # DeepSeek's thinking switch
    efforts: tuple[str, ...] = ()  # accepted reasoning efforts, for models that take one
    fixed_params: dict | None = None  # sent on every request instead of the caller's temperature

    def default_model(self) -> str:
        return (os.environ.get(self.model_env, "").strip() if self.model_env else "") or self.model

    def default_base_url(self) -> str | None:
        url = os.environ.get(self.base_url_env, "").strip() if self.base_url_env else ""
        return url or self.base_url

    def bound_for(self, model: str) -> str | None:
        """The knowledge bound of `model`, or None when it is not registered here."""
        return self.knowledge_bound if model in (self.default_model(), *self.same_bound) else None


PROVIDERS: dict[str, Provider] = {
    # Claude Haiku 5.5 is the cheapest current Claude model. Claude Opus 5.5 has the same
    # knowledge cutoff (June 2026), so `--model claude-opus-5-5` keeps the same holdout.
    "claude": Provider(
        model="claude-haiku-5-5",
        same_bound=("claude-opus-5-5",),
        api_key_env="ANTHROPIC_API_KEY",
        knowledge_bound="2026-07-01",
        cutoff_published=True,
        api="anthropic",
        efforts=("low", "medium", "high", "xhigh", "max"),
    ),
    # K2 Horizon (MBZUAI's Institute of Foundation Models) through IFM's own API; K2_BASE_URL
    # points it at another host, and K2_MODEL covers a host that names the model differently.
    # It always reasons; its model card recommends high effort, temperature 1.0, top_p 0.95
    # and room for at least 32,768 output tokens.
    "k2": Provider(
        model="IFM/K2-Horizon-375B-A23B",
        api_key_env="K2_API_KEY",
        knowledge_bound="2026-09-03",
        cutoff_published=False,
        base_url="https://api.ifm.ai/v1",
        base_url_env="K2_BASE_URL",
        model_env="K2_MODEL",
        efforts=("low", "medium", "high"),
        fixed_params={"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768},
    ),
    # The deepseek-flash name moves to newer models over time; the bound is the release date of
    # the model serving it now (V4.1-Flash-1017).
    "deepseek": Provider(
        model="deepseek-flash",
        api_key_env="DEEPSEEK_API_KEY",
        knowledge_bound="2026-10-17",
        cutoff_published=False,
        base_url="https://api.deepseek.com",
        thinking="disabled",
    ),
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}") from None


def api_key_from_env(name: str) -> str:
    """Return the API key held in environment variable `name`.

    Raises if the variable is unset, empty or still a placeholder, so a run stops before any
    work instead of failing on its first model call.
    """
    key = os.environ.get(name, "").strip()
    if not key:
        raise RuntimeError(
            f"environment variable {name} is not set; set it to your API key or put it in .env "
            "(never put the key in code or committed files)"
        )
    if key == PLACEHOLDER_KEY:
        raise RuntimeError(f"{name} is still a placeholder; replace it with your real key")
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


def _cache_path(cache_dir: Path, request: dict) -> Path:
    key = hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return cache_dir / key[:2] / f"{key}.json"


def _read_cache(path: Path) -> LLMReply | None:
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return LLMReply(saved["text"], saved["prompt_tokens"], saved["completion_tokens"], cached=True)


def _write_cache(path: Path, request: dict, reply: LLMReply) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"request": request, **reply.__dict__}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


@dataclass
class OpenAICompatibleLLM:
    """Any chat-completions endpoint that speaks the OpenAI protocol (K2 Horizon, DeepSeek, ...).

    Every request and reply is cached under `cache_dir`, keyed by a hash of the request.
    Re-running an experiment therefore replays the same replies: results are reproducible
    and cost nothing the second time. The API key is read from the environment only.
    """

    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    # DeepSeek's thinking switch, which DeepSeek turns on by default; "disabled" keeps replies
    # fast. None leaves the field out for providers without one.
    thinking: str | None = "disabled"
    # K2 Horizon's reasoning depth (low, medium or high), passed to its chat template.
    reasoning_effort: str | None = None
    # Request fields sent on every call, such as a model's recommended sampling; they replace
    # the caller's temperature.
    fixed_params: dict | None = None
    cache_dir: Path = Path(".cache/llm")
    budget: Budget = field(default_factory=Budget)
    timeout: float = 600.0  # reasoning models can take minutes per reply

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self._client = None

    @property
    def name(self) -> str:
        mode = f", thinking {self.thinking}" if self.thinking else ""
        if self.reasoning_effort:
            mode += f", effort {self.reasoning_effort}"
        return f"{self.model}{mode} @ {urlparse(self.base_url).hostname}"

    def _connect(self):
        if self._client is None:
            key = api_key_from_env(self.api_key_env)
            from openai import OpenAI

            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def complete(self, messages: Messages, temperature: float = 0.7) -> LLMReply:
        options: dict = dict(self.fixed_params or {"temperature": temperature})
        extra: dict = {}
        if self.thinking is not None:
            extra["thinking"] = {"type": self.thinking}
        if self.reasoning_effort is not None:
            extra["chat_template_kwargs"] = {"reasoning_effort": self.reasoning_effort}
        if extra:
            options["extra_body"] = extra
        if self.thinking != "enabled" and self.reasoning_effort is None:
            # No JSON mode for models that reason first: DeepSeek advises against combining the
            # two, and K2 Horizon hosts may not offer it. The prompts ask for JSON either way, and
            # only `content` is read: the reasoning comes in a separate field.
            options["response_format"] = {"type": "json_object"}
        request = {"model": self.model, "messages": messages, **options}
        path = _cache_path(self.cache_dir, request)
        if (reply := _read_cache(path)) is not None:
            self.budget.charge(reply)
            return reply

        self.budget.check()
        response = self._connect().chat.completions.create(
            model=self.model, messages=messages, **options
        )
        usage = response.usage
        reply = LLMReply(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        _write_cache(path, request, reply)
        self.budget.charge(reply)
        return reply


# Claude models without the API's server-side refusal fallback; their requests leave it out.
NO_SERVER_FALLBACK = frozenset({"claude-haiku-5-5"})


@dataclass
class AnthropicLLM:
    """Claude through the official Anthropic SDK.

    Current Claude models always think and choose their own sampling, so the caller's
    temperature is not sent; `effort` sets how hard the model thinks. Each role sends the same
    system prompt every round, so it is marked for prompt caching. Replies are cached on disk
    and charged to the budget like the other clients. A refusal comes back as an empty reply,
    which the research loop handles like any other reply it cannot use.
    """

    model: str = "claude-haiku-5-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    effort: str = "high"
    max_tokens: int = 16000
    cache_dir: Path = Path(".cache/llm")
    budget: Budget = field(default_factory=Budget)
    timeout: float = 600.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self._client = None

    @property
    def name(self) -> str:
        return f"{self.model}, effort {self.effort} @ Anthropic API"

    def _connect(self):
        if self._client is None:
            key = api_key_from_env(self.api_key_env)
            import anthropic

            self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout)
        return self._client

    def complete(self, messages: Messages, temperature: float = 0.7) -> LLMReply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        request = {
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": turns,
        }
        path = _cache_path(self.cache_dir, request)
        if (reply := _read_cache(path)) is not None:
            self.budget.charge(reply)
            return reply

        self.budget.check()
        options: dict = {}
        if system:
            options["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if self.model not in NO_SERVER_FALLBACK:
            # On a policy refusal the API reruns the request on an earlier Claude model.
            options["betas"] = ["server-side-fallback-2026-07-01"]
            options["fallbacks"] = "default"
        # Streamed so that a long reply cannot hit a request timeout.
        with self._connect().beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=turns,
            output_config={"effort": self.effort},
            **options,
        ) as stream:
            response = stream.get_final_message()
        text = ""
        if response.stop_reason != "refusal":
            text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        reply = LLMReply(
            text=text,
            prompt_tokens=(usage.input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0),
            completion_tokens=usage.output_tokens or 0,
        )
        _write_cache(path, request, reply)
        self.budget.charge(reply)
        return reply


def make_llm(
    provider: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    thinking: str | None = None,
    effort: str | None = None,
    cache_dir: Path = Path(".cache/llm"),
    budget: Budget | None = None,
) -> LLM:
    """Build the client for `provider`, overriding any of its defaults.

    `thinking` None keeps the provider's default; "omit" leaves the field out of requests.
    `effort` None means "high" for providers that take an effort.
    """
    preset = get_provider(provider)
    if effort is not None and effort not in preset.efforts:
        accepted = f"choose from {', '.join(preset.efforts)}" if preset.efforts else "it takes none"
        raise ValueError(f"{provider} does not take effort {effort!r}; {accepted}")
    effort = effort or ("high" if preset.efforts else None)
    if thinking is not None and preset.thinking is None:
        raise ValueError(f"{provider} has no thinking switch; set the effort instead")
    key_env = api_key_env or preset.api_key_env
    budget = budget or Budget()
    model = model or preset.default_model()
    if preset.api == "anthropic":
        if base_url is not None:
            raise ValueError("a base URL applies to OpenAI-compatible providers only")
        return AnthropicLLM(
            model=model, api_key_env=key_env, effort=effort, cache_dir=cache_dir, budget=budget
        )
    base_url = base_url or preset.default_base_url()
    if not base_url:
        hint = f"set {preset.base_url_env} or " if preset.base_url_env else ""
        raise RuntimeError(f"{provider} has no API address; {hint}pass --base-url")
    if thinking is None:
        thinking = preset.thinking
    elif thinking == "omit":
        thinking = None
    return OpenAICompatibleLLM(
        model=model, base_url=base_url, api_key_env=key_env, thinking=thinking,
        reasoning_effort=effort, fixed_params=preset.fixed_params, cache_dir=cache_dir,
        budget=budget,
    )  # fmt: skip


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
    """Parse a model reply as a JSON object.

    Tolerates a Markdown code fence around it, and the reasoning that a server without a
    reasoning parser leaves in front of the answer.
    """
    text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value

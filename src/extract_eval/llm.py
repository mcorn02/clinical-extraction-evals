"""One thin wrapper around LiteLLM so every provider looks the same.

Model strings follow LiteLLM's format, e.g. "gpt-5-mini", "anthropic/claude-haiku-5-5",
"gemini/gemini-2.5-flash". API keys come from environment variables (see .env.example).
"""

import time
from dataclasses import dataclass

import litellm

# Some models (e.g. the GPT-5 family) reject temperature=0. Drop unsupported params instead of failing.
litellm.drop_params = True


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def call_llm(model: str, messages: list[dict], temperature: float = 0.0, **kwargs) -> LLMResponse:
    start = time.perf_counter()
    resp = litellm.completion(model=model, messages=messages, temperature=temperature, **kwargs)
    latency = time.perf_counter() - start

    usage = getattr(resp, "usage", None)
    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception:
        cost = None  # LiteLLM doesn't have pricing for every model; record it as unknown

    return LLMResponse(
        text=resp.choices[0].message.content or "",
        latency_s=latency,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        cost_usd=cost,
    )

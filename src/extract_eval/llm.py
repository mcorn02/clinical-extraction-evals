"""One thin wrapper around LiteLLM so every provider looks the same.

Model strings follow LiteLLM's format, e.g. "gpt-5-mini", "anthropic/claude-haiku-5-5",
"gemini/gemini-2.5-flash". API keys come from environment variables (see .env.example).

`call_llm` times every call, including ones that raise: on failure the exception gets a
`latency_s` attribute before it is re-raised. `classify_error` / `describe_error` turn an
exception into a stable error type ("rate_limit", "auth", ...) and a log-safe message.
"""

import re
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
    try:
        resp = litellm.completion(model=model, messages=messages, temperature=temperature, **kwargs)
    except Exception as e:
        # Record how long the failed attempt took so slow failures (timeouts, 5xx) are visible
        try:
            e.latency_s = time.perf_counter() - start
        except Exception:
            pass
        raise
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


# --------------------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------------------

# Error types for API failures. "invalid_output" (parse/validation failure) is set by the extractor.
API_ERROR_TYPES = (
    "rate_limit", "auth", "timeout", "connection", "bad_request", "context_window",
    "server_error", "not_found", "api_error_other",
)

# Order matters: subclasses before their parents (ContextWindowExceeded is a BadRequest,
# Timeout is an APIConnectionError, etc.). Classes missing from an older litellm are skipped.
_LITELLM_CLASSES = [
    ("ContextWindowExceededError", "context_window"),
    ("ContentPolicyViolationError", "bad_request"),
    ("BadRequestError", "bad_request"),
    ("UnprocessableEntityError", "bad_request"),
    ("AuthenticationError", "auth"),
    ("PermissionDeniedError", "auth"),
    ("RateLimitError", "rate_limit"),
    ("Timeout", "timeout"),
    ("APIConnectionError", "connection"),
    ("NotFoundError", "not_found"),
    ("ServiceUnavailableError", "server_error"),
    ("InternalServerError", "server_error"),
    ("BadGatewayError", "server_error"),
]
_CLASS_MAP = [(getattr(litellm, n), t) for n, t in _LITELLM_CLASSES if isinstance(getattr(litellm, n, None), type)]

_STATUS_MAP = {400: "bad_request", 401: "auth", 403: "auth", 404: "not_found", 408: "timeout", 413: "context_window", 422: "bad_request", 429: "rate_limit"}


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def classify_error(exc: BaseException) -> str:
    """Map an exception from call_llm to one of API_ERROR_TYPES."""
    for cls, err_type in _CLASS_MAP:
        if isinstance(exc, cls):
            return err_type
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"
    code = _status_code(exc)  # e.g. a raw provider SDK error or litellm.APIError
    if code is not None:
        if code in _STATUS_MAP:
            return _STATUS_MAP[code]
        if code >= 500:
            return "server_error"
    return "api_error_other"


# Never let something that looks like a credential reach a log file.
_SECRET_PATTERNS = [
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|x-api-key|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"), r"\1[REDACTED]"),
    (re.compile(r"\b(?:sk|pk|rk|xai|gsk)[-_][A-Za-z0-9_\-]{8,}|\bAIza[A-Za-z0-9_\-]{20,}"), "[REDACTED]"),
]


def redact(text: str) -> str:
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


def describe_error(exc: BaseException, err_type: str | None = None, limit: int = 500) -> str:
    """'<type>: <ExceptionClass> (status N): <message>', redacted and truncated."""
    err_type = err_type or classify_error(exc)
    code = _status_code(exc)
    status = f" (status {code})" if code is not None else ""
    return redact(f"{err_type}: {type(exc).__name__}{status}: {exc}")[:limit]

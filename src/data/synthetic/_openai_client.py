"""SmartHandover - Multi-provider client pool with round-robin + backoff.

Supports two pools:

  * **text pool**  - used by ``generate_text`` and ``filter_text``.
                     Backed by IAEDU (custom multipart streaming protocol)
                     when ``configs/iaedu_accounts.json`` exists, OR by the
                     standard OpenAI SDK when ``OPENAI_API_KEYS`` is set.

  * **tts pool**   - used by ``generate_audio``.
                     Always uses the standard OpenAI SDK (TTS is not
                     supported by the IAEDU classroom proxy).

Behaviour
---------
* Round-robin acquisition across all accounts (thread-safe).
* On rate-limit / connection errors the offending account is put into a
  short cooldown so the next call goes to a different account.
* If every account in a pool is cooling, the caller blocks until the
  earliest cooldown expires.
* Retries are exponential with full jitter, honouring ``Retry-After``.

The IAEDU adapter (``IAEduClient``) exposes the same ``client.chat.
completions.create(...)`` surface as the OpenAI SDK so the rest of the
pipeline (which was written against OpenAI) is provider-agnostic.

Configuration
-------------
Preferred IAEDU setup (free, requires no paid OpenAI key for text):

    # configs/iaedu_accounts.json
    {
      "default_endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/<agent>/stream",
      "accounts": [
        {"name": "Conta 1", "api_key": "sk-usr-...", "channel_id": "cm..."},
        ...
      ]
    }

Standard OpenAI fallback (if no IAEDU file is present):

    # .env
    OPENAI_API_KEYS=sk-proj-...,sk-proj-...      # comma-separated, optional rotation
    OPENAI_BASE_URL=https://api.openai.com/v1    # optional override

For TTS (``generate_audio.py``), always:

    OPENAI_TTS_API_KEY=sk-...
    OPENAI_TTS_BASE_URL=https://api.openai.com/v1   # optional
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar

from src.data.synthetic import config as cfg

_T = TypeVar("_T")

# Default location of the IAEDU account pool file.
IAEDU_ACCOUNTS_FILE = os.path.join("configs", "iaedu_accounts.json")

# UUID-only "ack" lines IAEDU streams - filter them out of the response.
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# OpenAI module accessor (lazy; the offline tests don't need it)
# ---------------------------------------------------------------------------


def _import_openai_module():
    try:
        import openai  # type: ignore
        return openai
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' Python SDK is required. Install with "
            "`pip install openai>=1.50.0`."
        ) from e


# ---------------------------------------------------------------------------
# .env loader (no external dependency)
# ---------------------------------------------------------------------------


def load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a key=value .env file. Silent if missing."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)


# ===========================================================================
# IAEduClient - drop-in replacement for openai.OpenAI for the chat endpoint
# ===========================================================================


class IAEduError(RuntimeError):
    """Base class for IAEDU-specific errors that we want the pool to retry."""


class IAEduRateLimit(IAEduError):
    """Raised when an IAEDU account hits its quota."""


class IAEduConnectionError(IAEduError):
    """Raised on transport-level failures (DNS, timeout, 5xx, etc)."""


class _ChatMessage:
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _Choice:
    __slots__ = ("message", "index", "finish_reason")

    def __init__(self, content: str):
        self.message = _ChatMessage(content)
        self.index = 0
        self.finish_reason = "stop"


class _ChatCompletion:
    """Minimal stand-in for openai.types.chat.ChatCompletion."""

    __slots__ = ("choices",)

    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _CompletionsAdapter:
    def __init__(self, client: "IAEduClient"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[dict] = None,
        timeout: Optional[float] = None,
        **_kwargs: Any,
    ) -> _ChatCompletion:
        # IAEDU only accepts ONE message per call - merge system+user.
        prompt = self._merge_messages(messages, response_format)
        text = self._client._post_iaedu(prompt, timeout=timeout)
        return _ChatCompletion(text)

    @staticmethod
    def _merge_messages(messages: List[dict],
                        response_format: Optional[dict]) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                parts.append(f"[USER]\n{content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}")
            else:
                parts.append(str(content))
        # Hint for the json_object response_format used by the LLM judge:
        if response_format and response_format.get("type") == "json_object":
            parts.append(
                "[OUTPUT FORMAT]\nReturn STRICT JSON only. "
                "No prose, no code fences."
            )
        return "\n\n".join(parts)


class _ChatAdapter:
    def __init__(self, client: "IAEduClient"):
        self.completions = _CompletionsAdapter(client)


@dataclass(frozen=True)
class IAEduAccount:
    name: str
    api_key: str
    channel_id: str
    endpoint: str


class IAEduClient:
    """OpenAI-compatible adapter around a single IAEDU account.

    Mimics ``openai.OpenAI`` enough that ``client.chat.completions.create(
    model=..., messages=[...], ...)`` works transparently.
    """

    def __init__(self, account: IAEduAccount, timeout: float = 60.0):
        self.account = account
        self.timeout = timeout
        self.chat = _ChatAdapter(self)

    # ------------------------------------------------------------------
    # IAEDU streaming protocol
    # ------------------------------------------------------------------

    def _post_iaedu(self, prompt: str,
                    timeout: Optional[float] = None) -> str:
        import requests
        from requests.exceptions import (
            ConnectionError as ReqConn,
            Timeout as ReqTimeout,
            RequestException,
        )

        headers = {"x-api-key": self.account.api_key}
        thread_id = str(uuid.uuid4())
        user_info = json.dumps({"name": "smarthandover", "role": "student"})
        # multipart/form-data via files=, mirroring the AP project.
        payload = {
            "channel_id": (None, self.account.channel_id),
            "message":    (None, prompt),
            "thread_id":  (None, thread_id),
            "user_info":  (None, user_info),
        }

        try:
            resp = requests.post(
                self.account.endpoint,
                headers=headers,
                files=payload,
                stream=True,
                timeout=timeout if timeout is not None else self.timeout,
            )
        except (ReqConn, ReqTimeout) as e:
            raise IAEduConnectionError(str(e)) from e
        except RequestException as e:
            raise IAEduConnectionError(str(e)) from e

        if resp.status_code == 429:
            raise IAEduRateLimit(f"HTTP 429 from IAEDU ({self.account.name})")
        if resp.status_code >= 500:
            raise IAEduConnectionError(
                f"HTTP {resp.status_code} from IAEDU ({self.account.name})"
            )
        if resp.status_code != 200:
            raise IAEduError(
                f"HTTP {resp.status_code} from IAEDU "
                f"({self.account.name}): {resp.text[:200]}"
            )

        # Parse streaming NDJSON response.
        chunks: List[str] = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue

            t = msg.get("type", "")
            if t in ("start", "end", "close"):
                continue
            if t == "error":
                err = str(msg.get("content", "unknown error"))
                if "429" in err or "rate" in err.lower():
                    raise IAEduRateLimit(err)
                raise IAEduError(err)

            content = msg.get("content")
            if isinstance(content, str):
                if _UUID_PATTERN.search(content):
                    continue
                if content.strip().lower() in ("", "processing"):
                    continue
                chunks.append(content)
            elif isinstance(content, dict):
                txt = content.get("text")
                if isinstance(txt, str) and txt.strip():
                    chunks.append(txt)

        text = "".join(chunks).strip()
        if not text:
            raise IAEduError(f"Empty response from IAEDU ({self.account.name})")
        return text


# ===========================================================================
# ClientPool - shared across IAEDU and OpenAI back-ends
# ===========================================================================


@dataclass
class _ClientEntry:
    name: str
    client: Any           # IAEduClient or openai.OpenAI
    cooling_until: float = 0.0
    n_ok: int = 0
    n_fail: int = 0
    n_rate_limited: int = 0


class ClientPool:
    """Thread-safe round-robin pool with per-client cooldown."""

    def __init__(self, name: str, entries: List[_ClientEntry]):
        if not entries:
            raise RuntimeError(
                f"ClientPool '{name}' has no clients - cannot proceed."
            )
        self.name = name
        self._entries = entries
        self._counter = 0
        self._lock = threading.Lock()

    def acquire(self) -> _ClientEntry:
        while True:
            now = time.time()
            with self._lock:
                n = len(self._entries)
                for offset in range(n):
                    idx = (self._counter + offset) % n
                    if self._entries[idx].cooling_until <= now:
                        self._counter = (idx + 1) % n
                        return self._entries[idx]
                soonest = min(e.cooling_until for e in self._entries)
                wait_s = max(0.0, soonest - now)
            time.sleep(min(wait_s, 5.0))

    def mark_ok(self, entry: _ClientEntry) -> None:
        with self._lock:
            entry.n_ok += 1

    def mark_rate_limited(self, entry: _ClientEntry,
                          cooldown_s: float = 10.0) -> None:
        with self._lock:
            entry.n_rate_limited += 1
            entry.n_fail += 1
            entry.cooling_until = time.time() + cooldown_s

    def mark_failed(self, entry: _ClientEntry,
                    cooldown_s: float = 2.0) -> None:
        with self._lock:
            entry.n_fail += 1
            entry.cooling_until = time.time() + cooldown_s

    def summary(self) -> str:
        with self._lock:
            lines = [f"  {self.name} pool ({len(self._entries)} client(s)):"]
            for e in self._entries:
                lines.append(
                    f"    {e.name:<15s}  ok={e.n_ok:>5d}  "
                    f"fail={e.n_fail:>3d}  rl={e.n_rate_limited:>3d}"
                )
            return "\n".join(lines)


# ===========================================================================
# Pool construction (cached)
# ===========================================================================


_pool_lock = threading.Lock()
_pools: dict = {}


def _load_iaedu_accounts(path: str = IAEDU_ACCOUNTS_FILE) -> List[IAEduAccount]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    default_endpoint = data.get("default_endpoint", "")
    out: List[IAEduAccount] = []
    for i, acc in enumerate(data.get("accounts", []), start=1):
        api_key = (acc.get("api_key") or "").strip()
        channel = (acc.get("channel_id") or "").strip()
        endpoint = (acc.get("endpoint") or default_endpoint or "").strip()
        if not api_key or api_key.startswith("sk-usr-PASTE"):
            continue   # placeholder - skip
        if not channel or channel.startswith("PASTE"):
            continue
        if not endpoint:
            continue
        out.append(IAEduAccount(
            name=acc.get("name", f"Conta {i}"),
            api_key=api_key,
            channel_id=channel,
            endpoint=endpoint,
        ))
    return out


def _build_text_pool() -> ClientPool:
    """Prefer IAEDU when configs/iaedu_accounts.json has live entries.
    Fall back to OpenAI standard via OPENAI_API_KEYS otherwise."""
    accounts = _load_iaedu_accounts()
    if accounts:
        entries = [
            _ClientEntry(name=acc.name, client=IAEduClient(acc))
            for acc in accounts
        ]
        return ClientPool("text(IAEDU)", entries)

    openai = _import_openai_module()
    keys_csv = os.environ.get("OPENAI_API_KEYS") or os.environ.get(
        "OPENAI_API_KEY", ""
    )
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    if not keys:
        raise RuntimeError(
            "No text-generation backend configured.\n"
            f"  - Either fill {IAEDU_ACCOUNTS_FILE} with IAEDU accounts\n"
            "  - Or set OPENAI_API_KEYS=sk-... in .env"
        )
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    entries = []
    for i, key in enumerate(keys, start=1):
        kwargs = {"api_key": key, "timeout": cfg.REQUEST_TIMEOUT_SEC}
        if base_url:
            kwargs["base_url"] = base_url
        entries.append(_ClientEntry(
            name=f"key#{i}", client=openai.OpenAI(**kwargs),
        ))
    return ClientPool("text(OpenAI)", entries)


def _build_judge_pool() -> ClientPool:
    """Build the pool used by the LLM judge (filter_text.py).

    Provider is selected via ``cfg.JUDGE_PROVIDER``:
      * "ollama"  - local Ollama via OpenAI-compatible /v1 (default)
      * "iaedu"   - reuse the IAEDU text pool (self-judge, biased)
      * "openai"  - standard OpenAI API with OPENAI_TTS_API_KEY/OPENAI_API_KEY
    """
    provider = cfg.JUDGE_PROVIDER

    if provider == "iaedu":
        # Same physical pool as text generation - just delegate.
        return _build_text_pool()

    openai = _import_openai_module()

    if provider == "ollama":
        # Ollama needs an api_key for the OpenAI client even though it does
        # not validate it - any non-empty string works.
        client = openai.OpenAI(
            api_key="ollama",
            base_url=cfg.OLLAMA_BASE_URL,
            timeout=cfg.REQUEST_TIMEOUT_SEC,
        )
        return ClientPool(
            "judge(Ollama)",
            [_ClientEntry(name=cfg.JUDGE_MODEL, client=client)],
        )

    if provider == "openai":
        key = (os.environ.get("OPENAI_TTS_API_KEY")
               or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            raise RuntimeError(
                "JUDGE_PROVIDER=openai but no OPENAI_TTS_API_KEY or "
                "OPENAI_API_KEY in env."
            )
        base_url = os.environ.get("OPENAI_TTS_BASE_URL") or None
        kwargs = {"api_key": key, "timeout": cfg.REQUEST_TIMEOUT_SEC}
        if base_url:
            kwargs["base_url"] = base_url
        return ClientPool(
            "judge(OpenAI)",
            [_ClientEntry(name="openai", client=openai.OpenAI(**kwargs))],
        )

    raise RuntimeError(
        f"Unknown JUDGE_PROVIDER='{provider}'. Use ollama / iaedu / openai."
    )


def _build_tts_pool() -> ClientPool:
    openai = _import_openai_module()
    tts_key = os.environ.get("OPENAI_TTS_API_KEY")
    if not tts_key:
        keys_csv = os.environ.get("OPENAI_API_KEYS") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        if not keys:
            raise RuntimeError(
                "No TTS key configured. Set OPENAI_TTS_API_KEY (preferred) "
                "or OPENAI_API_KEY in .env."
            )
        tts_key = keys[0]
    base_url = os.environ.get("OPENAI_TTS_BASE_URL") or None
    kwargs = {"api_key": tts_key, "timeout": cfg.REQUEST_TIMEOUT_SEC}
    if base_url:
        kwargs["base_url"] = base_url
    return ClientPool(
        "tts",
        [_ClientEntry(name="tts", client=openai.OpenAI(**kwargs))],
    )


_BUILDERS = {
    "text":  _build_text_pool,
    "judge": _build_judge_pool,
    "tts":   _build_tts_pool,
}


def get_pool(purpose: str) -> ClientPool:
    if purpose not in _BUILDERS:
        raise ValueError(f"Unknown pool purpose: {purpose}")
    with _pool_lock:
        pool = _pools.get(purpose)
        if pool is None:
            pool = _BUILDERS[purpose]()
            _pools[purpose] = pool
        return pool


def describe_pools() -> str:
    out = []
    for purpose in ("text", "judge", "tts"):
        try:
            pool = get_pool(purpose)
            out.append(pool.summary())
        except RuntimeError as e:
            out.append(f"  {purpose} pool: NOT CONFIGURED ({e})")
    return "\n".join(out)


# ===========================================================================
# Retry / backoff decorator
# ===========================================================================


def with_pool_backoff(
    pool: ClientPool,
    fn: Callable[[Any], _T],
    *,
    max_retries: int = cfg.MAX_RETRIES,
    initial_backoff: float = cfg.INITIAL_BACKOFF_SEC,
    max_backoff: float = cfg.MAX_BACKOFF_SEC,
    rl_cooldown_s: float = 10.0,
) -> _T:
    """Call ``fn(client)`` against the pool with retry + per-account cooldown.

    Apanha tanto exceções IAEDU como OpenAI - o pool pode misturar ambos.
    """
    openai = _import_openai_module()
    openai_retryable = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
        ConnectionError,
        TimeoutError,
    )

    backoff = initial_backoff
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        entry = pool.acquire()
        try:
            result = fn(entry.client)
            pool.mark_ok(entry)
            return result
        # IAEDU-specific
        except IAEduRateLimit as e:
            pool.mark_rate_limited(entry, cooldown_s=rl_cooldown_s)
            last_err = e
        except IAEduConnectionError as e:
            pool.mark_failed(entry, cooldown_s=2.0)
            last_err = e
        except IAEduError as e:
            pool.mark_failed(entry, cooldown_s=1.0)
            last_err = e
        # OpenAI / generic
        except openai.RateLimitError as e:
            wait = rl_cooldown_s
            ra = _retry_after(e)
            if ra is not None:
                wait = max(wait, ra)
            pool.mark_rate_limited(entry, cooldown_s=wait)
            last_err = e
        except openai_retryable as e:
            pool.mark_failed(entry, cooldown_s=2.0)
            last_err = e
        except openai.AuthenticationError:
            raise
        except openai.PermissionDeniedError:
            raise

        if attempt == max_retries:
            break
        sleep_s = min(random.uniform(0, backoff), max_backoff)
        time.sleep(sleep_s)
        backoff = min(backoff * 2, max_backoff)

    assert last_err is not None
    raise last_err


def _retry_after(err: Any) -> Optional[float]:
    ra = getattr(err, "retry_after", None)
    if ra is not None:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    resp = getattr(err, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {}) or {}
        for h in ("retry-after", "Retry-After"):
            if h in headers:
                try:
                    return float(headers[h])
                except ValueError:
                    pass
    return None


# ===========================================================================
# JSONL helpers (used by all generators)
# ===========================================================================


_file_locks_lock = threading.Lock()
_file_locks: dict = {}


class _file_lock:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def __enter__(self):
        with _file_locks_lock:
            lk = _file_locks.get(self.path)
            if lk is None:
                lk = threading.Lock()
                _file_locks[self.path] = lk
        self._lk = lk
        self._lk.acquire()
        return self

    def __exit__(self, *args):
        self._lk.release()


def append_jsonl(path: str, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _file_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def loaded_ids_from_jsonl(path: str, key: str = "id") -> set:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in rec:
                seen.add(rec[key])
    return seen


# ===========================================================================
# Backward-compat shims (legacy callers)
# ===========================================================================


def get_client():
    return get_pool("text").acquire().client


def with_backoff(fn: Callable[..., _T], *args, **kwargs) -> _T:
    pool = get_pool("text")
    return with_pool_backoff(pool, lambda _client: fn(*args, **kwargs))

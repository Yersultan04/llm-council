"""
LLM Council — 6 моделей обсуждают вопрос, Claude подводит итог.

Совет:
  1. Llama-70B      — Groq (бесплатно)
  2. Gemini-Flash   — Google AI Studio (бесплатно)
  3. QwQ-32B        — Groq (бесплатно)
  4. DeepSeek       — OpenRouter free tier (бесплатно)
  5. Ollama-Local   — локально (бесплатно)
  6. Claude-Haiku   — Anthropic API (платно, дёшево)

Процесс:
  Фаза 1 — все N моделей отвечают параллельно
  Фаза 2 — Claude-председатель синтезирует итог
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Загружаем из директории файла, потом из корня воркспейса
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# --- Синглтон-клиенты ---

_openai_clients: dict[str, AsyncOpenAI] = {}
_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    cache_key = f"{base_url}:{api_key[:8]}"
    if cache_key not in _openai_clients:
        _openai_clients[cache_key] = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "placeholder",
            timeout=30.0,
        )
    return _openai_clients[cache_key]


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


# --- Датаклассы ---

@dataclass
class Member:
    name: str
    model: str
    base_url: str | None
    key_env: str | None
    use_anthropic: bool = False
    free: bool = True


class CouncilResult(TypedDict):
    question: str
    opinions: dict[str, str]
    synthesis: str
    available: int
    total: int


# --- Состав совета ---

COUNCIL: list[Member] = [
    Member(
        name="Llama-70B",
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        free=True,
    ),
    Member(
        name="Gemini-Flash",
        model="gemini-2.0-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY",
        free=True,
    ),
    Member(
        name="QwQ-32B",
        model="qwen-qwq-32b",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        free=True,
    ),
    Member(
        name="DeepSeek",
        model="deepseek/deepseek-chat:free",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        free=True,
    ),
    Member(
        name="Ollama-Local",
        model="llama3.3",
        base_url="http://localhost:11434/v1",
        key_env=None,
        free=True,
    ),
    Member(
        name="Claude-Haiku",
        model="claude-haiku-4-5",
        base_url=None,
        key_env="ANTHROPIC_API_KEY",
        use_anthropic=True,
        free=False,
    ),
]

FREE_MEMBERS = [m for m in COUNCIL if m.free]
COUNCIL_BY_NAME: dict[str, Member] = {m.name: m for m in COUNCIL}


# --- Запрос к моделям ---

async def _ask_openai(member: Member, question: str) -> str:
    api_key = os.environ.get(member.key_env, "") if member.key_env else ""
    client = _get_openai_client(member.base_url or "", api_key)
    resp = await client.chat.completions.create(
        model=member.model,
        messages=[{"role": "user", "content": question}],
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


async def _ask_anthropic(member: Member, question: str) -> str:
    client = _get_anthropic_client()
    resp = await asyncio.wait_for(
        client.messages.create(
            model=member.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": question}],
        ),
        timeout=30.0,
    )
    return resp.content[0].text


async def ask_member(member: Member, question: str) -> tuple[str, str]:
    try:
        if member.use_anthropic:
            text = await _ask_anthropic(member, question)
        else:
            text = await _ask_openai(member, question)
        logger.debug("%s ответил (%d символов)", member.name, len(text))
        return member.name, text
    except Exception as e:
        logger.warning("%s недоступен: %s", member.name, e)
        return member.name, f"[НЕДОСТУПЕН: {type(e).__name__}: {e}]"


# --- Синтез председателем ---

async def chairman_synthesize(
    question: str,
    opinions: dict[str, str],
    verbose: bool = True,
) -> str:
    valid = {k: v for k, v in opinions.items() if not v.startswith("[НЕДОСТУПЕН")}

    if not valid:
        return "Все члены совета недоступны."

    opinions_block = "\n\n".join(
        f"=== {name} ===\n{text}" for name, text in valid.items()
    )

    prompt = f"""Ты — председатель совета ИИ-моделей. Синтезируй ответы участников в единый итог.

ВОПРОС: {question}

ОТВЕТЫ СОВЕТА ({len(valid)} участников):
{opinions_block}

Предоставь:
1. **Консенсус** — в чём модели согласны
2. **Ключевые расхождения** — важные разногласия или уникальные точки зрения
3. **Итоговый ответ** — твой синтезированный, авторитетный ответ

Будь конкретным и лаконичным. Отвечай на том же языке, что и вопрос."""

    client = _get_anthropic_client()
    result = ""

    async with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system="Ты мудрый председатель совета. Синтезируй разные точки зрения ИИ в чёткие actionable ответы.",
        messages=[{"role": "user", "content": prompt}],
        cache_control={"type": "ephemeral"},
    ) as stream:
        async for chunk in stream.text_stream:
            if verbose:
                print(chunk, end="", flush=True)
            result += chunk

    if verbose:
        print()
    return result


# --- Основные функции ---

async def run_council(
    question: str,
    verbose: bool = True,
    members: list[Member] | None = None,
) -> CouncilResult:
    council = members or COUNCIL
    separator = "=" * 60

    if verbose:
        print(f"\n{separator}")
        print(f"  LLM COUNCIL — {len(council)} моделей")
        print(f"{separator}")
        print(f"Вопрос: {question}\n")
        print("📋 Фаза 1: Собираем мнения (параллельно)...\n")

    tasks = [ask_member(m, question) for m in council]
    results = await asyncio.gather(*tasks)
    opinions: dict[str, str] = {}

    for name, text in results:
        opinions[name] = text
        if verbose:
            is_error = text.startswith("[НЕДОСТУПЕН")
            icon = "❌" if is_error else "✅"
            preview = text[:400] + "..." if len(text) > 400 else text
            print(f"{icon}  [{name}]\n{preview}\n")

    available = sum(1 for t in opinions.values() if not t.startswith("[НЕДОСТУПЕН"))

    if verbose:
        print(f"{separator}")
        print(f"🎯 Фаза 2: Председатель синтезирует ({available}/{len(council)})...\n")

    synthesis = await chairman_synthesize(question, opinions, verbose=verbose)

    return CouncilResult(
        question=question,
        opinions=opinions,
        synthesis=synthesis,
        available=available,
        total=len(council),
    )


async def run_quick(question: str, verbose: bool = True) -> CouncilResult:
    """Только бесплатные модели, без синтеза."""
    council = FREE_MEMBERS
    separator = "=" * 60

    if verbose:
        print(f"\n{separator}")
        print(f"  QUICK COUNCIL — {len(council)} бесплатных моделей")
        print(f"{separator}\n")

    tasks = [ask_member(m, question) for m in council]
    results = await asyncio.gather(*tasks)
    opinions: dict[str, str] = {}

    for name, text in results:
        opinions[name] = text
        if verbose:
            is_error = text.startswith("[НЕДОСТУПЕН")
            icon = "❌" if is_error else "✅"
            preview = text[:400] + "..." if len(text) > 400 else text
            print(f"{icon}  [{name}]\n{preview}\n")

    available = sum(1 for t in opinions.values() if not t.startswith("[НЕДОСТУПЕН"))

    return CouncilResult(
        question=question,
        opinions=opinions,
        synthesis="",
        available=available,
        total=len(council),
    )


def get_models_status() -> list[dict]:
    """Возвращает статус всех моделей совета."""
    status = []
    for m in COUNCIL:
        if m.key_env is None:
            key_ok = True
            key_note = "локально (ключ не нужен)"
        else:
            key_val = os.environ.get(m.key_env, "")
            key_ok = bool(key_val)
            key_note = "ключ задан" if key_ok else f"{m.key_env} не задан"

        status.append({
            "name": m.name,
            "model": m.model,
            "free": m.free,
            "key_ok": key_ok,
            "key_note": key_note,
        })
    return status

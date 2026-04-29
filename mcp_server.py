import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from council import (  # noqa: E402
    COUNCIL_BY_NAME,
    CouncilResult,
    get_models_status,
    run_council,
    run_quick,
    ask_member,
)

mcp = FastMCP("LLM Council")

TIMEOUT_SECONDS = 120


def _format_result(result: CouncilResult, show_synthesis: bool = True) -> str:
    header = f"## LLM Council — {result['available']}/{result['total']} моделей ответили\n"
    lines = [header, f"**Вопрос:** {result['question']}\n"]

    for name, text in result["opinions"].items():
        status = "❌" if text.startswith("[НЕДОСТУПЕН") else "✅"
        lines.append(f"### {status} {name}\n{text}\n")

    if show_synthesis and result["synthesis"]:
        lines.append(f"---\n## Синтез председателя\n{result['synthesis']}")

    return "\n".join(lines)


@mcp.tool()
async def ask_council(question: str, models: list[str] | None = None) -> str:
    """
    Задать вопрос совету из LLM моделей — все ответят параллельно, Claude синтезирует итог.

    Параметры:
      question — вопрос на любом языке
      models   — (опционально) список имён моделей: Llama-70B, Gemini-Flash, QwQ-32B,
                 DeepSeek, Ollama-Local, Claude-Haiku. Если не указан — все 6.
    """
    members = None
    if models:
        members = [COUNCIL_BY_NAME[n] for n in models if n in COUNCIL_BY_NAME]
        unknown = [n for n in models if n not in COUNCIL_BY_NAME]
        if unknown:
            return f"Неизвестные модели: {', '.join(unknown)}. Используй list_models() чтобы увидеть доступные."
        if not members:
            return "Нет доступных моделей по указанным именам."

    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            result = await run_council(question, verbose=False, members=members)
        return _format_result(result, show_synthesis=True)
    except TimeoutError:
        return f"⏱️ Совет превысил лимит времени ({TIMEOUT_SECONDS} сек). Попробуй ask_quick() для быстрого ответа."
    except Exception as e:
        return f"❌ Ошибка совета: {type(e).__name__}: {e}"


@mcp.tool()
async def ask_quick(question: str) -> str:
    """
    Быстрый запрос только к бесплатным моделям (Llama, QwQ, Gemini, DeepSeek, Ollama).
    Без синтеза — только сырые ответы. Быстрее и дешевле ask_council().
    """
    try:
        async with asyncio.timeout(40):
            result = await run_quick(question, verbose=False)
        return _format_result(result, show_synthesis=False)
    except TimeoutError:
        return "⏱️ Превышен лимит времени (40 сек)."
    except Exception as e:
        return f"❌ Ошибка: {type(e).__name__}: {e}"


@mcp.tool()
async def ask_model(question: str, model: str) -> str:
    """
    Задать вопрос одной конкретной модели совета.
    Используй list_models() чтобы увидеть доступные имена.

    Пример: ask_model("Что такое RAG?", "Llama-70B")
    """
    member = COUNCIL_BY_NAME.get(model)
    if not member:
        names = ", ".join(COUNCIL_BY_NAME.keys())
        return f"Модель '{model}' не найдена. Доступные: {names}"

    try:
        async with asyncio.timeout(35):
            name, text = await ask_member(member, question)
        status = "❌" if text.startswith("[НЕДОСТУПЕН") else "✅"
        return f"## {status} {name}\n\n**Вопрос:** {question}\n\n{text}"
    except TimeoutError:
        return f"⏱️ {model} не ответил за 35 сек."
    except Exception as e:
        return f"❌ Ошибка: {type(e).__name__}: {e}"


@mcp.tool()
def list_models() -> str:
    """
    Показать все модели совета: имя, провайдер, статус API ключа, тип (бесплатная/платная).
    """
    statuses = get_models_status()
    lines = ["## LLM Council — Модели\n"]

    for s in statuses:
        key_icon = "✅" if s["key_ok"] else "❌"
        tier = "🆓 бесплатно" if s["free"] else "💳 платно"
        lines.append(
            f"**{s['name']}** (`{s['model']}`) — {key_icon} {s['key_note']} | {tier}"
        )

    chairman_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    chairman_icon = "✅" if chairman_ok else "❌"
    lines.append(f"\n---\n**Председатель:** Claude Haiku — {chairman_icon} ANTHROPIC_API_KEY")
    lines.append("\n> Используй `ask_council()`, `ask_quick()`, или `ask_model()` для запросов.")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()

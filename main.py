#!/usr/bin/env python3
"""CLI для LLM Council."""

import asyncio
import sys

from council import run_council


def main() -> None:
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:]).strip()
    else:
        try:
            question = input("Введите вопрос для совета: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nОтмена.")
            sys.exit(0)

    if not question:
        print("Ошибка: вопрос не может быть пустым.")
        sys.exit(1)

    asyncio.run(run_council(question))


if __name__ == "__main__":
    main()

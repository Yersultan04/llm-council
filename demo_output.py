"""Demo script — simulates council output for GIF recording."""
import time, sys

def w(text="", delay=0.03, end="\n"):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print(end=end, flush=True)

def fast(text):
    print(text, flush=True)

sep = "=" * 62

fast(f"\n{sep}")
fast("  LLM COUNCIL — 6 моделей")
fast(f"{sep}")
fast("Вопрос: PostgreSQL или MongoDB для аналитики?\n")
fast("📋 Фаза 1: Собираем мнения (параллельно)...\n")
time.sleep(2.5)

fast("✅  [Llama-70B]")
w("PostgreSQL — явный победитель для аналитики. Поддержка оконных функций, "
  "материализованных представлений и партиционирования делает его\n"
  "незаменимым для OLAP-запросов. MongoDB хорош для документов, но\n"
  "агрегационный пайплайн медленнее и менее выразителен.\n", delay=0.008)
time.sleep(0.3)

fast("✅  [Gemini-Flash]")
w("Зависит от задачи. Для структурированной аналитики и сложных JOIN —\n"
  "PostgreSQL с расширением TimescaleDB. Для полуструктурированных данных\n"
  "и горизонтального масштабирования — MongoDB Atlas. Hybrid подход\n"
  "тоже популярен у крупных компаний.\n", delay=0.008)
time.sleep(0.3)

fast("✅  [QwQ-32B]")
w("PostgreSQL. Аналитика требует: GROUP BY, WINDOW FUNCTIONS, CTE,\n"
  "EXPLAIN ANALYZE. Всё это нативно в PG. MongoDB агрегации — это\n"
  "костыль поверх документной модели. Для аналитики берите правильный\n"
  "инструмент сразу.\n", delay=0.008)
time.sleep(0.3)

fast("✅  [DeepSeek]")
w("PostgreSQL для транзакционной + аналитической нагрузки (HTAP).\n"
  "Если данные > 1TB и нужен sharding — рассмотрите ClickHouse или\n"
  "Redshift. MongoDB оправдан только если схема нестабильна и\n"
  "JOIN'ы не нужны совсем.\n", delay=0.008)
time.sleep(0.3)

fast("❌  [Ollama-Local]")
fast("[НЕДОСТУПЕН: ConnectError: Ollama не запущен]\n")
time.sleep(0.3)

fast("✅  [Claude-Haiku]")
w("PostgreSQL выигрывает по всем аналитическим метрикам: оконные\n"
  "функции, полнотекстовый поиск, JSONB для гибридных данных.\n"
  "MongoDB стоит выбирать только при доминировании документной\n"
  "модели и минимальных аналитических требованиях.\n", delay=0.008)

fast(f"\n{sep}")
fast("🎯 Фаза 2: Председатель Claude синтезирует (5/6)...\n")
time.sleep(1.5)

synthesis = """## Консенсус
Все доступные модели единогласно рекомендуют **PostgreSQL** для аналитических задач.

## Ключевые расхождения
- QwQ и Llama категоричны: PostgreSQL без вариантов
- Gemini добавляет нюанс: MongoDB допустим при гибкой схеме
- DeepSeek указывает порог: при данных >1TB стоит смотреть на ClickHouse

## Итоговый ответ
**Выбирайте PostgreSQL.** Оконные функции, CTE, партиционирование и
расширяемость (TimescaleDB, pgvector) делают его стандартом для аналитики.
MongoDB оправдан только если у вас документная модель данных и аналитика
вторична. Для серьёзной аналитики (>1TB) — ClickHouse или Redshift."""

for line in synthesis.split("\n"):
    w(line, delay=0.012)
    time.sleep(0.05)

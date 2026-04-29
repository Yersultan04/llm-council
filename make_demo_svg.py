"""Generate demo SVG for README using rich."""
import io
from rich.console import Console

# Write to StringIO to avoid Windows cp1251 encoding issues
buf = io.StringIO()
console = Console(file=buf, record=True, width=90)

sep = "=" * 62

console.print()
console.print(f"[bold cyan]{sep}[/bold cyan]")
console.print("[bold cyan]  LLM COUNCIL — 6 models[/bold cyan]")
console.print(f"[bold cyan]{sep}[/bold cyan]")
console.print("[bold]Question:[/bold] PostgreSQL or MongoDB for analytics?\n")
console.print("[bold yellow]>> Phase 1: Collecting opinions (parallel)...[/bold yellow]\n")

console.print("[bold green]OK  [Llama-70B][/bold green]")
console.print("PostgreSQL wins for analytics. Window functions, materialized views,\n"
              "and partitioning make it ideal for OLAP. MongoDB aggregation pipeline\n"
              "is slower and less expressive for complex queries.\n")

console.print("[bold green]OK  [Gemini-Flash][/bold green]")
console.print("Depends on use case. For structured analytics and complex JOINs —\n"
              "PostgreSQL + TimescaleDB. For flexible schema and horizontal scale\n"
              "— MongoDB Atlas. Hybrid approach is also popular at scale.\n")

console.print("[bold green]OK  [QwQ-32B][/bold green]")
console.print("PostgreSQL. Analytics requires: GROUP BY, WINDOW FUNCTIONS, CTE,\n"
              "EXPLAIN ANALYZE — all native in PG. MongoDB aggregations are a\n"
              "workaround on top of a document model. Use the right tool.\n")

console.print("[bold green]OK  [DeepSeek][/bold green]")
console.print("PostgreSQL for HTAP workloads. If data > 1TB and you need sharding\n"
              "— consider ClickHouse or Redshift. MongoDB only if schema is\n"
              "unstable and JOINs are not needed at all.\n")

console.print("[bold red]FAIL  [Ollama-Local][/bold red]")
console.print("[dim][UNAVAILABLE: ConnectError — Ollama not running][/dim]\n")

console.print("[bold green]OK  [Claude-Haiku][/bold green]")
console.print("PostgreSQL wins on all analytics metrics: window functions,\n"
              "full-text search, JSONB for hybrid data + pgvector for AI use cases.\n"
              "Choose MongoDB only when document model dominates.\n")

console.print(f"\n[bold cyan]{sep}[/bold cyan]")
console.print("[bold yellow]>> Phase 2: Chairman synthesizes (5/6 responded)...[/bold yellow]\n")

console.print("[bold]## Consensus[/bold]")
console.print("All available models unanimously recommend [bold green]PostgreSQL[/bold green]"
              " for analytics.\n")

console.print("[bold]## Key Differences[/bold]")
console.print("- QwQ and Llama: PostgreSQL, no exceptions")
console.print("- Gemini: MongoDB acceptable if schema is truly flexible")
console.print("- DeepSeek: at >1TB consider ClickHouse instead\n")

console.print("[bold]## Final Answer[/bold]")
console.print("[bold green]Choose PostgreSQL.[/bold green] Window functions, CTE, partitioning,")
console.print("and extensions (TimescaleDB, pgvector) make it the analytics standard.")
console.print("MongoDB is justified only when document model dominates and JOINs")
console.print("are not required. For serious analytics (>1TB) — ClickHouse.")
console.print()

console.save_svg("demo.svg", title="LLM Council — ask_council('PostgreSQL or MongoDB?')")
print("Saved: demo.svg")

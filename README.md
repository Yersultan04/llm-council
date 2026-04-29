# LLM Council 🏛️

> Ask 6 AI models the same question in parallel. Claude synthesizes a final answer.

A Model Context Protocol (MCP) server that orchestrates a council of 6 LLMs — free and paid — to give you multi-perspective answers on any question.

## How It Works

```
Your question
      ↓
┌─────────────────────────────────────┐
│  Phase 1 — Parallel (5-10 sec)      │
│                                     │
│  Llama-70B    →  opinion            │
│  Gemini-Flash →  opinion            │
│  QwQ-32B      →  opinion            │
│  DeepSeek     →  opinion            │
│  Ollama-Local →  opinion            │
│  Claude-Haiku →  opinion            │
└─────────────────────────────────────┘
      ↓
Phase 2 — Claude chairman reads all 6 opinions
      ↓
Consensus + Key Differences + Final Answer
```

## Why

Any single model has blind spots and training biases. When you need a reliable answer on something important — architecture decisions, complex bugs, business choices — one model isn't enough. LLM Council is like asking 6 experts instead of one, for less than $0.001 per query.

## MCP Tools

| Tool | Description |
|------|-------------|
| `ask_council(question, models?)` | Full council + synthesis. Optionally filter models |
| `ask_quick(question)` | Free models only, no synthesis — fast |
| `ask_model(question, model)` | Single model query |
| `list_models()` | Status of all models and API keys |

## Cost

| Model | Provider | Cost |
|-------|----------|------|
| Llama-3.3-70B | Groq | Free |
| Gemini-2.0-Flash | Google AI Studio | Free |
| QwQ-32B | Groq | Free |
| DeepSeek-Chat | OpenRouter | Free |
| Llama-3.3 | Ollama (local) | Free |
| Claude-Haiku-4-5 | Anthropic | ~$0.001/call |

**One full council query costs less than $0.001.**

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_key    # required — chairman + Claude member
GROQ_API_KEY=your_key         # free at console.groq.com
GEMINI_API_KEY=your_key       # free at aistudio.google.com
OPENROUTER_API_KEY=your_key   # free at openrouter.ai/keys
```

Ollama is optional — install from [ollama.com](https://ollama.com) and run `ollama pull llama3.3`.

### 3. Add to Claude Code (MCP)

Add to your `~/.claude/mcp.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "python",
      "args": ["/absolute/path/to/llm-council/mcp_server.py"]
    }
  }
}
```

Restart Claude Code. You'll see `ask_council`, `ask_quick`, `ask_model`, `list_models` available as tools.

### 4. Or run as CLI

```bash
python main.py "What is the best database for real-time analytics?"
```

## Usage Examples

**Full council:**
```
ask_council("Should I use PostgreSQL or MongoDB for this project?")
```

**Quick check (free only, no synthesis):**
```
ask_quick("Explain the difference between RAG and fine-tuning")
```

**Single model:**
```
ask_model("Write a Python async generator", "Llama-70B")
```

**Selective council:**
```
ask_council("Best approach for mobile auth?", ["Llama-70B", "Gemini-Flash", "DeepSeek"])
```

## Architecture

```
mcp_server.py       ← FastMCP server, 4 tools, timeouts, error handling
council.py          ← Core logic: async parallel queries + chairman synthesis
main.py             ← CLI entrypoint
```

**Key design decisions:**
- `asyncio.gather()` — all models queried in parallel, not sequentially
- Graceful degradation — unavailable models show `[UNAVAILABLE]`, rest continue
- Singleton clients — `AsyncOpenAI` and `AsyncAnthropic` created once, reused
- `verbose=False` in MCP mode — no stdout pollution on stdio transport
- Prompt caching on synthesis — reduces cost on repeated similar queries

## Requirements

- Python 3.11+
- `anthropic>=0.40.0`
- `openai>=1.50.0`
- `python-dotenv>=1.0.0`
- `fastmcp>=2.0.0`

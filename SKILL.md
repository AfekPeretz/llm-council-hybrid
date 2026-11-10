---
name: llm-council-hybrid
description: Run any decision, plan, or idea through an 8-voice LLM Council — 3 external models (GPT-4o, Gemini 1.5 Pro, Perplexity sonar-pro) plus 5 internal Claude lens agents — with anonymous peer review and a final chairman verdict. Use when the user says "council this", "run the council", "war room this", "pressure-test this", or asks for a multi-model / multi-perspective review of a decision.
---

# LLM Council Hybrid

## What this skill does

Takes one question or decision and runs it through a full council pipeline:

1. **8 parallel voices at T=0** (`asyncio.gather` — waits for all before proceeding):
   - GPT-4o → Executor lens (concrete next steps, no hedging)
   - Gemini 1.5 Pro → Expansionist lens (upside everyone else misses)
   - Perplexity sonar-pro → Outsider lens (fresh eyes + live web search with citations)
   - Internal Claude × 5 → Contrarian, First Principles, Expansionist, Outsider, Executor
2. **Anonymous peer review** — 5 internal Claude reviewers read all successful responses shuffled and labeled A–H (no source names), in parallel. Each answers: strongest response and why, biggest blind spot, what ALL responses missed.
3. **Chairman synthesis** — one final Claude call sees everything de-anonymized with source labels plus all 5 reviews, and prints a structured markdown verdict.

## Triggers

Invoke this skill when the user says any of:
- "council this" / "run the council"
- "war room this"
- "pressure-test this"
- or otherwise asks for a multi-model, multi-perspective verdict on a decision

## How to run it

```bash
.venv/bin/python scripts/council.py "<the user's question, framed as one clear decision>"
```

- Add `-q` / `--quiet` to suppress progress logs and print only the verdict.
- The script reads API keys from a `.env` file next to it (see `.env.template`). `ANTHROPIC_API_KEY` is mandatory; if any of the other three keys is missing or a call fails, that voice degrades gracefully and the chairman is told to treat it as an absent voice.

## Workflow for Claude Code

1. Take the user's decision/question. If it's vague, tighten it into ONE clear question first (the council performs badly on mushy prompts) — confirm the framing with the user if you changed it substantially.
2. `cd` into this skill's directory (the folder containing this SKILL.md) and run the command above with the `.venv` Python — the dependencies are installed only there. If `.venv` is missing, create it first: `python3 -m venv .venv && .venv/bin/pip install httpx anthropic python-dotenv`.
3. Expect ~20–60 seconds: external calls have a 10s timeout with one 8s retry; Perplexity is usually the slowest because it runs a live web search.
4. Present the verdict markdown to the user as-is (it ends with "The One Thing to Do First").
5. If voices failed, mention which ones were absent — the verdict already accounts for it.

## Cost note

One council run ≈ 3 external API calls + 11 internal Claude calls (5 lenses, 5 reviewers, 1 chairman). Peer review is deliberately internal-only — sending responses back to GPT/Gemini/Perplexity would double external cost for no proportional gain.

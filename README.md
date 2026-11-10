# LLM Council Hybrid

Run any decision through a council of 8 AI advisors — 3 different frontier models plus 5 Claude sub-agents, each locked into a distinct thinking lens — then let them anonymously peer-review each other before a chairman delivers one structured verdict.

Works two ways: as a **standalone CLI script**, or as a **Claude Code skill** you trigger with "council this".

```
$ python scripts/council.py "should I charge monthly or yearly for my SaaS?"

## Council Verdict: SaaS pricing cadence
### Where the Council Agrees
### Where the Council Clashes
### Blind Spots the Council Caught
### The Recommendation
### The One Thing to Do First
```

## Why 8 voices, and why these

Asking one model for "pros and cons" gets you mush. This council forces genuine disagreement two ways at once:

**Different models** — GPT-4o, Gemini 1.5 Pro, Perplexity sonar-pro, and Claude have different training, different blind spots, and (in Perplexity's case) live web search. When they disagree on the same prompt, that disagreement is information.

**Different lenses** — every voice is locked into one persona and told not to hedge:

| Voice | Model | Lens |
|---|---|---|
| 1 | GPT-4o | **Executor** — what do I do Monday morning? |
| 2 | Gemini 1.5 Pro | **Expansionist** — the upside everyone else misses |
| 3 | Perplexity sonar-pro | **Outsider** — fresh eyes, grounded in live web search |
| 4–8 | Claude (internal) | Contrarian, First Principles, Expansionist, Outsider, Executor |

The internal Claude agents use the *same* lens prompts as the external models, so cross-model disagreement inside one lens is signal, not prompt noise.

## How it works

```
question
  → 8 voices in parallel (asyncio.gather, temperature 0)
  → shuffle + anonymize responses as A–H
  → 5 Claude peer reviewers in parallel:
      strongest? biggest blind spot? what did ALL miss?
  → chairman sees everything de-anonymized + all reviews
  → structured markdown verdict
```

Peer review is internal-only (Claude reviewers) on purpose — sending responses back out to three external APIs would roughly double the external cost for little extra rigor. The reviewers never see source names, so no brand-based bias.

**Graceful degradation:** every external call gets a 10-second timeout, one retry at 8 seconds, and if it still fails the pipeline continues — the chairman is explicitly told to treat that voice as absent. Perplexity is the usual suspect (it runs a real web search mid-call).

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/<you>/llm-council-hybrid.git
cd llm-council-hybrid
pip install httpx anthropic python-dotenv
cp .env.template .env
```

Fill `.env` with your keys:

```
OPENAI_API_KEY=      # platform.openai.com
GOOGLE_API_KEY=      # aistudio.google.com
PERPLEXITY_API_KEY=  # perplexity.ai/settings/api
ANTHROPIC_API_KEY=   # console.anthropic.com  (required)
```

Only `ANTHROPIC_API_KEY` is mandatory (it powers the 5 internal voices, 5 reviewers, and chairman — 11 calls per run). Missing external keys just mean those voices sit out and the chairman is told.

`.env` is gitignored. Keys never leave your machine — the script talks directly to the four official APIs and nowhere else.

## Usage

### Standalone

```bash
python scripts/council.py "your decision here"
python scripts/council.py -q "your decision here"   # verdict only, no progress logs
```

A run takes roughly 20–60 seconds and costs about 3 external calls + 11 Claude calls.

### As a Claude Code skill

Copy (or clone) the folder into your skills directory:

```bash
# global — available in every project
cp -r llm-council-hybrid ~/.claude/skills/

# or per-project
cp -r llm-council-hybrid <project>/.claude/skills/
```

Then in any Claude Code session:

> council this: should I build feature X before launch or ship without it?

Trigger phrases: **"council this"**, **"run the council"**, **"war room this"**, **"pressure-test this"**. Claude Code reads `SKILL.md`, runs the script, and presents the verdict.

## Tips for good verdicts

- Ask **one clear decision**, not three tangled ones. "Should I do X or Y, given constraint Z" beats "thoughts on my roadmap?"
- Include the constraints that matter (budget, deadline, skills) — the Executor and Contrarian feed on specifics.
- The last section, **The One Thing to Do First**, is the point of the whole exercise. If you don't do it, the council was entertainment.

## Project structure

```
llm-council-hybrid/
├── SKILL.md            # triggers + workflow for Claude Code
├── scripts/
│   └── council.py      # the entire pipeline, single file
├── .env.template       # the 4 keys
├── test_step2.py       # retry/parallelism unit tests (no keys needed)
├── test_pipeline.py    # mocked end-to-end pipeline test (no keys needed)
└── README.md
```

Run the tests without any API keys:

```bash
python test_step2.py && python test_pipeline.py
```

## Credits

Inspired by Andrej Karpathy's LLM Council concept and the multi-model plumbing pattern from [gcpdev/llm-council-skill](https://github.com/gcpdev/llm-council-skill). This project combines both: multi-model *and* multi-persona, with async parallelism and graceful degradation layered on top.

#!/usr/bin/env python3
"""
LLM Council Hybrid
Runs a decision through 8 parallel advisors (3 external models + 5 internal
Claude sub-agents), peer-reviews the answers anonymously with 5 internal
Claude reviewers, then synthesizes a final chairman verdict.

Pipeline:
  question
    -> 8 voices in parallel (asyncio.gather, T=0)
    -> anonymize (shuffle, label A-H)
    -> 5 Claude peer reviewers in parallel
    -> chairman (sees de-anonymized responses + all reviews)
    -> structured markdown verdict printed to terminal
"""

import argparse
import asyncio
import os
import random
import string
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

GPT_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-1.5-pro"
PERPLEXITY_MODEL = "sonar-pro"
CLAUDE_MODEL = "claude-sonnet-4-5"

TIMEOUT_FIRST = 10.0   # seconds, first attempt (external calls)
TIMEOUT_RETRY = 8.0    # seconds, retry attempt (external calls)
CLAUDE_MAX_TOKENS = 1024
CHAIRMAN_MAX_TOKENS = 2048

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ---------------------------------------------------------------------------
# Lens prompts — the lens IS the system prompt. No generic "analyze this".
# ---------------------------------------------------------------------------

LENS_PROMPTS = {
    "Executor": (
        "You are The Executor on an LLM Council. You only care about one thing: "
        "what can actually be done, and what is the fastest path to doing it? "
        "Ignore theory, strategy, and big-picture thinking. Look at every idea "
        "through the lens of 'what do I do Monday morning?' If an idea sounds "
        "brilliant but has no clear first step, say so. Be direct. Don't hedge. "
        "150-300 words."
    ),
    "Contrarian": (
        "You are The Contrarian on an LLM Council. Your job is to attack the "
        "premise. Assume the plan as stated is wrong, and find the strongest "
        "case for why. Where is the hidden assumption? What is everyone taking "
        "for granted that could collapse? Do not offer balance — other council "
        "members handle that. If you genuinely can't break it, say which single "
        "attack came closest. Be direct. Don't hedge. 150-300 words."
    ),
    "First Principles": (
        "You are First Principles on an LLM Council. Strip the question down to "
        "its atoms: what is actually true here, independent of convention, "
        "analogy, or what others do? Rebuild the answer from the ground up. "
        "Ignore 'best practices' — derive what's right from the fundamentals of "
        "this specific situation. If the question itself is malformed, restate "
        "it correctly first. Be direct. Don't hedge. 150-300 words."
    ),
    "Expansionist": (
        "You are The Expansionist on an LLM Council. Your job is to find the "
        "upside everyone else misses. What is the biggest version of this idea? "
        "What doors does it open that the asker hasn't noticed? What adjacent "
        "opportunity is hiding one step away? Do not manage risk — other "
        "council members handle that. Think bigger, then bigger again. Be "
        "direct. Don't hedge. 150-300 words."
    ),
    "Outsider": (
        "You are The Outsider on an LLM Council. You have fresh eyes and no "
        "prior context or loyalty to this project. React the way a smart "
        "stranger would on first contact: what seems strange, overcomplicated, "
        "or missing to someone who just walked in? Question the vocabulary and "
        "the framing itself. If the emperor has no clothes, say so plainly. Be "
        "direct. Don't hedge. 150-300 words."
    ),
}

# Perplexity's Outsider variant leans on its live web search — that's WHY it
# holds this seat: a genuine outside view backed by what the world says today.
PERPLEXITY_OUTSIDER_PROMPT = LENS_PROMPTS["Outsider"] + (
    " You have live web access — use it. Ground your outside view in what is "
    "actually happening in the world right now: current tools, competitors, "
    "prices, recent news. Cite what you find."
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VoiceResponse:
    """Uniform shape every voice returns so downstream stages (anonymize,
    peer review, chairman) never need to care where a response came from."""
    source: str            # "GPT" | "Gemini" | "Perplexity" | "Internal Claude"
    lens: str              # "Executor" | "Contrarian" | ...
    text: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def label(self) -> str:
        return f"{self.source} — {self.lens} lens"


# ---------------------------------------------------------------------------
# Shared retry wrapper for external calls
# ---------------------------------------------------------------------------

async def with_retry(source: str, lens: str,
                     attempt_fn: Callable[[float], "asyncio.Future"]) -> VoiceResponse:
    """Error pattern from the spec:
    - first attempt: 10s hard timeout
    - on timeout/error: retry ONCE with 8s timeout
    - if retry also fails: return an error VoiceResponse (never raise)
    """
    last_error = "unknown"
    for timeout in (TIMEOUT_FIRST, TIMEOUT_RETRY):
        try:
            text = await attempt_fn(timeout)
            return VoiceResponse(source=source, lens=lens, text=text)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
    return VoiceResponse(source=source, lens=lens,
                         error=f"unavailable after retry ({last_error})")


# ---------------------------------------------------------------------------
# External voices
# ---------------------------------------------------------------------------

async def call_gpt_executor(question: str) -> VoiceResponse:
    """GPT-4o = The Executor. OpenAI shape: Bearer header, system role message,
    text at choices[0].message.content."""
    async def attempt(timeout: float) -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": GPT_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": LENS_PROMPTS["Executor"]},
                        {"role": "user", "content": question},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    return await with_retry("GPT", "Executor", attempt)


async def call_gemini_expansionist(question: str) -> VoiceResponse:
    """Gemini 1.5 Pro = The Expansionist. Gemini shape: key as query param,
    system prompt in system_instruction, temperature in generationConfig,
    text at candidates[0].content.parts[0].text."""
    async def attempt(timeout: float) -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GOOGLE_API_KEY},
                json={
                    "system_instruction": {
                        "parts": [{"text": LENS_PROMPTS["Expansionist"]}]
                    },
                    "contents": [{"role": "user", "parts": [{"text": question}]}],
                    "generationConfig": {"temperature": 0},
                },
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    return await with_retry("Gemini", "Expansionist", attempt)


async def call_perplexity_outsider(question: str) -> VoiceResponse:
    """Perplexity sonar-pro = The Outsider WITH live web search.
    OpenAI-compatible chat shape, different base URL. This is the voice most
    likely to hit the timeout (it searches the web mid-call) — which is exactly
    why the retry/graceful-degradation pattern exists."""
    async def attempt(timeout: float) -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"},
                json={
                    "model": PERPLEXITY_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": PERPLEXITY_OUTSIDER_PROMPT},
                        {"role": "user", "content": question},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    return await with_retry("Perplexity", "Outsider", attempt)


# ---------------------------------------------------------------------------
# Internal Claude voices — one function, five lenses
# ---------------------------------------------------------------------------

async def call_claude(system: str, user: str, max_tokens: int = CLAUDE_MAX_TOKENS) -> str:
    """Single choke point for every internal Claude call (lens agents,
    reviewers, chairman). T=0 everywhere per the spec."""
    resp = await anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


async def call_claude_lens(question: str, lens: str) -> VoiceResponse:
    """Internal Claude sub-agent for one lens. Uses the SAME lens prompts as
    the external models — the seat defines the voice, not the model.
    The SDK has its own retries; if it still fails we degrade gracefully
    like an external voice would."""
    try:
        text = await call_claude(LENS_PROMPTS[lens], question)
        return VoiceResponse(source="Internal Claude", lens=lens, text=text)
    except Exception as e:
        return VoiceResponse(source="Internal Claude", lens=lens,
                             error=f"unavailable ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Phase 1 — fire all 8 voices, wait for ALL of them
# ---------------------------------------------------------------------------

async def run_council(question: str) -> list[VoiceResponse]:
    tasks = [
        call_gpt_executor(question),
        call_gemini_expansionist(question),
        call_perplexity_outsider(question),
        call_claude_lens(question, "Contrarian"),
        call_claude_lens(question, "First Principles"),
        call_claude_lens(question, "Expansionist"),
        call_claude_lens(question, "Outsider"),
        call_claude_lens(question, "Executor"),
    ]
    # Every task already catches its own exceptions and returns a
    # VoiceResponse, so gather gets 8 clean objects — but return_exceptions
    # stays on as a last line of defense against bugs in our own code.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    cleaned: list[VoiceResponse] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            cleaned.append(VoiceResponse(source="unknown", lens=f"task-{i}",
                                         error=f"internal bug: {r}"))
        else:
            cleaned.append(r)
    return cleaned


# ---------------------------------------------------------------------------
# Phase 2 — anonymize + peer review (internal Claude only)
# ---------------------------------------------------------------------------

def anonymize(responses: list[VoiceResponse],
              rng: Optional[random.Random] = None) -> tuple[str, list[tuple[str, VoiceResponse]]]:
    """Shuffle the successful responses and label them A, B, C...
    Returns (packet_text, [(letter, response), ...]) — the mapping is kept
    so the chairman can de-anonymize later. Failed voices are excluded from
    the review packet (reviewers can't review an error message) but the
    chairman is still told about them."""
    rng = rng or random.Random()
    ok_responses = [r for r in responses if r.ok]
    shuffled = ok_responses[:]
    rng.shuffle(shuffled)
    labeled = list(zip(string.ascii_uppercase, shuffled))
    packet = "\n\n".join(
        f"### Response {letter}\n{r.text}" for letter, r in labeled
    )
    return packet, labeled


REVIEWER_SYSTEM = (
    "You are a peer reviewer on an LLM Council. You will receive the original "
    "question and several anonymized responses labeled with letters. You do "
    "not know which model or persona wrote which response — judge only the "
    "content. Answer exactly three questions:\n"
    "1. Which response is strongest, and why?\n"
    "2. Which response has the biggest blind spot, and what is it missing?\n"
    "3. What did ALL responses miss that the council should consider?\n"
    "Refer to responses by their letters. Be direct and specific. Don't hedge."
)


async def run_peer_review(question: str, packet: str, n_reviewers: int = 5) -> list[str]:
    """5 internal Claude reviewers in parallel, each reading the full
    anonymized packet. Internal-only by design: sending responses back out to
    GPT/Gemini/Perplexity would double external cost for no proportional gain."""
    user_msg = f"Original question:\n{question}\n\nAnonymized council responses:\n\n{packet}"

    async def one_review(i: int) -> str:
        try:
            return await call_claude(REVIEWER_SYSTEM, user_msg)
        except Exception as e:
            return f"[Reviewer {i + 1} unavailable: {type(e).__name__}]"

    return await asyncio.gather(*(one_review(i) for i in range(n_reviewers)))


# ---------------------------------------------------------------------------
# Phase 3 — chairman synthesis
# ---------------------------------------------------------------------------

CHAIRMAN_SYSTEM = (
    "You are the Chairman of an LLM Council. You receive the original "
    "question, the full set of council responses WITH their source labels "
    "(which model and which lens produced each one), and a set of anonymous "
    "peer reviews. Weigh everything and produce the final verdict.\n\n"
    "Output EXACTLY this markdown structure and nothing else:\n"
    "## Council Verdict: {short topic}\n"
    "### Where the Council Agrees\n"
    "### Where the Council Clashes\n"
    "### Blind Spots the Council Caught\n"
    "### The Recommendation\n"
    "### The One Thing to Do First\n\n"
    "Be concrete and decisive. The verdict must be usable by someone who "
    "reads nothing else."
)


def build_chairman_input(question: str,
                         responses: list[VoiceResponse],
                         reviews: list[str]) -> str:
    parts = [f"Original question:\n{question}\n", "## Council responses (de-anonymized)\n"]
    for r in responses:
        if r.ok:
            parts.append(f"### {r.label}\n{r.text}\n")
        else:
            # Spec: chairman is notified about failed voices explicitly.
            parts.append(f"### {r.label}\n[{r.source} call failed — treat as absent voice: {r.error}]\n")
    parts.append("## Peer reviews (anonymous reviewers)\n")
    for i, rev in enumerate(reviews, 1):
        parts.append(f"### Reviewer {i}\n{rev}\n")
    return "\n".join(parts)


async def run_chairman(question: str,
                       responses: list[VoiceResponse],
                       reviews: list[str]) -> str:
    return await call_claude(
        CHAIRMAN_SYSTEM,
        build_chairman_input(question, responses, reviews),
        max_tokens=CHAIRMAN_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def council_pipeline(question: str, verbose: bool = True) -> str:
    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    log(f"⚖️  Council convening on: {question}")
    log("→ Phase 1/3: firing 8 voices in parallel (T=0)...")
    responses = await run_council(question)

    ok = [r for r in responses if r.ok]
    failed = [r for r in responses if not r.ok]
    log(f"   {len(ok)}/8 voices answered" +
        (f" — absent: {', '.join(r.label for r in failed)}" if failed else ""))
    if not ok:
        return "All 8 voices failed — check API keys and connectivity."

    log("→ Phase 2/3: anonymizing + 5 parallel peer reviewers...")
    packet, _ = anonymize(responses)
    reviews = await run_peer_review(question, packet)

    log("→ Phase 3/3: chairman synthesis...")
    verdict = await run_chairman(question, responses, reviews)
    log("✓ Done.\n")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a question through the LLM Council")
    parser.add_argument("question", nargs="*", help="The question/decision to put to the council")
    parser.add_argument("-q", "--quiet", action="store_true", help="Print only the verdict")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("What's the question for the council? ").strip()
    if not question:
        print("No question given, exiting.")
        sys.exit(1)

    missing = [name for name, val in [
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("GOOGLE_API_KEY", GOOGLE_API_KEY),
        ("PERPLEXITY_API_KEY", PERPLEXITY_API_KEY),
        ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    ] if not val]
    if "ANTHROPIC_API_KEY" in missing:
        print("ANTHROPIC_API_KEY is required (internal voices, reviewers, chairman). "
              "Copy .env.template to .env and fill it in.")
        sys.exit(1)
    if missing:
        print(f"Note: missing keys ({', '.join(missing)}) — those voices will "
              f"degrade gracefully and the chairman will be notified.", file=sys.stderr)

    verdict = asyncio.run(council_pipeline(question, verbose=not args.quiet))
    print(verdict)


if __name__ == "__main__":
    main()

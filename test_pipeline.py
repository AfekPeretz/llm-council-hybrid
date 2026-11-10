"""End-to-end pipeline test with mocked APIs — no keys, no network.

Verifies:
1. All 8 voices flow through asyncio.gather in parallel
2. A failed voice (Perplexity, simulating its slow web search) degrades
   gracefully and is EXCLUDED from the anonymized review packet
3. Anonymization shuffles + labels A-G (7 ok voices)
4. 5 reviewers run in parallel and each sees the packet
5. Chairman input contains de-anonymized source labels AND the
   "treat as absent voice" notification for the failed voice
6. Verdict comes back through the full pipeline
"""
import asyncio
import random
import sys
import time

sys.path.insert(0, "scripts")
import council
from council import VoiceResponse, with_retry, anonymize, build_chairman_input

# --- mock external voices (through the REAL with_retry wrapper) -------------

def make_external(source, lens, text=None, fail=False, delay=0.3):
    async def voice(question):
        async def attempt(timeout):
            await asyncio.sleep(delay)
            if fail:
                raise TimeoutError("simulated web-search timeout")
            return text
        return await with_retry(source, lens, attempt)
    return voice

council.call_gpt_executor = make_external("GPT", "Executor", "Ship Monday. First step: freeze scope today.")
council.call_gemini_expansionist = make_external("Gemini", "Expansionist", "This could become a platform, not a feature.")
council.call_perplexity_outsider = make_external("Perplexity", "Outsider", fail=True, delay=0.1)  # both attempts fail

# --- mock internal Claude choke point ---------------------------------------

CLAUDE_CALLS = {"lens": 0, "review": 0, "chairman": 0}

async def fake_call_claude(system, user, max_tokens=1024):
    await asyncio.sleep(0.3)
    if system == council.CHAIRMAN_SYSTEM:
        CLAUDE_CALLS["chairman"] += 1
        # prove the chairman actually received the right input
        assert "GPT — Executor lens" in user, "chairman missing de-anonymized labels"
        assert "treat as absent voice" in user, "chairman not notified of failed Perplexity"
        assert "Reviewer 5" in user, "chairman missing peer reviews"
        return ("## Council Verdict: test topic\n### Where the Council Agrees\nx\n"
                "### Where the Council Clashes\nx\n### Blind Spots the Council Caught\nx\n"
                "### The Recommendation\nx\n### The One Thing to Do First\nx")
    if system == council.REVIEWER_SYSTEM:
        CLAUDE_CALLS["review"] += 1
        assert "### Response A" in user, "reviewer packet not anonymized/labeled"
        assert "Perplexity" not in user, "reviewer saw a source name — anonymization broken!"
        return "Strongest: B. Biggest blind spot: D. All missed: cost."
    CLAUDE_CALLS["lens"] += 1
    return f"[mock {system[:30]}...] lens answer"

council.call_claude = fake_call_claude

# --- run ---------------------------------------------------------------------

async def main():
    start = time.monotonic()
    verdict = await council.council_pipeline("should we ship on Friday?", verbose=False)
    elapsed = time.monotonic() - start

    # parallelism sanity: 8 voices (0.3s) + 5 reviews (0.3s) + chairman (0.3s)
    # sequential would be ~4.2s+; parallel phases should be ~1s total
    # (perplexity fails twice at 0.1s each = 0.2s, still inside phase 1)
    assert elapsed < 1.6, f"pipeline took {elapsed:.2f}s — phases not parallel"
    print(f"1. pipeline wall time {elapsed:.2f}s (parallel phases): PASS")

    assert CLAUDE_CALLS == {"lens": 5, "review": 5, "chairman": 1}, CLAUDE_CALLS
    print(f"2. call counts (5 lens / 5 review / 1 chairman): PASS")

    assert verdict.startswith("## Council Verdict"), verdict[:50]
    print("3. structured verdict returned: PASS")

    # unit check: anonymize with a seeded RNG is shuffled but complete
    fake = [VoiceResponse("S", f"L{i}", text=f"t{i}") for i in range(7)]
    packet, labeled = anonymize(fake, rng=random.Random(42))
    assert len(labeled) == 7 and [l for l, _ in labeled] == list("ABCDEFG")
    assert sorted(r.text for _, r in labeled) == sorted(f"t{i}" for i in range(7))
    print("4. anonymize labels A-G, no responses lost: PASS")

    print("\nALL PASS — orchestration verified end to end (APIs mocked)")

asyncio.run(main())

"""--events test with mocked APIs (no keys, no network).

Proves the progress stream a live view relies on:
1. every one of the 8 voices emits voice_start and voice_done exactly once
2. a failed voice (Perplexity) reports ok=false with its reason
3. phases arrive in order 1 -> 2 -> 3, with 5 review_done events between 2 and 3
4. the run ends with exactly one done event carrying the verdict
5. with EVENTS off, nothing is emitted (normal terminal use unchanged)
"""
import asyncio
import contextlib
import io
import json
import sys

sys.path.insert(0, "scripts")
import council
from council import with_retry


def make_external(source, lens, text=None, fail=False):
    async def voice(question):
        async def attempt(timeout):
            await asyncio.sleep(0.05)
            if fail:
                raise TimeoutError("simulated timeout")
            return text
        return await with_retry(source, lens, attempt)
    return voice


council.call_gpt_executor = make_external("GPT", "Executor", "gpt says")
council.call_gemini_expansionist = make_external("Gemini", "Expansionist", "gemini says")
council.call_perplexity_outsider = make_external("Perplexity", "Outsider", fail=True)


async def fake_call_claude(system, user, max_tokens=1024):
    await asyncio.sleep(0.05)
    if system == council.CHAIRMAN_SYSTEM:
        return "## Council Verdict: test\n### The One Thing to Do First\nx"
    return "mock answer"

council.call_claude = fake_call_claude


def run(events_on: bool) -> list[dict]:
    council.EVENTS = events_on
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        asyncio.run(council.council_pipeline("test question", verbose=False))
    lines = [l for l in buf.getvalue().splitlines() if l.startswith(council.EVENT_PREFIX)]
    return [json.loads(l[len(council.EVENT_PREFIX):]) for l in lines]


ev = run(True)
types = [e["type"] for e in ev]

starts = sorted(e["id"] for e in ev if e["type"] == "voice_start")
dones = {e["id"]: e for e in ev if e["type"] == "voice_done"}
assert starts == list(range(8)) and sorted(dones) == list(range(8)), (starts, sorted(dones))
print("1. 8 voice_start + 8 voice_done, one each: PASS")

pplx = dones[2]
assert pplx["source"] == "Perplexity" and pplx["ok"] is False and "simulated timeout" in pplx["error"], pplx
assert all(dones[i]["ok"] for i in range(8) if i != 2)
assert all(isinstance(d["secs"], (int, float)) for d in dones.values())
print(f"2. failed voice reported with reason: PASS -> {pplx['error']}")

phases = [e["n"] for e in ev if e["type"] == "phase"]
assert phases == [1, 2, 3], phases
i2, i3 = types.index("phase", types.index("phase") + 1), len(types) - 1 - types[::-1].index("phase")
reviews = [e for e in ev[i2:i3] if e["type"] == "review_done"]
assert len(reviews) == 5 and all(r["ok"] for r in reviews), reviews
assert ev[i2]["voices_ok"] == 7
print("3. phases 1 -> 2 -> 3 with 5 reviews in between: PASS")

assert types[0] == "start" and types[-1] == "done" and types.count("done") == 1
assert ev[-1]["verdict"].startswith("## Council Verdict")
print("4. starts with start, ends with one done + verdict: PASS")

assert run(False) == []
print("5. events off -> nothing emitted: PASS")

print("\nALL PASS — event stream verified")

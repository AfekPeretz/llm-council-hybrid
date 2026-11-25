"""Step 2 verification — no real keys needed.

Proves three things:
1. with_retry returns success on first attempt
2. with_retry retries once, succeeds on retry
3. with_retry fails both attempts -> error VoiceResponse, never raises
4. asyncio.gather actually runs calls in PARALLEL (wall clock check)
"""
import asyncio
import time
import sys

sys.path.insert(0, "scripts")
from council import with_retry, VoiceResponse


async def main():
    # 1. success first try
    async def ok(timeout):
        return "fine"
    r = await with_retry("Test", "Lens", ok)
    assert r.ok and r.text == "fine"
    print("1. first-attempt success: PASS")

    # 2. fail once, succeed on retry
    state = {"calls": 0}
    async def flaky(timeout):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TimeoutError("simulated slow API")
        return "recovered"
    r = await with_retry("Test", "Lens", flaky)
    assert r.ok and r.text == "recovered" and state["calls"] == 2
    print("2. retry-then-success: PASS")

    # 3. fail both -> error object, no exception escapes
    async def dead(timeout):
        raise ConnectionError("api down")
    r = await with_retry("Perplexity", "Outsider", dead)
    assert not r.ok and r.error.startswith("unavailable (")
    print(f"3. graceful failure: PASS -> {r.error}")

    # 3b. a 404 must NOT be retried (it won't fix itself), and the API's explanation must show
    import httpx
    calls = {"n": 0}
    async def not_found(timeout):
        calls["n"] += 1
        req = httpx.Request("POST", "https://example.test")
        raise httpx.HTTPStatusError("404", request=req,
            response=httpx.Response(404, request=req, text="model not found"))
    r = await with_retry("Gemini", "Expansionist", not_found)
    assert not r.ok and calls["n"] == 1 and "model not found" in r.error, (calls, r.error)
    print(f"3b. 404 tried once, reason shown: PASS -> {r.error}")

    # 4. parallelism: 3 calls x 1s each should take ~1s total, not ~3s
    async def slow(timeout):
        await asyncio.sleep(1.0)
        return "slow done"
    start = time.monotonic()
    results = await asyncio.gather(
        with_retry("A", "x", slow),
        with_retry("B", "y", slow),
        with_retry("C", "z", slow),
    )
    elapsed = time.monotonic() - start
    assert all(r.ok for r in results)
    assert elapsed < 1.5, f"took {elapsed:.2f}s — that's sequential, not parallel!"
    print(f"4. parallelism: PASS — 3x 1s calls finished in {elapsed:.2f}s wall time")


asyncio.run(main())

"""Test RayCluster connectivity and model integration.

Verifies that the gpt-oss-20b model on the SV RayCluster is reachable
and that our LM wrappers correctly route through the configured api_base.
"""

import sys
import time

import litellm


def test_raw_http():
    """Test 1: Raw HTTP connectivity to the RayCluster /v1/models endpoint."""
    import urllib.request
    import json

    url = "http://10.0.10.66:8123/v1/models"
    print(f"[1/5] GET {url} ... ", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer sv-openai-api-key"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"OK — {len(models)} model(s): {models}")
            return True
    except Exception as e:
        print(f"FAIL — {e}")
        return False


def test_litellm_completion():
    """Test 2: LiteLLM completion() with openai/ prefix + api_base."""
    print("[2/5] litellm.completion(openai/openai/gpt-oss-120b) ... ", end="", flush=True)
    try:
        t0 = time.time()
        resp = litellm.completion(
            model="openai/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            api_base="http://10.0.10.66:8123/v1",
            api_key="sv-openai-api-key",
            temperature=0.6,
            max_tokens=32,
            timeout=30,
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        # Some models return reasoning in a separate field
        if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            content = f"(reasoning only, {len(msg.reasoning_content)} chars)"
        elapsed = time.time() - t0
        print(f"OK — response: {content!r} ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"FAIL — {e}")
        return False


def test_lm_wrapper():
    """Test 3: Our LM wrapper class with Settings-driven config."""
    from gepa_mutations.config import Settings
    from gepa_mutations.runner.experiment import LM
    from gepa_mutations.config import model_id, api_base_kwargs

    settings = Settings()
    model = model_id(settings)
    api_kwargs = api_base_kwargs(settings)

    print(f"[3/5] LM wrapper (model={model}, api_base={api_kwargs.get('api_base', 'default')}) ... ", end="", flush=True)
    try:
        lm = LM(
            model,
            temperature=settings.gepa_temperature,
            max_tokens=64,
            top_p=settings.gepa_top_p,
            **api_kwargs,
        )
        t0 = time.time()
        result = lm("What is the capital of France? Answer in one word.")
        elapsed = time.time() - t0
        print(f"OK — response: {result.strip()!r} ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"FAIL — {e}")
        return False


def test_reflection_lm():
    """Test 4: build_reflection_lm() from base.py."""
    from gepa_mutations.config import Settings
    from gepa_mutations.base import build_reflection_lm

    print("[4/5] build_reflection_lm() ... ", end="", flush=True)
    try:
        settings = Settings()
        lm = build_reflection_lm(settings)
        t0 = time.time()
        result = lm("Say 'hello' and nothing else.")
        elapsed = time.time() - t0
        print(f"OK — response: {result.strip()!r} ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"FAIL — {e}")
        return False


def test_multi_turn():
    """Test 5: Multi-turn messages (system + user) matching GEPA's usage pattern."""
    from gepa_mutations.config import Settings
    from gepa_mutations.runner.experiment import LM
    from gepa_mutations.config import model_id, api_base_kwargs

    settings = Settings()
    model = model_id(settings)
    api_kwargs = api_base_kwargs(settings)

    print(f"[5/5] Multi-turn messages (system+user) ... ", end="", flush=True)
    try:
        lm = LM(model, temperature=0.6, max_tokens=128, **api_kwargs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": "What are the first 5 prime numbers? List them separated by commas."},
        ]
        t0 = time.time()
        result = lm(messages)
        elapsed = time.time() - t0
        print(f"OK — response: {result.strip()!r} ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"FAIL — {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RayCluster Connectivity Tests")
    print("Target: http://10.0.10.66:8123/v1 (gpt-oss-20b)")
    print("=" * 60)
    print()

    results = []
    results.append(("Raw HTTP", test_raw_http()))
    print()
    results.append(("LiteLLM completion", test_litellm_completion()))
    print()
    results.append(("LM wrapper", test_lm_wrapper()))
    print()
    results.append(("Reflection LM", test_reflection_lm()))
    print()
    results.append(("Multi-turn", test_multi_turn()))

    print()
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)

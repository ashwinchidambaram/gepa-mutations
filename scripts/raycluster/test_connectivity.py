"""Test connectivity and measure latency against the cluster inference API.

Run this FIRST before deploying experiments. Can be run from either:
- Local Mac (over VPN) to verify connectivity
- gho-vm-2 (on cluster network) to measure true latency

Usage:
    python scripts/raycluster/test_connectivity.py
"""

import json
import sys
import time
from pathlib import Path

import requests

# Allow running from repo root or from scripts/raycluster/
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INFERENCE_BASE_URL,
    INFERENCE_HOST,
    INFERENCE_PORT,
    MAX_TOKENS_QA,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
)


def check_api_reachable() -> bool:
    """Check if the inference API is reachable."""
    print(f"[1/5] Checking API at {INFERENCE_BASE_URL}...")
    try:
        resp = requests.get(f"{INFERENCE_BASE_URL}/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json()
            print(f"  OK — API reachable. Available models:")
            for m in models.get("data", []):
                print(f"    - {m['id']}")
            return True
        else:
            print(f"  FAIL — Status {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.ConnectionError:
        print(f"  FAIL — Connection refused. Is VPN active? Is {INFERENCE_HOST}:{INFERENCE_PORT} up?")
        return False
    except requests.Timeout:
        print(f"  FAIL — Timeout after 10s.")
        return False


def test_short_generation() -> dict | None:
    """Test short generation (evaluation-style) and measure latency."""
    print(f"\n[2/5] Short generation test (max_tokens={MAX_TOKENS_QA})...")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        "max_tokens": MAX_TOKENS_QA,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{INFERENCE_BASE_URL}/chat/completions",
            json=payload,
            timeout=60,
        )
        latency = time.time() - t0
        data = resp.json()
        msg = data["choices"][0]["message"]

        # Check where the response actually lives
        content = msg.get("content")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")

        print(f"  Latency: {latency:.2f}s")
        print(f"  content field: {repr(content)}")
        print(f"  reasoning_content field: {repr(reasoning)[:200] if reasoning else 'None'}")
        print(f"  Usage: {data.get('usage', {})}")

        return {
            "latency_s": latency,
            "content": content,
            "reasoning_content": reasoning,
            "usage": data.get("usage"),
            "full_message": msg,
        }
    except Exception as e:
        print(f"  FAIL — {e}")
        return None


def test_thinking_disabled() -> dict | None:
    """Test if we can disable thinking mode via chat_template_kwargs."""
    print(f"\n[3/5] Testing with thinking disabled (chat_template_kwargs)...")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the capital of France? One word answer."}],
        "max_tokens": MAX_TOKENS_QA,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{INFERENCE_BASE_URL}/chat/completions",
            json=payload,
            timeout=60,
        )
        latency = time.time() - t0
        data = resp.json()

        if "error" in data:
            print(f"  chat_template_kwargs not supported: {data['error']}")
            # Try extra_body approach
            return _test_thinking_disabled_extra_body()

        msg = data["choices"][0]["message"]
        content = msg.get("content")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")

        print(f"  Latency: {latency:.2f}s")
        print(f"  content field: {repr(content)}")
        print(f"  reasoning_content field: {repr(reasoning)[:100] if reasoning else 'None'}")

        thinking_disabled = content is not None and content.strip() != ""
        print(f"  Thinking mode disabled: {'YES' if thinking_disabled else 'NO (content still null)'}")

        return {
            "latency_s": latency,
            "content": content,
            "reasoning_content": reasoning,
            "thinking_disabled": thinking_disabled,
        }
    except Exception as e:
        print(f"  FAIL — {e}")
        return None


def _test_thinking_disabled_extra_body() -> dict | None:
    """Fallback: try extra_body or model-specific params to disable thinking."""
    print("  Trying extra_body approach...")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the capital of France? One word answer."}],
        "max_tokens": MAX_TOKENS_QA,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{INFERENCE_BASE_URL}/chat/completions",
            json=payload,
            timeout=60,
        )
        latency = time.time() - t0
        data = resp.json()
        if "error" in data:
            print(f"  extra_body also not supported: {data['error']}")
            return {"thinking_disabled": False, "note": "Cannot disable thinking mode"}

        msg = data["choices"][0]["message"]
        content = msg.get("content")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        thinking_disabled = content is not None and content.strip() != ""
        print(f"  Latency: {latency:.2f}s")
        print(f"  content: {repr(content)}")
        print(f"  Thinking disabled: {'YES' if thinking_disabled else 'NO'}")
        return {"latency_s": latency, "thinking_disabled": thinking_disabled}
    except Exception as e:
        print(f"  FAIL — {e}")
        return None


def test_medium_generation() -> dict | None:
    """Test medium-length generation to measure latency scaling."""
    print(f"\n[4/5] Medium generation test (max_tokens=100)...")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Explain photosynthesis in 2 sentences."}],
        "max_tokens": 100,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{INFERENCE_BASE_URL}/chat/completions",
            json=payload,
            timeout=120,
        )
        latency = time.time() - t0
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        response_text = content or reasoning or ""

        print(f"  Latency: {latency:.2f}s")
        print(f"  Response length: {len(response_text)} chars")
        print(f"  Usage: {data.get('usage', {})}")
        print(f"  Response: {response_text[:200]}")

        return {"latency_s": latency, "response_length": len(response_text)}
    except Exception as e:
        print(f"  FAIL — {e}")
        return None


def test_benchmark_style_query() -> dict | None:
    """Test a realistic benchmark query (HotpotQA style)."""
    print(f"\n[5/5] Benchmark-style query (HotpotQA format)...")

    system_prompt = "You are a helpful assistant. Answer the question concisely."
    user_prompt = (
        "Context:\nAlbert Einstein was born in Ulm, Germany in 1879. "
        "He developed the theory of relativity.\n\n"
        "Question: Where was Einstein born?"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS_QA,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{INFERENCE_BASE_URL}/chat/completions",
            json=payload,
            timeout=60,
        )
        latency = time.time() - t0
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        response_text = content or reasoning or ""

        # Check if answer is correct
        correct = "ulm" in response_text.lower()

        print(f"  Latency: {latency:.2f}s")
        print(f"  Response: {repr(response_text)}")
        print(f"  Correct (contains 'Ulm'): {correct}")

        return {"latency_s": latency, "response": response_text, "correct": correct}
    except Exception as e:
        print(f"  FAIL — {e}")
        return None


def main():
    print("=" * 60)
    print("Raycluster Connectivity & Latency Test")
    print(f"Target: {INFERENCE_BASE_URL} ({MODEL_NAME})")
    print("=" * 60)

    results = {}

    # Step 1: Check reachability
    if not check_api_reachable():
        print("\n ABORT — API not reachable. Check VPN and cluster status.")
        sys.exit(1)

    # Step 2-5: Run tests
    results["short"] = test_short_generation()
    results["thinking_disabled"] = test_thinking_disabled()
    results["medium"] = test_medium_generation()
    results["benchmark"] = test_benchmark_style_query()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Determine response extraction strategy
    if results.get("thinking_disabled", {}).get("thinking_disabled"):
        print("Strategy: Thinking mode CAN be disabled — use content field directly")
    elif results.get("short", {}).get("content"):
        print("Strategy: Content field is populated — standard extraction works")
    else:
        print("Strategy: Must extract from reasoning_content field (thinking model)")

    # Latency summary
    latencies = []
    for key in ("short", "medium", "benchmark"):
        if results.get(key) and results[key].get("latency_s"):
            latencies.append((key, results[key]["latency_s"]))
            print(f"  {key}: {results[key]['latency_s']:.2f}s")

    if latencies:
        avg = sum(l for _, l in latencies) / len(latencies)
        print(f"  average: {avg:.2f}s")

    # Feasibility assessment
    print("\nFeasibility for baseline (300 examples × 5 benchmarks = 1500 calls):")
    if results.get("short", {}).get("latency_s"):
        est_minutes = (1500 * results["short"]["latency_s"]) / 60
        print(f"  Estimated time: ~{est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")

    print("\nDone. Results saved to test_connectivity_results.json")

    # Save results
    out_path = Path(__file__).parent / "test_connectivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()

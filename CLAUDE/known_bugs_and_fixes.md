# Known Bugs and Fixes

## 1. gepa state save FileNotFoundError

**Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'gepa_state/run_log.json.tmp'`
Observed in: `qwen3-1.7b / ifbench / gepa / seed=42`

**Root cause:** `gepa/src/gepa/core/state.py` — `_atomic_write_json()` and `save()` call
`os.replace(tmp, final)` without first creating the parent directory. On NFS-mounted paths
the directory may not pre-exist when a new run starts.

**Fix applied** (`gepa/src/gepa/core/state.py`):
```python
def _atomic_write_json(self, run_dir: str, filename: str, data: Any) -> None:
    os.makedirs(run_dir, exist_ok=True)   # ADDED
    target_path = os.path.join(run_dir, filename)
    tmp_path = target_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=json_default)
    os.replace(tmp_path, target_path)

def save(self, run_dir: str | None, *, use_cloudpickle: bool = False) -> None:
    if run_dir is None:
        return
    os.makedirs(run_dir, exist_ok=True)   # ADDED
    ...
```

**Note:** `gepa/` is a git submodule (v0.1.1). This fix is applied directly in the submodule.
Do not pull upstream updates without re-applying this patch.

---

## 2. _env_model_tag() substring collision: "14b" vs "4b"

**Symptom:** `GEPA_MODEL=Qwen/Qwen3-14B` would map to `runs/qwen3-4b/` instead of `runs/qwen3-14b/`

**Root cause:** `scripts/run_all_local.py:_env_model_tag()` — the `if "4b" in m` check precedes
`if "14b" in m`, and `"qwen3-14b"` contains `"4b"` as a substring.

**Fix applied** (`scripts/run_all_local.py`):
```python
def _env_model_tag() -> str:
    m = os.environ.get("GEPA_MODEL", "").lower()
    if "27b" in m: return "qwen3-27b-awq"
    if "14b" in m: return "qwen3-14b"   # must come before "4b"
    if "8b" in m:  return "qwen3-8b"
    if "4b" in m:  return "qwen3-4b"
    if "1.7b" in m or "1b" in m: return "qwen3-1.7b"
    return ""
```

---

## 3. vLLM IPC socket path length bug

**Symptom:** vLLM v1 fails to start with IPC socket error if the current working directory
path exceeds ~107 characters (OS limit on Unix socket path length).

**Fix:** Always `cd /tmp` before launching vLLM. All `serve_vllm_*.sh` scripts do this.

```bash
cd /tmp
HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server ...
```

---

## 4. tournament / long methods invisible to get_in_progress()

**Symptom:** Monitoring shows "0 running" for 8B or 4B even when jobs are active.

**Root cause:** `tournament` (and similar methods) don't write any intermediate checkpoint
files — they only write `result.json` on completion (which can take 2–3 hours). 
`get_in_progress()` uses a 45-min stale mtime window and finds nothing to detect.

**Workaround:** Use vLLM `/metrics` endpoint to confirm real GPU activity:
```bash
curl -s http://<ip>:<port>/metrics | grep num_requests_running
```
A non-zero value confirms experiments are running.

---

## 5. HF_HOME must be set explicitly

**Symptom:** vLLM tries to download models to `~/.cache/huggingface` instead of NFS share;
fails with disk quota errors or re-downloads already-cached models.

**Fix:** Always set `HF_HOME=/models/huggingface` before vLLM launch.
All `serve_vllm_*.sh` scripts set this.

---

## 6. CLAUDE/ folder was gitignored

**Symptom:** `.gitignore` had `CLAUDE.md` on its own line, which gitignored the file but also
gitignored the `CLAUDE/` directory (pattern `CLAUDE.md` matches files named exactly that;
`CLAUDE/` as a directory path was not explicitly handled — but `.gitignore` had `.claude/`
not `CLAUDE/`).

Actually the original `.gitignore` had `.claude/` which ignores the hidden `.claude/` dir
(Claude Code session files). The capital `CLAUDE/` folder is intentionally tracked.

**Resolution:** Confirmed `.gitignore` only ignores `.claude/` (hidden). `CLAUDE/` (capital,
tracked) is unaffected. No change needed.

---

## 7. GEPA Pareto dominance stall (O(n^2) in remove_dominated_programs)

**Symptom:** GEPA optimization freezes silently at iterations 15-23. Process stays alive but
makes no LM calls and produces no log output. The orchestrator's 30-min stall detector
eventually kills it, but no `result.json` is written — entire experiment lost.

Observed in: exp-04 pilot — `gepa/555`, `gepa/999`, `gepa/1337` all stalled at iter 6-23.

**Root cause:** `gepa/src/gepa/gepa_utils.py:remove_dominated_programs()` has a
`while found_to_remove` loop that calls `is_dominated()` for every program against every
other program. Complexity is O(n^2) to O(n^3) per call. This function is invoked every
iteration via `select_program_candidate_from_pareto_front()`.

By iteration 15-23, there are enough candidates that the dominance checking takes minutes.
The stall happens INSIDE an iteration (during `propose()` → `select_candidate_idx()`), so
GEPA's `_should_stop()` (checked only at the top of each iteration) never fires.

This was NOT a thread leak or timeout issue — it was legitimate CPU-bound work with no
log visibility.

**Why it wasn't caught before:** The GEPA paper tested with GPT-5 as reflection LM (5-30s
per reflection call). The O(n^2) overhead (~0.1-1s) was invisible as a percentage of
wall-clock time. On faster local models (Qwen3-8B via vLLM, 0.5-2s per call), the Pareto
overhead becomes the bottleneck.

**Fix applied** (`gepa/src/gepa/gepa_utils.py`):
1. **Self-limiting timeout:** Break out of the `while` loop if elapsed time exceeds
   `_DOMINANCE_TIMEOUT` (default 120s). Returns a conservatively over-inclusive front.
2. **Timing instrumentation:** Log a warning when the function takes >30s.

```python
_DOMINANCE_TIMEOUT: float = 120.0

def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    t0 = time.monotonic()
    # ... existing setup ...
    while found_to_remove:
        if time.monotonic() - t0 > _DOMINANCE_TIMEOUT:
            _logger.warning("remove_dominated_programs timed out ...")
            break
        # ... existing inner loop ...
```

**Why this is safe:** The `while` loop ONLY removes dominated programs. Breaking early means
some dominated programs survive in the sampling list, slightly diluting candidate selection
quality. But no non-dominated program is ever incorrectly removed. The assertion
`any(p in front for p in dominators)` still holds.

**Note:** This is a submodule patch (same as bug #1). Do not pull upstream updates without
re-applying.

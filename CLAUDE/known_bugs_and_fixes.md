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

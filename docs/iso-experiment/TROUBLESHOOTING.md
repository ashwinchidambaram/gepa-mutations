# ISO Experiment Troubleshooting Guide

## V1: Environment Validation Failed

**Symptom:** Python version mismatch or missing packages.

**Diagnostics:**
```bash
python --version
pip list | grep -E "torch|vllm|dspy|mlflow"
```

**Common causes:**
- Wrong Python in PATH (system 3.10 instead of venv 3.12)
- `uv sync` failed silently

**Fix:** Re-run `bash scripts/iso_setup.sh`

---

## V2: Model Weight Validation Failed

**Symptom:** Model files missing or wrong SHA.

**Diagnostics:**
```bash
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/
python -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen3-8B').sha)"
```

**Common causes:**
- Network timeout during download
- HF_HOME points to wrong location

**Fix:** Re-run `bash scripts/iso_download_models.sh`

---

## V3: vLLM Server Launch Failed

**Symptom:** Health check times out on port 8000 or 8001.

**Diagnostics:**
```bash
nvidia-smi                    # GPU visible?
tmux attach -t iso-task       # Check vLLM logs
tmux attach -t iso-reflection
curl http://localhost:8000/health
```

**Common causes:**
- Another process holding GPU memory
- Insufficient VRAM for both models
- Wrong quantization flag (AWQ for reflection model)
- Model not downloaded yet

**Fix:**
1. Kill other GPU processes: `nvidia-smi` then `kill <pid>`
2. Verify models downloaded: `bash scripts/iso_download_models.sh`
3. Restart servers: `bash scripts/iso_start_servers.sh --phase pilot`

---

## V4: Basic Inference Failed

**Symptom:** Server responds to /health but not to completion requests.

**Diagnostics:**
```bash
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
```

**Common causes:**
- Model loading still in progress (wait for "Application startup complete" in logs)
- max_model_len too large for available KV cache

**Fix:** Wait for model loading, or reduce `--max-model-len`

---

## V5: DSPy Integration Failed

**Symptom:** DSPy can't connect to vLLM or produces empty output.

**Diagnostics:**
```python
import dspy
lm = dspy.LM("openai/Qwen/Qwen3-8B", api_base="http://localhost:8000/v1", api_key="EMPTY")
print(lm("test"))
```

**Common causes:**
- Wrong model name in DSPy LM constructor
- DSPy version incompatibility

**Fix:** Verify model name matches `GET /v1/models` output exactly

---

## V6: Benchmark Loading Failed

**Symptom:** Benchmark fails to load or returns empty splits.

**Diagnostics:**
```python
from gepa_mutations.benchmarks.loader import load_benchmark
data = load_benchmark("hotpotqa", seed=0)
print(len(data.train), len(data.val), len(data.test))
```

**Common causes:**
- Network timeout downloading dataset from HuggingFace
- DSPy datasets API change

**Fix:** Check internet connectivity, try loading one benchmark at a time

---

## V7: Logging Validation Failed

**Symptom:** MLflow not reachable or JSONL files corrupt.

**Diagnostics:**
```bash
curl http://localhost:5000  # MLflow UI
ls -la /workspace/mlflow/   # Tracking directory
```

**Common causes:**
- MLflow server not started
- Tracking URI misconfigured
- Disk full

**Fix:** Check `tmux attach -t iso-mlflow`, verify disk space

---

## V8: Checkpoint/Resume Failed

**Symptom:** Resume doesn't restore state correctly.

**Diagnostics:**
```bash
ls runs/*/checkpoints/       # Checkpoint dirs exist?
python -c "
from iso_harness.experiment.checkpoint import load_latest_checkpoint
from pathlib import Path
state = load_latest_checkpoint(Path('runs/<run_id>'))
print(state)
"
```

**Common causes:**
- Corrupt checkpoint JSON (disk full during write)
- Missing candidates.json in checkpoint dir

**Fix:** Delete corrupt checkpoint dir, restart run (resumes from previous valid checkpoint)

---

## V9: End-to-End Smoke Test Failed

**Symptom:** Smoke test doesn't complete or produces missing artifacts.

**Diagnostics:**
```bash
ls runs/*/
cat runs/*/errors.log
cat runs/*/summary.json
```

**Common causes:**
- Server crashed during run (check tmux sessions)
- Budget exhausted before any useful work
- Evaluation timeout (LM calls hanging)

**Fix:** Check server health, increase timeout in config, review error logs

---

## V10: Rsync Validation Failed

**Symptom:** Sync script fails or produces no results.

**Diagnostics:**
```bash
ssh -i ~/.ssh/runpod_key root@<pod-ip> "ls /workspace/iso-experiment/runs/"
```

**Common causes:**
- SSH key not configured or wrong permissions
- Pod IP changed after restart
- No COMPLETE markers (runs still in progress)

**Fix:** Verify SSH access, check pod IP, wait for runs to complete

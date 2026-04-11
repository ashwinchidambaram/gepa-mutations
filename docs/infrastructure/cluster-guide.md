# GPU Cluster Usage Guide — SupportVectors Lab

This document covers how to discover, allocate, and use GPU resources on the SupportVectors lab cluster. Written based on practical experience running vLLM inference for the gepa-mutations project (April 2026).

## Cluster Overview

The lab runs a SLURM-based cluster with multiple GPU nodes. The login/compute VM (`gho-vm-2.lab.supportvectors.ai`) has no GPU itself — it submits jobs to GPU nodes via SLURM.

### Key Commands

```bash
# See all nodes, partitions, and GPU types
sinfo -o "%N %P %G %m %c %T %e"

# See detailed per-node info
sinfo -N -l

# Check what's running
squeue                    # all jobs
squeue -u $(whoami)       # your jobs

# Submit a job
sbatch my_job.sh

# Run a quick interactive command on a node
srun -p <partition> -w <node> --gres=gpu:1 --time=00:05:00 bash -c "nvidia-smi"

# Cancel a job
scancel <jobid>
```

## Available Partitions & Nodes (as of April 2026)

| Node | Partition | GPU | VRAM | RAM | CPUs | Time Limit | Notes |
|------|-----------|-----|------|-----|------|------------|-------|
| **inference** | inference | 2x RTX PRO 6000 | 48GB each | 380GB | 192 | 7 days | Shared vLLM server at 10.0.10.66:8123 |
| **bourbaki** | ray-cluster | RTX 5090 | 32GB | 120GB | 32 | 90 days | Blackwell arch, CUDA 13.0 |
| archimedes | ray-cluster | RTX 4090 | 24GB | 120GB | 32 | 90 days | |
| feynman | ray-cluster | RTX 4090 | 24GB | 60GB | 32 | 90 days | |
| gradient-descent | ray-cluster | RTX 4090 | 24GB | 120GB | 32 | 90 days | |
| deepseek | ray-cluster | RTX 3090 | 24GB | 60GB | 16 | 90 days | |
| **kolmogorov** | student-gpu | RTX 4090 | 24GB | 120GB | 32 | 2 hours | Best student node |
| hinton | student-gpu | RTX 3090 | 24GB | 28GB | 32 | 2 hours | Low RAM |
| gho-gpu-vm | student-gpu | RTX 4060 Ti | 16GB | 46GB | 8 | 2 hours | |
| transformer | student-gpu | RTX 4080 | 16GB | 60GB | 16 | 2 hours | |
| mandelbrot | student-gpu | RTX 4060 Ti | 16GB | 30GB | 16 | 2 hours | |
| sapphire | capstone | RTX 4090 | 24GB | 122GB | 16 | 8 hours | |
| manifold | capstone | RTX 5090 | 32GB | 120GB | 32 | 8 hours | |
| sar-gpu-vm | capstone | RTX 3090 | 24GB | 46GB | 16 | 8 hours | |

### Partition Selection Guide

- **ray-cluster** (90-day limit): Best for long-running inference servers or multi-day experiments. Nodes: bourbaki, archimedes, feynman, gradient-descent, deepseek.
- **student-gpu** (2-hour limit): Quick experiments, debugging, testing. Default partition.
- **capstone** (8-hour limit): Medium-length jobs. Good for training runs.
- **inference** (7-day limit): The shared inference server node. Usually occupied by the shared vLLM instance.

## Network & Filesystem

### Node IP Addresses

Nodes are on a private LAN (`10.0.10.x`). Key addresses:
- `10.0.10.65` — bourbaki
- `10.0.10.66` — inference node (shared vLLM server)

To find a node's IP:
```bash
getent hosts <nodename>
```

### Shared Filesystem

- `/users/achidamb/` — NFS mount from `10.0.10.99:/safe/home`, accessible from all nodes
- `/home/achidamb/` — local to the login VM only, NOT accessible from GPU nodes
- GPU nodes have their own local `/home/` and large local storage at `/dev/nvme*`

**Critical**: Always use `/users/achidamb/` paths for anything that needs to be shared between the login VM and GPU nodes. Virtual environments, model caches, scripts, and data should all live under `/users/`.

## Setting Up vLLM on a GPU Node

### Step 1: Create a Virtual Environment

The GPU nodes don't have `python3-venv` installed, so use `--without-pip` and bootstrap:

```bash
srun -p ray-cluster -w bourbaki --gres=gpu:1 --time=00:10:00 --mem=32G bash -c "
    VENV=/users/achidamb/projects/my-project/vllm-venv
    python3 -m venv --without-pip \$VENV
    curl -sS https://bootstrap.pypa.io/get-pip.py | \$VENV/bin/python3
    echo 'Venv ready'
"
```

### Step 2: Install vLLM

This takes 15-30 minutes (large downloads). Use `sbatch` with a generous time limit rather than `srun`:

```bash
cat > /users/achidamb/projects/my-project/install_vllm.sh << 'EOF'
#!/bin/bash
#SBATCH --partition=ray-cluster
#SBATCH --nodelist=bourbaki
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --job-name=vllm-install
#SBATCH --output=/users/achidamb/projects/my-project/vllm-install.log

VENV=/users/achidamb/projects/my-project/vllm-venv
$VENV/bin/pip install vllm
$VENV/bin/python -c 'import vllm; print("vLLM:", vllm.__version__)'
$VENV/bin/python -c 'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())'
echo "DONE"
EOF

sbatch /users/achidamb/projects/my-project/install_vllm.sh
```

Monitor with: `tail -f /users/achidamb/projects/my-project/vllm-install.log`

### Step 3: Serve a Model

```bash
cat > /users/achidamb/projects/my-project/serve_vllm.sh << 'EOF'
#!/bin/bash
#SBATCH --partition=ray-cluster
#SBATCH --nodelist=bourbaki
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=vllm-server
#SBATCH --output=/users/achidamb/projects/my-project/vllm-server.log

VENV=/users/achidamb/projects/my-project/vllm-venv
PORT=8124
MODEL=QuantTrio/Qwen3.5-27B-AWQ

export HF_TOKEN=<your-hf-token>

$VENV/bin/python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype auto \
    --quantization awq \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --api-key my-api-key \
    --no-enable-log-requests
EOF

sbatch /users/achidamb/projects/my-project/serve_vllm.sh
```

### Step 4: Connect from Your Code

From the login VM or any other node on the LAN:

```python
import litellm
resp = litellm.completion(
    model="openai/QuantTrio/Qwen3.5-27B-AWQ",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://10.0.10.65:8124/v1",   # bourbaki IP + port
    api_key="my-api-key",
    max_tokens=512,
)
```

Check server health:
```bash
# Is the API responding?
curl -s http://10.0.10.65:8124/v1/models -H "Authorization: Bearer my-api-key"

# Queue depth (should be low)
curl -s http://10.0.10.65:8124/metrics | grep -E "num_requests_running|num_requests_waiting" | grep -v "^#"
```

## Model Sizing & Quantization

### VRAM Requirements

| Model Size | FP16 | INT8 | AWQ/GPTQ 4-bit |
|-----------|------|------|-----------------|
| 7B | ~14GB | ~8GB | ~4GB |
| 13B | ~26GB | ~14GB | ~8GB |
| 27B | ~54GB | ~28GB | ~16GB |
| 70B | ~140GB | ~72GB | ~38GB |

Add 2-8GB for KV cache depending on `--max-model-len`. torch.compile adds ~6GB overhead (disable with `--enforce-eager` if tight on memory).

### Quantization Decision Tree

1. **Model fits in VRAM at FP16?** Use `--dtype auto` (best quality)
2. **Fits at INT8?** Use `--quantization bitsandbytes` (note: vLLM bitsandbytes defaults to 8-bit, not 4-bit)
3. **Doesn't fit?** Use a pre-quantized AWQ model from HuggingFace:
   - Search: `huggingface_hub.HfApi().list_models(search='<model>-AWQ')`
   - Use `--quantization awq` — faster than bitsandbytes, ~14GB for 27B models
4. **Still doesn't fit?** Reduce `--max-model-len`, add `--enforce-eager`, lower `--gpu-memory-utilization`

### Lessons Learned: Qwen3.5-27B on RTX 5090 (32GB)

We tried multiple configurations:

| Attempt | Config | Result |
|---------|--------|--------|
| 1 | FP16, max_model_len=20000 | OOM (54GB model > 32GB VRAM) |
| 2 | bitsandbytes, max_model_len=16384 | OOM (8-bit = 26GB + KV cache overflow) |
| 3 | bitsandbytes, max_model_len=4096 | OOM (same — 26GB leaves no KV cache room) |
| 4 | AWQ 4-bit, max_model_len=16384 | OOM (20GB model + 6GB torch.compile = 26GB, KV cache overflow) |
| **5** | **AWQ 4-bit, max_model_len=4096, enforce-eager** | **Success** (20GB model, no compile overhead, small KV cache) |

The key was combining AWQ quantization (~20GB) + `--enforce-eager` (saves ~6GB from torch.compile) + reduced context length.

## Troubleshooting

### "No `python3-venv` available"
Use `python3 -m venv --without-pip` then bootstrap pip with `get-pip.py`.

### "Permission denied" for Docker
User is not in the `docker` group. Use pip-based vLLM installation instead.

### OOM during model loading
- Reduce `--max-model-len` (biggest lever for KV cache)
- Add `--enforce-eager` (saves ~6GB from torch.compile graphs)
- Use AWQ quantized model instead of bitsandbytes
- Lower `--gpu-memory-utilization` (counterintuitively, sometimes helps by reducing pre-allocation pressure)

### SLURM job killed by time limit
`srun` interactive jobs have shorter effective limits. Use `sbatch` for anything over 10 minutes.

### Shared filesystem NFS lag
Model downloads and weight loading can be slow over NFS. First download caches to `~/.cache/huggingface/` on the GPU node's local disk. Subsequent loads are faster.

### vLLM API flag differences across versions
vLLM CLI flags change between versions. Key differences we hit:
- v0.17: `--disable-log-requests`
- v0.19: `--no-enable-log-requests` (old flag removed)

Always check `vllm serve --help` on your installed version.

## Shared Server Etiquette

The inference node (10.0.10.66:8123) runs a shared vLLM server used by multiple people. Key rules:

1. **Check queue before running parallel workers**: `curl -s http://10.0.10.66:8123/metrics | grep num_requests_waiting`
2. **If queue > 5, reduce your parallelism** — you're competing with other users
3. **Qwen models with thinking mode** (`include_reasoning: true`) generate 500-5000 tokens per call and monopolize the GPU. Disable with `extra_body={"chat_template_kwargs": {"enable_thinking": False}, "include_reasoning": False}`
4. **For long experiments, launch your own vLLM on a ray-cluster node** rather than saturating the shared server

## Quick Reference: Full Workflow

```bash
# 1. Check available GPUs
sinfo -o "%N %P %G %m %T"

# 2. Test GPU access
srun -p ray-cluster -w bourbaki --gres=gpu:1 --time=00:01:00 nvidia-smi

# 3. Create venv (one-time)
srun -p ray-cluster -w bourbaki --gres=gpu:1 --time=00:10:00 bash -c "
    python3 -m venv --without-pip /users/\$USER/my-venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | /users/\$USER/my-venv/bin/python3
"

# 4. Install vLLM (sbatch for long install)
sbatch install_script.sh

# 5. Launch server
sbatch serve_script.sh

# 6. Monitor
tail -f /users/$USER/projects/my-project/vllm-server.log
squeue -u $USER
curl -s http://<node-ip>:<port>/v1/models

# 7. Use from code
# Point API_BASE_URL to http://<node-ip>:<port>/v1
```

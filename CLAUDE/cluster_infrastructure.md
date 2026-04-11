# Cluster Infrastructure (sweep/cluster branch)

Small model nodes (1.7B, 4B) have been retired from this cluster and moved to the Mac.
See `sweep/mac` branch for their setup.

## Active Nodes

| Model | Node | Partition | GPU | VRAM | IP:Port | SLURM time limit |
|-------|------|-----------|-----|------|---------|-----------------|
| Qwen3-27B-AWQ | manifold | capstone | RTX 5090 | 32GB | 10.0.10.69:8124 | 8h (chain jobs) |
| Qwen3-8B | archimedes | ray-cluster | RTX 5090 | 32GB | 10.0.10.58:8125 | 90 days |
| Qwen3-14B | kolmogorov | student-gpu | RTX 4090 | 24GB | 10.0.10.52:8128 | 2h (chain jobs) |

## Downed Nodes (since ~2026-04-10 15:23)

| Node | Original role | GPU | Status |
|------|--------------|-----|--------|
| bourbaki | 27B primary | RTX 5090 | down / Not responding |
| ansatz | 4B primary | RTX 4090 | down / Not responding |
| deepseek | 1.7B primary | RTX 3090 | down / Not responding |

Recovery is monitored by `scripts/check_node_recovery.sh` (system crontab, every 15 min).
Sends Telegram alert on recovery. Do NOT auto-relaunch — discuss with user first.

## Retired Nodes (moved to Mac)

| Node | Old role | Reason |
|------|----------|--------|
| sapphire | Qwen3-4B primary | 4B runs moved to Mac (sweep/mac) |
| mandelbrot | Qwen3-4B secondary | 4B runs moved to Mac (sweep/mac) |
| sar-gpu-vm | Qwen3-1.7B primary | 1.7B runs moved to Mac (sweep/mac) |
| gho-gpu-vm | Qwen3-1.7B secondary | 1.7B runs moved to Mac (sweep/mac) |

## Partition Time Limits

| Partition | Max time | Notes |
|-----------|----------|-------|
| ray-cluster | 90 days | Ideal for long-running jobs (use for 8B) |
| capstone | 8 hours | Chain jobs with --dependency=afterany |
| student-gpu | 2 hours | Chain jobs with --dependency=afterany (use for 14B) |

## SLURM Job Chains

```bash
# 27B on capstone (8h limit) — chain for continuous coverage
J1=$(sbatch scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J3=$(sbatch --dependency=afterany:$J2 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')

# 14B on student-gpu (2h limit) — chain more links for longer coverage
J1=$(sbatch scripts/serve_vllm_14b_kolmogorov.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_14b_kolmogorov.sh | awk '{print $4}')
```

## vLLM Environment

- venv: `/users/achidamb/projects/gepa-mutations/vllm-venv/`
- Models: `/models/huggingface/` (NFS shared)
- IPC path bug: vLLM v1 IPC sockets fail if CWD path >107 chars. Fix: `cd /tmp` before launching.
- HF_HOME must be set explicitly: `HF_HOME=/models/huggingface`

## Accessing Nodes

Nodes are accessible from the login node via hostname (DNS resolves within the cluster).
SSH: `ssh <nodename>` — GPU jobs can't be inspected via srun if the GPU is held.
Use curl to vLLM `/metrics` endpoint to check live request counts instead.

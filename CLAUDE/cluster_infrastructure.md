# Cluster Infrastructure

## Active Nodes (as of 2026-04-11)

| Model | Node | Partition | GPU | VRAM | IP:Port | SLURM time limit |
|-------|------|-----------|-----|------|---------|-----------------|
| Qwen3-27B-AWQ | manifold | capstone | RTX 5090 | 32GB | 10.0.10.69:8124 | 8h (chain jobs) |
| Qwen3-8B | archimedes | ray-cluster | RTX 5090 | 32GB | 10.0.10.58:8125 | 90 days |
| Qwen3-4B | sapphire | capstone | RTX 4090 | 24GB | 10.0.100.99:8126 | 8h (chain jobs) |
| Qwen3-1.7B | sar-gpu-vm | capstone | RTX 3090 | 24GB | 10.0.50.99:8127 | 8h (chain jobs) |
| Qwen3-14B (planned) | kolmogorov | student-gpu | RTX 4090 | 24GB | 10.0.10.52:8128 | 2h (chain jobs) |
| Qwen3-4B (2nd) | mandelbrot | student-gpu | RTX 4060 Ti | 16GB | 10.0.10.53:8129 | 2h (chain jobs) |
| Qwen3-1.7B (2nd) | gho-gpu-vm | student-gpu | RTX 4060 Ti | 16GB | 10.0.50.69:8130 | 2h (chain jobs) |

## Downed Nodes (since ~2026-04-10 15:23)

| Node | Original role | GPU | Status |
|------|--------------|-----|--------|
| bourbaki | 27B primary | RTX 5090 | down* / Not responding |
| ansatz | 4B primary | RTX 4090 | down* / Not responding |
| deepseek | 1.7B primary | RTX 3090 | down* / Not responding |

Recovery is monitored by `scripts/check_node_recovery.sh` (system crontab, every 15 min).
Sends Telegram alert on recovery. Do NOT auto-relaunch — discuss with user first.

## Idle Nodes (available if needed)

| Node | Partition | GPU | VRAM | IP |
|------|-----------|-----|------|----|
| kolmogorov | student-gpu | RTX 4090 | 24GB | 10.0.10.52 |
| mandelbrot | student-gpu | RTX 4060 Ti | 16GB | 10.0.10.53 |
| gho-gpu-vm | student-gpu | RTX 4060 Ti | 16GB | 10.0.50.69 |

## Partition Time Limits

| Partition | Max time | Notes |
|-----------|----------|-------|
| ray-cluster | 90 days | Ideal for long-running jobs |
| capstone | 8 hours | Chain jobs with --dependency=afterany |
| student-gpu | 2 hours | Chain jobs with --dependency=afterany |

## vLLM Environment

- venv: `/home/achidamb/projects/gepa-mutations/vllm-venv/`
- Models: `/models/huggingface/` (NFS shared)
- IPC path bug: vLLM v1 IPC sockets fail if CWD path >107 chars. Fix: `cd /tmp` before launching.
- HF_HOME must be set explicitly: `HF_HOME=/models/huggingface`

## SLURM Job Chains (capstone, 8h limit)

Submit first job, then chain with `--dependency=afterany:<JOBID>`. Example:
```bash
J1=$(sbatch scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J3=$(sbatch --dependency=afterany:$J2 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
```

## Accessing Nodes

Nodes are accessible from the login node via hostname (DNS resolves within the cluster).
SSH: `ssh <nodename>` — but note GPU jobs can't be inspected via srun if the GPU is held.
Use curl to vLLM `/metrics` endpoint to check live request counts instead.

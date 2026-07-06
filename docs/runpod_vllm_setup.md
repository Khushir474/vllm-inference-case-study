# Self-hosting vLLM on RunPod

Switched from AWS after `g6.xlarge` vCPU quota requests were denied twice —
new AWS accounts start with a G/VT instance quota of 0, and getting that
raised through Support isn't guaranteed or fast. RunPod has no equivalent
approval gate: GPUs are available to rent the moment your account has funds
on it, and it's also cheaper for this GPU class.

## GPU / Pod

| | |
|---|---|
| **GPU** | NVIDIA L4, 24GB VRAM — same headroom reasoning as the AWS doc: Llama-3-8B-Instruct in bf16 needs ~18GB including KV cache, so 24GB is comfortable |
| **Price** | ~$0.39/hr on Community Cloud (shared, cheaper), ~10-30% more on Secure Cloud (dedicated data centers, better SLA). Billed **per second**, starting when the pod runs and stopping the instant you terminate it — no forgotten-instance risk like a stopped-but-still-billing EBS volume |
| **Pod type** | **GPU Pod** (persistent instance you SSH into), not Serverless. Serverless auto-scales to zero (even cheaper for idle time) but abstracts away the exact `vllm serve` flags we need for consistent latency measurement (`--enforce-eager`, `--max-model-len`, `--gpu-memory-utilization`) and adds cold-start latency to the first request — that would muddy the latency numbers we're trying to measure cleanly. A Pod behaves like a normal box, matching the setup this study already assumes. |
| **Template** | Any CUDA-preinstalled template (search "PyTorch" in RunPod's template list) — we install vLLM ourselves regardless of what's preloaded |
| **Disk** | 30-50GB container disk (model weights ~16GB bf16 + vLLM install). Don't bother with a persistent network volume for a one-off benchmark — that adds cost for storage that outlives the pod, which you don't need here. |

## Setup steps

1. **Sign up at [runpod.io](https://www.runpod.io/)** and add funds (pay-as-you-go, no quota/approval process).
2. **Deploy → Pods → GPU Pod** → select **L4** → Community Cloud (cheaper) unless you specifically want Secure Cloud's SLA → pick a PyTorch-based template → set container disk to 30-50GB.
3. **Expose port 8000**: in the pod's network/port settings, add `8000` to the exposed TCP ports before deploying, so `vllm serve` is reachable from outside once it's running.
4. **Connect**: once the pod is running, RunPod's console gives you an SSH command (or a web terminal) — use either.
5. **Install vLLM and get model access**:
   ```bash
   pip install vllm
   huggingface-cli login   # HF token with read access to meta-llama/Meta-Llama-3-8B-Instruct
   ```
6. **Start the server**:
   ```bash
   tmux new -s vllm
   vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
     --host 0.0.0.0 \
     --port 8000 \
     --gpu-memory-utilization 0.9 \
     --max-model-len 8192 \
     --enforce-eager
   # detach with Ctrl-b d
   ```
   Same flags and reasoning as the original AWS plan: `--enforce-eager` avoids
   CUDA-graph warmup skewing early-request latency; `--max-model-len 8192` is
   Llama-3-8B-Instruct's native context window (a few of the longer
   transcripts in this dataset, up to ~21K characters, get close to it).
7. **Get the public endpoint**: the pod's "Connect" panel shows the mapped
   public IP/port for the exposed 8000 — put that in `.env`:
   ```
   VLLM_BASE_URL=http://<runpod-public-ip>:<mapped-port>/v1
   VLLM_GPU_HOURLY_COST_USD=0.39
   ```
8. **Terminate the pod when done** — RunPod bills per second while it's
   running, so there's no "stop vs terminate" storage gotcha to worry about
   the way there is on AWS, as long as you didn't attach a persistent volume.

## Verifying it's up

```bash
curl http://<runpod-public-ip>:<mapped-port>/v1/models
```
should return the served model name. Then run:
```bash
python -m app.benchmark.run_benchmark --backend vllm --n 10
```

## Staying on budget

Even cheaper than the AWS plan: ~$0.39/hr on Community Cloud vs. `g6.xlarge`'s
~$0.805/hr, and per-second billing means no minimum-hour charge. A
10-transcript pilot (driver already working via the template, model download,
short run, terminate) should cost well under $1.

- Community Cloud availability can be less predictable than Secure Cloud
  (it's a shared host pool) — if a pod won't schedule, try Secure Cloud or a
  different region.
- Still terminate promptly when done; per-second billing helps but doesn't
  eliminate the "left it running overnight by accident" risk.

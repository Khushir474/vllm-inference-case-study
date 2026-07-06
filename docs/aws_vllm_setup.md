# Self-hosting vLLM on AWS

## Instance

| | |
|---|---|
| **Instance type** | `g6.xlarge` — 1x NVIDIA L4 (24GB VRAM), 4 vCPU, 16GB RAM |
| **On-demand price** | ~$0.805/hr (us-east-1, check [AWS pricing](https://aws.amazon.com/ec2/pricing/on-demand/) for current rate) |
| **Storage** | 100GB gp3 EBS (model weights are ~16GB in bf16; leave room for the vLLM/CUDA install and logs) |
| **AMI** | "Deep Learning AMI GPU PyTorch" (Ubuntu 22.04) from the AWS Marketplace — comes with NVIDIA drivers, CUDA, and Docker preinstalled, so you skip driver setup |

Why `g6.xlarge` over `g5.xlarge`: both have 24GB VRAM, comfortably enough for
Llama-3-8B-Instruct in bf16 (~18GB including KV cache headroom), but the
G6's L4 GPU is newer and ~20% cheaper on-demand than the G5's A10G. Only step
up to a bigger/multi-GPU instance if you raise `--max-model-len` far beyond
default or need higher concurrency than a single L4 handles.

If cost matters more than fidelity, an AWQ/INT4-quantized build of the same
model needs as little as ~5GB VRAM and could run on a cheaper `g5g` or
`g4dn.xlarge` — but that changes what's being measured (quantized vs
full-precision), so stick with bf16 on `g6.xlarge` for the baseline comparison.

## Setup steps

1. **Launch the instance**: EC2 console → Launch instance → AMI: "Deep
   Learning AMI GPU PyTorch (Ubuntu 22.04)" → type `g6.xlarge` → 100GB gp3
   root volume.
2. **Security group**: allow SSH (22) and a custom TCP rule for port 8000,
   both restricted to *your* IP only — don't open 8000 to `0.0.0.0/0`, that's
   an unauthenticated LLM endpoint on the open internet.
3. **SSH in and install vLLM**:
   ```bash
   ssh -i your-key.pem ubuntu@<instance-public-ip>
   pip install vllm
   ```
4. **Get Hugging Face access to Llama-3-8B-Instruct**: accept Meta's
   license on the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct),
   then on the instance:
   ```bash
   huggingface-cli login   # paste an HF token with read access
   ```
5. **Start the server** (keep it running after you disconnect with `tmux` or `nohup`):
   ```bash
   tmux new -s vllm
   vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
     --host 0.0.0.0 \
     --port 8000
   # detach with Ctrl-b d
   ```
6. **Point this repo at it** — in `.env`:
   ```
   VLLM_BASE_URL=http://<instance-public-ip>:8000/v1
   VLLM_GPU_HOURLY_COST_USD=0.805
   ```
7. **Stop or terminate the instance when you're done** — GPU instances bill
   hourly whether or not you're actively benchmarking against them.

## Verifying it's up

```bash
curl http://<instance-public-ip>:8000/v1/models
```
should return the served model name. Then run:
```bash
python -m app.benchmark.run_benchmark --backend vllm --n 10
```

## Staying on budget

The GPU instance is the only real cost risk in this study — Together AI and
Anthropic judge costs are a few dollars even at a thousand-transcript scale
(published per-token rates keep both under ~$10 for that volume). A `g6.xlarge`
pilot of 10 transcripts should cost under $3 total if you follow this:

- **Set a billing alarm the moment you launch** (AWS Budgets or a CloudWatch
  billing alarm, e.g. trigger at $5). The single biggest risk is forgetting
  to terminate the instance — idle billing is ~$19/day.
- **Batch your GPU sessions.** Launch once, run the pilot and any follow-up
  scaled-up runs back-to-back in the same session, then terminate — don't
  pay for a fresh driver/model-download cycle every time you want to test
  something.
- **Consider a Spot instance** for `g6.xlarge` (often 50–70% off on-demand).
  Fine here since a benchmark run is short and easy to redo if interrupted.
- **Terminate, don't just stop**, once you're fully done — a stopped
  instance doesn't bill compute, but the attached EBS volume still bills
  storage (minor, but avoidable) until the instance is terminated.

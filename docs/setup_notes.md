# Setup Complexity Log

Filled in from this study's actual setup experience (see `docs/setup_notes_template.md`
for the blank version).

## Backend: hosted (Together AI)

- Date: 2026-07-06
- Provider / instance type: Together AI serverless, `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
- Time from starting signup to first successful API call: well under an hour
- Manual config steps: create Together account, generate an API key, set
  `HOSTED_API_KEY` in `.env`. No model download, no server flags, no GPU
  provisioning — the endpoint is live the moment the key works.
- Errors or surprises: the originally-planned model
  (`Llama-3.1-8B-Instruct-Turbo`) turned out to be dedicated-endpoint-only —
  Together's `/v1/models` catalog listed it, but chat completions rejected it
  with "Unable to access non-serverless model." Confirmed via a live API call,
  not just reading docs — the catalog listing a model doesn't mean it's
  actually servable pay-per-token. Switched to `Meta-Llama-3-8B-Instruct-Lite`
  (Together's own quantized serverless variant) to keep the study apples-to-apples
  on the Llama-3 8B family.
- External docs needed: Together's pricing page (to confirm per-token cost)
  and their model catalog page.
- Overall friction: 2/5 — the model-availability surprise was the only real
  snag; otherwise this is close to zero-setup.

## Backend: vllm (self-hosted on RunPod)

- Date: 2026-07-06 to 2026-07-07
- Provider / instance type: RunPod GPU Pod, 1x NVIDIA L4 (24GB), Community
  Cloud, ~$0.39/hr, per-second billing
- Time from starting signup to first successful API call: several hours,
  spread across multiple false starts (see below) — not a quick "spin up and
  go" experience compared to the hosted side.
- Manual config steps: RunPod signup + funding, deploy a GPU Pod with port
  8000 exposed, generate and register an SSH key (none existed locally, had
  to `ssh-keygen` from scratch and add the public key to RunPod's account
  settings), request access to the gated `meta-llama/Meta-Llama-3-8B-Instruct`
  repo on Hugging Face (manual review, not instant — a real source of
  unpredictable delay), edit the pod's Container Start Command to replace
  the template's default model, set `HF_TOKEN` as a pod environment variable,
  discover and use the auto-generated `VLLM_API_KEY` as a Bearer token,
  update `.env` with the RunPod public proxy URL.
- Errors and surprises (the real cost of self-hosting vs. hosted):
  - AWS was the original plan; abandoned after `g6.xlarge` GPU vCPU quota
    requests were denied twice (new AWS accounts start at 0 quota for GPU
    instance families — not a fast or guaranteed approval path).
  - RunPod's SSH proxy requires PTY allocation (`ssh -tt`) — a plain
    non-interactive SSH command fails with "Your SSH client doesn't support
    PTY," which isn't obvious from the error message.
  - This pod's template bakes `vllm serve Qwen/Qwen3-8B` in as the
    container's own PID 1 / start command, not a background convenience
    process. Killing it (even by exact PID, avoiding PID 1 directly) still
    recreated the whole container each time, because this minimal init
    process exits — and the container restarts — the moment its supervised
    child process dies. Two 8B bf16 models can't coexist on one 24GB GPU
    anyway, so the real fix was editing the pod's Container Start Command in
    the RunPod console to serve the target model directly.
  - The console's Container Start Command field expects **arguments only**
    (no `vllm serve` prefix) — supplying the full command produced
    `vllm serve vllm serve meta-llama/...`, which crashed the container
    entirely and required a second edit to fix.
  - `hf auth login` run interactively over SSH did not survive the container
    being recreated by a console edit (state lived under `/workspace`, which
    didn't persist across that particular kind of restart) — had to switch
    to setting `HF_TOKEN` as a pod-level environment variable instead, which
    does survive restarts.
  - The Llama-3-8B-Instruct gate on Hugging Face required manual approval
    from the repo owners (`gated: 'manual'`) rather than the instant
    auto-approval typical of most Llama-3 releases — an unpredictable wait
    that hosted APIs simply don't expose you to.
  - The template auto-sets `VLLM_API_KEY`; the OpenAI-compatible server
    401s without it passed as a Bearer token — not documented anywhere
    obvious, found by inspecting the container's environment directly.
- External docs needed: RunPod's own docs were insufficient for the
  Container Start Command / env var behavior above — mostly figured out by
  direct SSH investigation (`ps aux`, `/proc/1/cmdline`, `env`) rather than
  documentation.
- Overall friction: 5/5 — every one of the above was a real blocker that
  stopped progress until resolved, several requiring a wrong guess first
  (e.g. the two failed pkill attempts that each restarted the container).
  This is the actual cost of self-hosting: not the GPU dollar rate, but the
  operational surface area — gated model approval queues, container
  lifecycle quirks specific to a given pod template, and SSH/auth plumbing
  that a hosted API abstracts away entirely.

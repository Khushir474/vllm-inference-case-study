"""Minimal non-vLLM baseline server: plain `transformers.generate()`, no
continuous batching, no PagedAttention. A global lock serializes every
request onto a single generate() call at a time — this is what "no batching"
actually looks like in a naive deployment, and is the fair baseline against
which to measure vLLM's real advantage on identical hardware.

Exposes just enough of the OpenAI chat completions schema for the existing
benchmark client (app/clients/openai_compatible.py) to work unmodified.
"""

import json
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

app = FastAPI()
_lock = threading.Lock()

print(f"Loading {MODEL_NAME} in bf16 (naive transformers baseline)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
model.eval()
print("Model loaded.")


class Message(BaseModel):
    role: str
    content: str | list[dict]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content if part.get("type") == "text")


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.text()} for m in req.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_tokens = inputs["input_ids"].shape[1]

    max_new_tokens = req.max_completion_tokens or req.max_tokens or 512

    # The lock is the whole point: this is what "no continuous batching" means
    # in practice — one generate() call occupies the GPU at a time, and every
    # concurrent request queues behind it, exactly like a naive single-worker
    # deployment would.
    with _lock:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    completion_tokens = output.shape[1] - prompt_tokens
    text = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)
    completion_id = f"naive-{time.time_ns()}"

    if req.stream:
        # generate() blocks until the whole response is ready — there is no
        # token-by-token streaming in a naive deployment, so this emits the
        # full text as a single SSE chunk. That means TTFT below honestly
        # equals total generation time, which is itself the real cost of not
        # having continuous batching / streaming decode.
        def sse():
            delta_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(delta_chunk)}\n\n"
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": req.model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

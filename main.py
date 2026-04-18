import json
import sys

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

app = FastAPI()

# Initialize vLLM engine
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    dtype="half",  # fp16
)

# Get tokenizer from vLLM
tokenizer = llm.get_tokenizer()

print("vLLM model loaded:", MODEL_NAME)

class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    print("Python version:", sys.version)

    messages = [
        {
            "role": "system",
            "content": (
                "Return only valid Python Polars code. "
                "No markdown fences. "
                "Assign the final Polars DataFrame to result. "
                f"Available datasets: {json.dumps(payload.tables, ensure_ascii=False)}"
            ),
        },
        {
            "role": "user",
            "content": payload.message,
        },
    ]

    # Apply chat template (same as before)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Sampling config (deterministic)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=["```"],  # safety
    )

    # Generate with vLLM
    outputs = llm.generate([prompt], sampling_params)

    response = outputs[0].outputs[0].text

    return ChatResponse(response=strip_code_fence(response))
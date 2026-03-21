import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_ENABLE_THINKING", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    log(f"[1/6] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"[1/6] gpu={props.name} vram_gb={props.total_memory / 1024 / 1024 / 1024:.2f}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    log(f"[2/6] loading tokenizer: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log(f"[2/6] tokenizer loaded in {time.time() - t0:.1f}s")

    log(f"[3/6] loading model (4-bit): {model_name}")
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quant_config,
    )
    model.eval()
    log(f"[3/6] model loaded in {time.time() - t1:.1f}s")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        log(f"[4/6] cuda allocated={used:.2f}GB reserved={reserved:.2f}GB")

    messages = [
        {"role": "system", "content": "You are a concise assistant. Answer in one short line."},
        {"role": "user", "content": "Only say ok."},
    ]
    log("[5/6] building chat template with enable_thinking=False")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    log(f"[5/6] prompt_tokens={inputs['input_ids'].shape[1]}")

    log("[6/6] generating")
    t2 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    log(f"[6/6] generated in {time.time() - t2:.1f}s")
    log(f"OUTPUT: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

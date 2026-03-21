import os
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ENABLE_THINKING = False

_TOKENIZER = None
_MODEL = None
_LOADED_MODEL_NAME = None


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_enable_thinking() -> bool:
    return _str_to_bool(os.getenv("HF_ENABLE_THINKING", str(DEFAULT_ENABLE_THINKING)))


def _build_quant_config():
    if not _str_to_bool(os.getenv("HF_LOAD_IN_4BIT", "true")):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _load_model(model_name: str):
    global _TOKENIZER, _MODEL, _LOADED_MODEL_NAME
    if _MODEL is not None and _LOADED_MODEL_NAME == model_name:
        return _TOKENIZER, _MODEL

    quantization_config = _build_quant_config()
    print(f"Loading local model: {model_name}")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    _MODEL.eval()
    _LOADED_MODEL_NAME = model_name
    return _TOKENIZER, _MODEL


def chat_completion(
    messages: Sequence[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
    retries: int = 1,
) -> str:
    del retries
    tokenizer, local_model = _load_model(model)
    enable_thinking = get_enable_thinking()

    prompt = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(local_model.device)

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.8,
                "top_k": 20,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = local_model.generate(**inputs, **generation_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


def extract_label_block(text: str) -> str:
    start = text.find("[label]")
    end = text.find("[/label]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + len("[/label]")]

    alt_start = text.find("<<label>>")
    alt_end = text.find("/label>>")
    if alt_start != -1 and alt_end != -1 and alt_end > alt_start:
        return text[alt_start : alt_end + len("/label>>")]

    return text.strip()


def normalize_yes_no(text: str) -> str:
    lowered = text.strip().lower()
    if lowered.startswith("yes"):
        return "Yes"
    if lowered.startswith("no"):
        return "No"
    if "yes" in lowered and "no" not in lowered:
        return "Yes"
    if "no" in lowered and "yes" not in lowered:
        return "No"
    return text.strip()

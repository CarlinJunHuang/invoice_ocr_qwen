from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from invoice_ocr_glm_paddle_eval.config import AppConfig, ModeConfig
from invoice_ocr_glm_paddle_eval.schema import SCHEMA_GUIDE
from invoice_ocr_glm_paddle_eval.text_parser import split_text_lines

SYSTEM_PROMPT = """You extract invoice or credit note fields conservatively.
Return JSON only.
Use only the supplied OCR text.
If a field is uncertain, return null or an empty array.
Keep evidence text short and verbatim from the OCR text when possible.
Do not explain your reasoning.
Do not include markdown fences.
Start your answer with '{' and end it with '}'."""

_TOKENIZER_CACHE: dict[str, Any] = {}
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _build_clean_ocr_text(raw_text: str) -> str:
    return "\n".join(split_text_lines(raw_text))


def _build_user_prompt(raw_text: str) -> str:
    ocr_text = _build_clean_ocr_text(raw_text)
    return (
        "Extract the invoice or credit note into the target JSON schema.\n"
        "Requirements:\n"
        "- doc_type must be INVOICE or CREDIT_NOTE\n"
        "- language should be en unless the OCR text clearly uses another language\n"
        "- use null or [] when unknown\n"
        "- evidence entries must include only text and page\n"
        "- line_items may be empty if not confidently extracted\n"
        "- do not invent extra keys beyond the schema\n"
        "- output exactly one JSON object, with no analysis before or after\n\n"
        f"OCR TEXT\n{ocr_text}\n\n"
        "TARGET JSON SHAPE\n"
        f"{SCHEMA_GUIDE}"
    )


def _iter_json_candidates(section: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"```json\s*(\{.*?\})\s*```", section, flags=re.DOTALL):
        candidates.append(match.group(1).strip())

    start = section.find("{")
    if start == -1:
        return candidates

    end = section.rfind("}")
    if end != -1 and end > start:
        candidates.append(section[start : end + 1].strip())
    candidates.append(section[start:].strip())
    return candidates


def load_json_payload(text: str) -> dict[str, Any]:
    from json_repair import repair_json

    last_error: Exception | None = None
    seen: set[str] = set()
    for candidate in _iter_json_candidates(text.strip()):
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                repaired = repair_json(candidate)
                return json.loads(repaired)
            except Exception as repair_exc:
                last_error = repair_exc

    if last_error is not None:
        raise ValueError(f"Qwen parser output did not contain a parseable JSON object: {last_error}") from last_error
    raise ValueError("Qwen parser output did not contain a JSON object.")


def _load_tokenizer_and_model(mode: ModeConfig, config: AppConfig):
    model_id = str(mode.options["parser_model_id"])
    precision = str(mode.options.get("parser_precision", "bf16"))
    device = str(mode.options.get("parser_device", mode.options.get("device", "cuda")))
    cache_key = (model_id, precision, device)

    if cache_key in _MODEL_CACHE and model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id], _MODEL_CACHE[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = _TOKENIZER_CACHE.get(model_id)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(config.hf_home),
            trust_remote_code=True,
        )
        _TOKENIZER_CACHE[model_id] = tokenizer

    use_cuda = device == "cuda" and torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {
        "cache_dir": str(config.hf_home),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    if precision == "4bit":
        if not use_cuda:
            raise RuntimeError("4-bit Qwen parser mode requires a CUDA device.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    _MODEL_CACHE[cache_key] = model
    return tokenizer, model


def _apply_chat_template(tokenizer: Any, mode: ModeConfig, messages: list[dict[str, str]]) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    chat_template_kwargs = mode.options.get("parser_chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict):
        kwargs.update(chat_template_kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)


def _generate_text(tokenizer: Any, model: Any, prompt_text: str, mode: ModeConfig) -> tuple[str, dict[str, int | float]]:
    model_inputs = tokenizer([prompt_text], return_tensors="pt")
    target_device = next(model.parameters()).device
    for key, value in list(model_inputs.items()):
        if hasattr(value, "to"):
            model_inputs[key] = value.to(target_device)

    temperature = float(mode.options.get("parser_temperature", 0.0))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(mode.options.get("parser_max_new_tokens", 900)),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    started_at = perf_counter()
    generated_ids = model.generate(**model_inputs, **generation_kwargs)
    elapsed_seconds = round(perf_counter() - started_at, 3)
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    output_tokens = len(trimmed_ids[0]) if trimmed_ids else 0
    raw_output = tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
    metrics = {
        "input_tokens": int(model_inputs["input_ids"].shape[-1]),
        "output_tokens": int(output_tokens),
        "total_tokens": int(model_inputs["input_ids"].shape[-1]) + int(output_tokens),
        "elapsed_seconds": elapsed_seconds,
    }
    return raw_output, metrics


def run_qwen_parse(raw_ocr_text: str, mode: ModeConfig, config: AppConfig) -> tuple[str, dict[str, Any], dict[str, object]]:
    tokenizer, model = _load_tokenizer_and_model(mode, config)
    prompt = _build_user_prompt(raw_ocr_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt_text = _apply_chat_template(tokenizer, mode, messages)
    parser_output, metrics = _generate_text(tokenizer, model, prompt_text, mode)
    metrics.update(
        {
            "stage": "qwen_parser",
            "model_id": str(mode.options["parser_model_id"]),
            "ocr_text_lines": len(split_text_lines(raw_ocr_text)),
        }
    )
    return parser_output, load_json_payload(parser_output), metrics

from __future__ import annotations

from typing import Any

from invoice_ocr_qwen.extractors.base import ExtractionContext, ExtractionResult
from invoice_ocr_qwen.extractors.common import SCHEMA_GUIDE, build_ocr_text, load_json_payload
from invoice_ocr_qwen.models.schema import build_empty_envelope, normalize_envelope

SYSTEM_PROMPT = """You extract invoice data conservatively.
Return JSON only.
Use only the supplied OCR text.
If a field is uncertain, return null or an empty array.
Keep evidence text short and verbatim from the OCR text when possible.
Do not explain your reasoning.
Do not include markdown fences.
Start your answer with '{' and end it with '}'."""

_TOKENIZER_CACHE: dict[str, Any] = {}
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _build_user_prompt(context: ExtractionContext) -> str:
    ocr_text = build_ocr_text(context.ocr_pages)
    return (
        "Extract the invoice into the target JSON schema.\n"
        "Requirements:\n"
        "- doc_type must be INVOICE\n"
        "- language should be en unless the document clearly uses another language\n"
        "- use null or [] when unknown\n"
        "- evidence entries must include text and page only\n"
        "- clauses should only be added when supported by the provided OCR text\n"
        "- eligibility.supporting_clauses must list only detected clause types\n"
        "- output exactly one JSON object, with no analysis text before or after\n"
        "- line_items may use this compact shape to save tokens:\n"
        '  {"description":"...","quantity":1,"unit_price":10.0,"line_total":10.0,'
        '"confidence":0.9,"evidence":[{"text":"...","page":1}]}\n'
        "- do not invent extra keys beyond the schema or this compact line_items form\n\n"
        f"OCR TEXT\n{ocr_text}\n\n"
        "TARGET JSON SHAPE\n"
        f"{SCHEMA_GUIDE}"
    )


def _load_tokenizer_and_model(context: ExtractionContext):
    mode = context.mode
    model_id = str(mode.options["model_id"])
    precision = str(mode.options.get("precision", "bf16"))
    device = str(mode.options.get("device", "cuda"))
    cache_key = (model_id, precision, device)

    if cache_key in _MODEL_CACHE and model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id], _MODEL_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("transformers/torch dependencies are not installed. Run the bootstrap script first.") from exc

    tokenizer = _TOKENIZER_CACHE.get(model_id)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(context.config.hf_home),
            trust_remote_code=True,
        )
        _TOKENIZER_CACHE[model_id] = tokenizer

    use_cuda = device == "cuda" and torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {
        "cache_dir": str(context.config.hf_home),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    if precision == "4bit":
        if not use_cuda:
            raise RuntimeError("4-bit Qwen text mode requires a CUDA device.")
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


def _build_generation_kwargs(context: ExtractionContext) -> dict[str, Any]:
    temperature = float(context.mode.options.get("temperature", 0.1))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(context.mode.options.get("max_new_tokens", 900)),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    for key in ("top_p", "top_k", "min_p", "repetition_penalty"):
        if key in context.mode.options:
            generation_kwargs[key] = context.mode.options[key]
    return generation_kwargs


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], context: ExtractionContext) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    chat_template_kwargs = context.mode.options.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict):
        kwargs.update(chat_template_kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)


def _generate_text(tokenizer: Any, model: Any, prompt_text: str, generation_kwargs: dict[str, Any]) -> str:
    model_inputs = tokenizer([prompt_text], return_tensors="pt")
    target_device = next(model.parameters()).device
    for key, value in list(model_inputs.items()):
        if hasattr(value, "to"):
            model_inputs[key] = value.to(target_device)

    generated_ids = model.generate(**model_inputs, **generation_kwargs)
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    return tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]


def _retry_json_format(tokenizer: Any, model: Any, context: ExtractionContext, draft_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Convert the user's draft extraction notes into exactly one valid JSON object. "
                "Return JSON only. No explanation. No markdown fences. "
                "Start with '{' and end with '}'."
            ),
        },
        {
            "role": "user",
            "content": (
                "DRAFT EXTRACTION NOTES\n"
                f"{draft_text}\n\n"
                "TARGET JSON SHAPE\n"
                f"{SCHEMA_GUIDE}"
            ),
        },
    ]
    prompt_text = _apply_chat_template(tokenizer, messages, context)
    generation_kwargs = _build_generation_kwargs(context)
    generation_kwargs["max_new_tokens"] = min(int(generation_kwargs.get("max_new_tokens", 900)), 500)
    return _generate_text(tokenizer, model, prompt_text, generation_kwargs)


class QwenTextExtractor:
    def extract(self, context: ExtractionContext) -> ExtractionResult:
        tokenizer, model = _load_tokenizer_and_model(context)
        user_prompt = _build_user_prompt(context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = _apply_chat_template(tokenizer, messages, context)
        decoded = _generate_text(tokenizer, model, prompt_text, _build_generation_kwargs(context))

        try:
            parsed = load_json_payload(decoded)
            envelope = normalize_envelope(parsed, request_id=context.request_id)
        except Exception as exc:
            formatter_output = _retry_json_format(tokenizer, model, context, decoded)
            combined_output = decoded + "\n\n--- JSON_FORMATTER_PASS ---\n" + formatter_output
            try:
                parsed = load_json_payload(combined_output)
                envelope = normalize_envelope(parsed, request_id=context.request_id)
                return ExtractionResult(envelope=envelope, raw_output=combined_output, parsed_output=parsed)
            except Exception as formatter_exc:
                envelope = build_empty_envelope(request_id=context.request_id)
                envelope.errors.append(
                    "Failed to parse model JSON output: "
                    f"{exc}; formatter pass failed: {formatter_exc}"
                )
                return ExtractionResult(envelope=envelope, raw_output=combined_output)

        return ExtractionResult(envelope=envelope, raw_output=decoded, parsed_output=parsed)

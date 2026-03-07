from __future__ import annotations

from typing import Any

from invoice_ocr_qwen.extractors.base import ExtractionContext, ExtractionResult
from invoice_ocr_qwen.extractors.common import SCHEMA_GUIDE, build_ocr_text, load_json_payload
from invoice_ocr_qwen.models.schema import build_empty_envelope, normalize_envelope

SYSTEM_PROMPT = """You extract invoice data conservatively.
Return JSON only.
Use only the visible document content and supplied OCR text.
If a field is uncertain, return null or an empty array.
Keep evidence text short and verbatim from the document when possible.
Do not invent clause types or eligibility reasons unsupported by the document.
Do not explain your reasoning.
Do not include markdown fences.
Start your answer with '{' and end it with '}'."""

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_PROCESSOR_CACHE: dict[str, Any] = {}


def _build_user_prompt(context: ExtractionContext) -> str:
    ocr_text = build_ocr_text(context.ocr_pages)

    return (
        "Extract the invoice into the target JSON schema.\n"
        "Requirements:\n"
        "- doc_type must be INVOICE\n"
        "- language should be en unless the document clearly uses another language\n"
        "- use null or [] when unknown\n"
        "- evidence entries must include text and page only\n"
        "- clauses should only be added when supported by the provided pages\n"
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


def _model_device(model: Any):
    return next(model.parameters()).device


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]], context: ExtractionContext) -> str:
    chat_template_kwargs = context.mode.options.get("chat_template_kwargs")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if chat_template_kwargs is not None:
        kwargs["chat_template_kwargs"] = chat_template_kwargs
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("chat_template_kwargs", None)
        if isinstance(chat_template_kwargs, dict):
            kwargs.update(chat_template_kwargs)
        return processor.apply_chat_template(messages, **kwargs)


def _build_generation_kwargs(context: ExtractionContext) -> dict[str, Any]:
    temperature = float(context.mode.options.get("temperature", 0.0))
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


def _generate_text(processor: Any, model: Any, prompt_text: str, generation_kwargs: dict[str, Any]) -> str:
    model_inputs = processor(
        text=[prompt_text],
        padding=True,
        return_tensors="pt",
    )
    target_device = _model_device(model)
    for key, value in list(model_inputs.items()):
        if hasattr(value, "to"):
            model_inputs[key] = value.to(target_device)

    generated_ids = model.generate(**model_inputs, **generation_kwargs)
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _retry_json_format(processor: Any, model: Any, context: ExtractionContext, draft_text: str) -> str:
    formatter_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Convert the user's draft extraction notes into exactly one valid JSON object. "
                        "Return JSON only. No explanation. No markdown fences. "
                        "Start with '{' and end with '}'."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "DRAFT EXTRACTION NOTES\n"
                        f"{draft_text}\n\n"
                        "TARGET JSON SHAPE\n"
                        f"{SCHEMA_GUIDE}"
                    ),
                }
            ],
        },
    ]
    prompt_text = _apply_chat_template(processor, formatter_messages, context)
    generation_kwargs = _build_generation_kwargs(context)
    generation_kwargs["max_new_tokens"] = min(int(generation_kwargs.get("max_new_tokens", 900)), 500)
    return _generate_text(processor, model, prompt_text, generation_kwargs)


def _load_processor_and_model(context: ExtractionContext):
    mode = context.mode
    model_id = str(mode.options["model_id"])
    precision = str(mode.options.get("precision", "bf16"))
    device = str(mode.options.get("device", "cuda"))
    cache_key = (model_id, precision, device)

    if cache_key in _MODEL_CACHE and model_id in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[model_id], _MODEL_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("transformers/torch dependencies are not installed. Run the bootstrap script first.") from exc

    processor = _PROCESSOR_CACHE.get(model_id)
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(context.config.hf_home),
            trust_remote_code=True,
        )
        _PROCESSOR_CACHE[model_id] = processor

    use_cuda = device == "cuda" and torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {
        "cache_dir": str(context.config.hf_home),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    attn_implementation = mode.options.get("attn_implementation")
    if use_cuda and attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if precision == "4bit":
        if not use_cuda:
            raise RuntimeError("4-bit Qwen VL mode requires a CUDA device.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_cuda else torch.float32

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:  # pragma: no cover - runtime only
        if "flash" in str(exc).lower() and "attn_implementation" in model_kwargs:
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        else:
            raise

    _MODEL_CACHE[cache_key] = model
    return processor, model


class QwenVLExtractor:
    def extract(self, context: ExtractionContext) -> ExtractionResult:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError("qwen-vl-utils is not installed. Run the bootstrap script first.") from exc

        processor, model = _load_processor_and_model(context)
        user_prompt = _build_user_prompt(context)
        min_pixels = context.mode.options.get("min_pixels")
        max_pixels = context.mode.options.get("max_pixels")
        image_content = []
        for path in context.image_paths:
            image_item: dict[str, Any] = {"type": "image", "image": str(path)}
            if min_pixels is not None:
                image_item["min_pixels"] = int(min_pixels)
            if max_pixels is not None:
                image_item["max_pixels"] = int(max_pixels)
            image_content.append(image_item)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": user_prompt}],
            },
        ]

        prompt_text = _apply_chat_template(processor, messages, context)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        target_device = _model_device(model)
        for key, value in list(model_inputs.items()):
            if hasattr(value, "to"):
                model_inputs[key] = value.to(target_device)

        generation_kwargs = _build_generation_kwargs(context)
        generated_ids = model.generate(**model_inputs, **generation_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]
        decoded = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        try:
            parsed = load_json_payload(decoded)
            envelope = normalize_envelope(parsed, request_id=context.request_id)
        except Exception as exc:
            formatter_output = _retry_json_format(processor, model, context, decoded)
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

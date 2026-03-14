from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from invoice_ocr_glm_paddle_eval.config import AppConfig, ModeConfig
from invoice_ocr_glm_paddle_eval.text_parser import parse_invoice_like_text

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_PROCESSOR_CACHE: dict[str, Any] = {}

MODEL_TEXT_PROMPTS = {
    "glm_ocr": "Text Recognition:",
    "paddleocr_vl_v1": "OCR:",
    "paddleocr_vl_v1_5": "OCR:",
    "firered_ocr": """You are an AI assistant specialized in converting document images to Markdown format.
Accurately recognize all text content without guessing or inferring.
Maintain the original document structure, including headings, paragraphs, and lists.
Convert mathematical formulas to LaTeX when present.
Convert tables to HTML wrapped in <table> and </table>.
Ignore image descriptions.
Return Markdown only, with no explanations or extra comments.""",
}


def _model_device(model: Any):
    return next(model.parameters()).device


def _load_processor_and_model(mode: ModeConfig, config: AppConfig):
    model_id = str(mode.options["model_id"])
    kind = mode.kind
    cache_key = (model_id, kind)
    if cache_key in _MODEL_CACHE and model_id in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[model_id], _MODEL_CACHE[cache_key]

    import torch
    import transformers.cache_utils as cache_utils
    import transformers.modeling_rope_utils as rope_utils
    import transformers.utils.generic as generic_utils
    from transformers import AutoModelForImageTextToText, AutoProcessor, GlmOcrForConditionalGeneration
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        Qwen3VLForConditionalGeneration = None

    if not hasattr(cache_utils, "SlidingWindowCache"):
        class SlidingWindowCache(cache_utils.Cache):
            pass

        cache_utils.SlidingWindowCache = SlidingWindowCache

    if not hasattr(generic_utils, "check_model_inputs"):
        def check_model_inputs(function):
            return function

        generic_utils.check_model_inputs = check_model_inputs

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
            import torch

            if config is None:
                raise ValueError("config is required for PaddleOCR-VL default rope compatibility.")
            dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            base = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    processor = _PROCESSOR_CACHE.get(model_id)
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(config.hf_home),
            trust_remote_code=True,
        )
        _PROCESSOR_CACHE[model_id] = processor

    model_kwargs: dict[str, Any] = {
        "cache_dir": str(config.hf_home),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    if kind == "glm_ocr":
        model = GlmOcrForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    elif kind in {"paddleocr_vl_v1", "paddleocr_vl_v1_5"}:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    elif kind == "firered_ocr":
        if Qwen3VLForConditionalGeneration is not None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        else:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    else:
        raise ValueError(f"Unsupported mode kind: {kind}")

    _MODEL_CACHE[cache_key] = model
    return processor, model


def _prepare_inputs(processor: Any, image_path: Path, mode: ModeConfig):
    image = Image.open(image_path).convert("RGB")
    prompt_text = MODEL_TEXT_PROMPTS.get(mode.kind, "OCR:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if mode.kind == "paddleocr_vl_v1_5":
        kwargs["images_kwargs"] = {"size": {"longest_edge": 1280 * 28 * 28}}
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("images_kwargs", None)
        return processor.apply_chat_template(messages, **kwargs)


def run_model_extract(image_path: Path, mode: ModeConfig, config: AppConfig) -> tuple[str, dict[str, Any], dict[str, object]]:
    processor, model = _load_processor_and_model(mode, config)
    model_inputs = _prepare_inputs(processor, image_path, mode)
    target_device = _model_device(model)
    for key, value in list(model_inputs.items()):
        if hasattr(value, "to"):
            model_inputs[key] = value.to(target_device)

    if "token_type_ids" in model_inputs and mode.kind == "glm_ocr":
        model_inputs.pop("token_type_ids")

    temperature = float(mode.options.get("temperature", 0.0))
    generate_kwargs = {
        "max_new_tokens": int(mode.options.get("max_new_tokens", 1400)),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    started_at = perf_counter()
    generated = model.generate(**model_inputs, **generate_kwargs)
    elapsed_seconds = round(perf_counter() - started_at, 3)
    prompt_length = int(model_inputs["input_ids"].shape[-1])
    trimmed = generated[:, prompt_length:]
    output_tokens = int(trimmed.shape[-1]) if len(trimmed.shape) > 1 else 0
    raw_output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    metrics = {
        "stage": "ocr_model",
        "model_id": str(mode.options["model_id"]),
        "kind": mode.kind,
        "elapsed_seconds": elapsed_seconds,
        "input_tokens": prompt_length,
        "output_tokens": output_tokens,
        "total_tokens": prompt_length + output_tokens,
    }
    return raw_output, parse_invoice_like_text(raw_output), metrics

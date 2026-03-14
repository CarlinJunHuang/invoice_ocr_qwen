from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from invoice_ocr_glm_paddle_eval.config import AppConfig
from invoice_ocr_glm_paddle_eval.schema import OCRLine, OCRPage

_READER_CACHE: dict[tuple[tuple[str, ...], bool, str], object] = {}


def _get_easyocr_reader(config: AppConfig):
    try:
        import torch
    except ImportError:  # pragma: no cover - runtime only
        torch = None

    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("easyocr is not installed. Run the bootstrap script first.") from exc

    use_gpu = config.ocr_gpu and bool(torch and torch.cuda.is_available())
    model_storage = str(config.torch_home / "easyocr")
    cache_key = (tuple(config.ocr_languages), use_gpu, model_storage)
    if cache_key not in _READER_CACHE:
        _READER_CACHE[cache_key] = easyocr.Reader(
            config.ocr_languages,
            gpu=use_gpu,
            model_storage_directory=model_storage,
        )
    return _READER_CACHE[cache_key]


def run_ocr(image_paths: list[Path], config: AppConfig) -> tuple[list[OCRPage], dict[str, object]]:
    started_at = perf_counter()
    reader = _get_easyocr_reader(config)
    pages: list[OCRPage] = []

    for page_number, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
            image_rgb = image.convert("RGB")
            image_array = np.array(image_rgb)

        raw_results = reader.readtext(image_array, detail=1, paragraph=False)
        lines = []
        for bbox, text, confidence in raw_results:
            if confidence is not None and float(confidence) < config.ocr_min_confidence:
                continue
            clean_text = text.strip()
            if not clean_text:
                continue
            lines.append(
                OCRLine(
                    text=clean_text,
                    confidence=float(confidence) if confidence is not None else None,
                    bbox=[(float(x), float(y)) for x, y in bbox],
                )
            )

        pages.append(
            OCRPage(
                page_number=page_number,
                image_path=str(image_path),
                width=width,
                height=height,
                lines=lines,
            )
        )

    metrics = {
        "engine": "easyocr",
        "elapsed_seconds": round(perf_counter() - started_at, 3),
        "page_count": len(pages),
        "line_count": sum(len(page.lines) for page in pages),
    }
    return pages, metrics

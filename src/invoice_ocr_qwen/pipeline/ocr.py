from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from invoice_ocr_qwen.config import AppConfig
from invoice_ocr_qwen.models.schema import OCRLine, OCRPage

_READER_CACHE: dict[tuple[tuple[str, ...], bool, str], object] = {}


def _get_easyocr_reader(config: AppConfig):
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised in runtime
        torch = None

    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover - exercised in runtime
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


def run_ocr(image_paths: list[Path], config: AppConfig) -> list[OCRPage]:
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
            lines.append(
                OCRLine(
                    text=text.strip(),
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

    return pages

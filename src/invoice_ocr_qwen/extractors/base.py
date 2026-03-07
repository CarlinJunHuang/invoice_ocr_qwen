from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from invoice_ocr_qwen.config import AppConfig, ModeConfig
from invoice_ocr_qwen.models.schema import Envelope, OCRPage


@dataclass(slots=True)
class ExtractionContext:
    request_id: str
    image_paths: list[Path]
    ocr_pages: list[OCRPage]
    config: AppConfig
    mode: ModeConfig
    output_dir: Path


@dataclass(slots=True)
class ExtractionResult:
    envelope: Envelope
    raw_output: str | None = None
    parsed_output: dict[str, Any] | None = None


class Extractor(Protocol):
    def extract(self, context: ExtractionContext) -> ExtractionResult:
        ...

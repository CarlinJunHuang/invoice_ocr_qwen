from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from invoice_ocr_qwen.config import AppConfig, ModeConfig
from invoice_ocr_qwen.extractors.base import ExtractionContext, Extractor, ExtractionResult
from invoice_ocr_qwen.extractors.rule_based import RuleBasedExtractor
from invoice_ocr_qwen.models.schema import build_empty_envelope
from invoice_ocr_qwen.pipeline.evidence import build_normalized_document, ground_envelope_evidence
from invoice_ocr_qwen.pipeline.ocr import run_ocr
from invoice_ocr_qwen.pipeline.overlay import render_overlay_images
from invoice_ocr_qwen.pipeline.runtime import ResourceMonitor, apply_runtime_environment


def _create_extractor(mode: ModeConfig) -> Extractor:
    if mode.kind == "rules":
        return RuleBasedExtractor()
    if mode.kind == "qwen_vl":
        from invoice_ocr_qwen.extractors.qwen_vl import QwenVLExtractor

        return QwenVLExtractor()
    if mode.kind == "qwen_text":
        from invoice_ocr_qwen.extractors.qwen_text import QwenTextExtractor

        return QwenTextExtractor()
    raise ValueError(f"Unsupported mode kind: {mode.kind}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_extraction(
    image_paths: list[Path],
    mode_name: str,
    config: AppConfig,
    request_id: str | None = None,
    run_name: str | None = None,
) -> dict[str, object]:
    if not image_paths:
        raise ValueError("At least one input image is required.")

    image_paths = [Path(path).resolve() for path in image_paths]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

    apply_runtime_environment(config)

    request_id = request_id or str(uuid4())
    run_name = run_name or f"extract-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    mode = config.require_mode(mode_name)
    output_dir = config.output_root / run_name / mode_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with ResourceMonitor(config.resource_poll_seconds) as monitor:
        ocr_pages = run_ocr(image_paths=image_paths, config=config)
        extractor = _create_extractor(mode)
        context = ExtractionContext(
            request_id=request_id,
            image_paths=image_paths,
            ocr_pages=ocr_pages,
            config=config,
            mode=mode,
            output_dir=output_dir,
        )
        try:
            result = extractor.extract(context)
        except Exception as exc:
            envelope = build_empty_envelope(request_id=request_id)
            envelope.errors.append(str(exc))
            result = ExtractionResult(envelope=envelope, raw_output=str(exc))

    metrics = monitor.snapshot()
    grounded = ground_envelope_evidence(
        envelope=result.envelope,
        ocr_pages=ocr_pages,
        threshold=config.fuzzy_threshold,
    )
    overlay_paths = []
    if config.overlay_enabled:
        overlay_paths = render_overlay_images(
            image_paths=image_paths,
            grounded_evidence=grounded,
            output_dir=output_dir,
            line_width=config.overlay_line_width,
            font_size=config.overlay_font_size,
        )

    normalized_document = build_normalized_document(ocr_pages)
    artifacts = {
        "ocr_pages": str(output_dir / "ocr_pages.json"),
        "normalized_document": str(output_dir / "normalized_document.json"),
        "envelope": str(output_dir / "envelope.json"),
        "grounded_evidence": str(output_dir / "grounded_evidence.json"),
        "overlay_images": overlay_paths,
    }

    _write_json(output_dir / "ocr_pages.json", [page.model_dump(mode="json") for page in ocr_pages])
    _write_json(output_dir / "normalized_document.json", normalized_document)
    _write_json(output_dir / "envelope.json", result.envelope.model_dump(mode="json"))
    _write_json(output_dir / "grounded_evidence.json", [item.model_dump(mode="json") for item in grounded])

    if result.parsed_output is not None:
        parsed_path = output_dir / "parsed_model_output.json"
        _write_json(parsed_path, result.parsed_output)
        artifacts["parsed_model_output"] = str(parsed_path)
    if result.raw_output is not None:
        raw_path = output_dir / "raw_model_output.txt"
        raw_path.write_text(result.raw_output, encoding="utf-8")
        artifacts["raw_model_output"] = str(raw_path)

    summary = {
        "request_id": request_id,
        "mode": mode_name,
        "mode_kind": mode.kind,
        "run_name": run_name,
        "images": [str(path) for path in image_paths],
        "metrics": {
            "wall_seconds": metrics.wall_seconds,
            "peak_rss_mb": metrics.peak_rss_mb,
            "peak_cuda_mb": metrics.peak_cuda_mb,
        },
        "artifacts": artifacts,
    }
    _write_json(output_dir / "run_summary.json", summary)
    summary["artifacts"]["run_summary"] = str(output_dir / "run_summary.json")
    return summary

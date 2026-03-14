from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from invoice_ocr_glm_paddle_eval.config import AppConfig
from invoice_ocr_glm_paddle_eval.grounding import ground_envelope_evidence
from invoice_ocr_glm_paddle_eval.models import run_model_extract
from invoice_ocr_glm_paddle_eval.ocr import run_ocr
from invoice_ocr_glm_paddle_eval.overlay import render_overlay_images
from invoice_ocr_glm_paddle_eval.qwen_parser import run_qwen_parse
from invoice_ocr_glm_paddle_eval.reporting import build_compare_report
from invoice_ocr_glm_paddle_eval.schema import EvidenceItem, ExtractedField, InvoiceEnvelope, build_empty_envelope, normalize_envelope


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _apply_runtime_environment(config: AppConfig) -> None:
    os.environ["HF_HOME"] = str(config.hf_home)
    os.environ["HF_HUB_CACHE"] = str(config.hf_home)
    os.environ["HF_MODULES_CACHE"] = str(config.hf_home / "modules")
    os.environ["TRANSFORMERS_CACHE"] = str(config.hf_home)
    os.environ["TORCH_HOME"] = str(config.torch_home)
    os.environ["XDG_CACHE_HOME"] = str(config.runtime_root)
    tmp_root = config.runtime_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)


def _ensure_field_evidence(field: ExtractedField) -> None:
    if field.value in (None, "") or field.evidence:
        return
    field.evidence = [EvidenceItem(text=str(field.value), page=1)]


def _backfill_missing_evidence(envelope: InvoiceEnvelope) -> None:
    extracted = envelope.extracted
    for field in (
        extracted.seller_name,
        extracted.buyer_name,
        extracted.invoice_number,
        extracted.invoice_date,
        extracted.due_date,
        extracted.payment_terms,
        extracted.currency,
        extracted.bank_details,
        extracted.totals.gross_amount,
        extracted.totals.tax_amount,
        extracted.totals.net_amount,
    ):
        _ensure_field_evidence(field)

    for item in extracted.line_items:
        for field in (item.description, item.quantity, item.unit_price, item.line_total):
            _ensure_field_evidence(field)


def run_extract(
    image_path: Path,
    mode_name: str,
    config: AppConfig,
    run_name: str | None = None,
    request_id: str | None = None,
) -> dict[str, object]:
    pipeline_started_at = perf_counter()
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    _apply_runtime_environment(config)
    request_id = request_id or str(uuid4())
    run_name = run_name or f"extract-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    mode = config.require_mode(mode_name)
    output_dir = config.output_root / run_name / image_path.stem / mode_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_pages, ocr_metrics = run_ocr([image_path], config)
    parsed_output = None
    raw_output = None
    parser_output = None
    model_metrics = None
    parser_metrics = None
    envelope: InvoiceEnvelope
    try:
        raw_output, parsed_output, model_metrics = run_model_extract(image_path, mode, config)
        if mode.options.get("parser_model_id"):
            parser_output, parsed_output, parser_metrics = run_qwen_parse(raw_output, mode, config)
        envelope = normalize_envelope(parsed_output, request_id=request_id)
        _backfill_missing_evidence(envelope)
    except Exception as exc:
        envelope = build_empty_envelope(request_id=request_id)
        envelope.errors.append(str(exc))
        raw_output = str(exc)

    grounded = ground_envelope_evidence(envelope, ocr_pages, config.fuzzy_threshold)
    overlay_paths = []
    if config.overlay_enabled:
        overlay_paths = render_overlay_images(
            image_paths=[image_path],
            grounded_evidence=grounded,
            output_dir=output_dir,
            line_width=config.overlay_line_width,
            font_size=config.overlay_font_size,
        )

    _write_json(output_dir / "ocr_pages.json", [page.model_dump(mode="json") for page in ocr_pages])
    _write_json(output_dir / "invoice_fields.json", envelope.model_dump(mode="json"))
    _write_json(output_dir / "grounded_boxes.json", [item.model_dump(mode="json") for item in grounded])
    if parsed_output is not None:
        _write_json(output_dir / "parsed_model_output.json", parsed_output)
    if raw_output is not None:
        (output_dir / "raw_model_output.txt").write_text(raw_output, encoding="utf-8")
    if parser_output is not None:
        (output_dir / "parser_output.txt").write_text(parser_output, encoding="utf-8")

    summary = {
        "request_id": request_id,
        "mode": mode_name,
        "run_name": run_name,
        "image": str(image_path),
        "metrics": {
            "ocr": ocr_metrics,
            "main_model": model_metrics,
            "parser_model": parser_metrics,
            "total_elapsed_seconds": round(perf_counter() - pipeline_started_at, 3),
        },
        "artifacts": {
            "ocr_pages": str(output_dir / "ocr_pages.json"),
            "invoice_fields": str(output_dir / "invoice_fields.json"),
            "grounded_boxes": str(output_dir / "grounded_boxes.json"),
            "overlay_images": overlay_paths,
            "parsed_model_output": str(output_dir / "parsed_model_output.json") if parsed_output is not None else None,
            "raw_model_output": str(output_dir / "raw_model_output.txt") if raw_output is not None else None,
            "parser_output": str(output_dir / "parser_output.txt") if parser_output is not None else None,
        },
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def run_compare(
    image_paths: list[Path],
    mode_names: list[str],
    config: AppConfig,
    run_name: str | None = None,
) -> dict[str, object]:
    run_name = run_name or f"compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results = []
    for image_path in image_paths:
        for mode_name in mode_names:
            results.append(run_extract(Path(image_path), mode_name, config, run_name=run_name))

    summary = {
        "run_name": run_name,
        "modes": mode_names,
        "images": [str(Path(path).resolve()) for path in image_paths],
        "results": results,
    }
    report_artifacts = build_compare_report(run_name=run_name, results=results, report_root=config.report_root)
    summary["report_artifacts"] = report_artifacts
    output_dir = config.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "benchmark_summary.json", summary)
    return summary

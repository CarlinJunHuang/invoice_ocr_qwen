from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from json_repair import repair_json
from PIL import Image, ImageDraw, ImageFont

from invoice_ocr_qwen.direct_bench.page_io import (
    RenderedPage,
    build_page_dimension_metadata,
    load_document_pages,
)


MODULE_ROOT = Path(__file__).resolve().parent
PROMPTS_ROOT = MODULE_ROOT / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_ROOT / "poc_invoice_contract_v3.txt"
OUTPUT_ROOT = Path(os.getenv("INVOICE_OCR_QWEN_DIRECT_OUTPUT_ROOT", "output/direct"))
SCALAR_TARGET_FIELDS = [
    "seller_name",
    "buyer_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "gross_amount",
    "net_amount",
    "total_amount",
    "tax_amount",
    "outstanding_balance",
    "currency",
    "payment_terms",
    "account_name",
    "bank_name",
    "account_number",
    "swift_code",
    "bank_branch_code",
]
LINE_ITEM_FIELDS = [
    "description",
    "quantity",
    "unit_price",
    "line_total",
]
SCALAR_AMOUNT_FIELDS = {
    "gross_amount",
    "net_amount",
    "total_amount",
    "tax_amount",
    "outstanding_balance",
}
LINE_ITEM_AMOUNT_FIELDS = {
    "unit_price",
    "line_total",
}
PAYMENT_TERM_TOKEN_RE = re.compile(
    r"(?i)\b("
    r"cash|cod|c\.?o\.?d\.?|upon receipt|net\s*\d+|"
    r"payment within \d+\s*days?|"
    r"\d+\s*days?(?:\s+from\s+invoice\s+date)?|"
    r"due on receipt"
    r")\b"
)
PAYMENT_TERM_REJECT_TOKEN_RE = re.compile(
    r"(?i)\b("
    r"localdel|delivery|period|ship|shipment|service period|"
    r"gst|tax invoice|invoice date|bank|swift|account"
    r")\b"
)
VISIBLE_AMOUNT_RE = re.compile(r"-?\d[\d,]*(?:\.\d{1,2})?")
VISIBLE_QUANTITY_RE = re.compile(r"-?\d+(?:\.\d+)?")

DEFAULT_PROMPT = """Extract invoice fields directly from this page image.
Return JSON only. Do not use markdown fences.
Return exactly one JSON object with these keys and no others:
{
  "seller_name": {"value": null, "bbox_2d": null},
  "buyer_name": {"value": null, "bbox_2d": null},
  "invoice_number": {"value": null, "bbox_2d": null},
  "invoice_date": {"value": null, "bbox_2d": null},
  "total_amount": {"value": null, "bbox_2d": null},
  "tax_amount": {"value": null, "bbox_2d": null},
  "currency": {"value": null, "bbox_2d": null},
  "payment_terms": {"value": null, "bbox_2d": null},
  "account_name": {"value": null, "bbox_2d": null},
  "bank_name": {"value": null, "bbox_2d": null},
  "account_number": {"value": null, "bbox_2d": null},
  "swift_code": {"value": null, "bbox_2d": null}
}
Rules:
- bbox_2d must be a single axis-aligned bounding box as [x1, y1, x2, y2]
- bbox_2d must use integer coordinates normalized to 0-1000 relative to this page image
- if a field is missing or uncertain, use {"value": null, "bbox_2d": null}
- keep values close to the visible text
- choose the box that best supports the chosen value
- do not add explanations or extra keys"""


def load_prompt_text(prompt_file: str | Path | None) -> str:
    prompt_path = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    if not prompt_path.is_absolute():
        prompt_path = prompt_path.resolve()
    return prompt_path.read_text(encoding="utf-8").strip()

PALETTE = [
    "#E85D04",
    "#0A9396",
    "#BB3E03",
    "#005F73",
    "#9B2226",
    "#CA6702",
    "#3A86FF",
    "#6A994E",
]


@dataclass(slots=True)
class RunConfig:
    backend: str
    model: str
    input_path: Path
    output_dir: Path
    prompt_text: str
    base_url: str
    api_key: str | None
    temperature: float
    max_tokens: int
    timeout_sec: float
    dpi: int
    max_pages: int
    max_long_side: int
    max_pixels: int
    jpeg_quality: int
    allow_thinking_fallback: bool


def _font(font_size: int) -> ImageFont.ImageFont:
    for font_name in ("Menlo.ttc", "Arial Unicode.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _page_to_data_url(page: RenderedPage, quality: int) -> str:
    image = Image.fromarray(page.image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=max(20, min(100, quality)))
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json_candidates(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    candidates: list[str] = []
    if stripped.startswith("```"):
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                candidates.append(part)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1].strip())
    candidates.append(stripped)
    return candidates


def _load_json_payload(text: str) -> Any:
    last_error: Exception | None = None
    seen: set[str] = set()
    for candidate in _extract_json_candidates(text):
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_json(candidate))
            except Exception as exc:
                last_error = exc
    if last_error is not None:
        raise ValueError(f"Model output did not contain parseable JSON: {last_error}") from last_error
    raise ValueError("Model output did not contain JSON.")


def _coerce_bbox(bbox_value: Any) -> tuple[int, int, int, int] | None:
    if bbox_value is None:
        return None
    while isinstance(bbox_value, list | tuple) and len(bbox_value) == 1 and isinstance(bbox_value[0], list | tuple):
        bbox_value = bbox_value[0]
    if isinstance(bbox_value, dict):
        try:
            if all(key in bbox_value for key in ("x1", "y1", "x2", "y2")):
                x1 = int(round(float(bbox_value.get("x1"))))
                y1 = int(round(float(bbox_value.get("y1"))))
                x2 = int(round(float(bbox_value.get("x2"))))
                y2 = int(round(float(bbox_value.get("y2"))))
            elif all(key in bbox_value for key in ("left", "top", "right", "bottom")):
                x1 = int(round(float(bbox_value.get("left"))))
                y1 = int(round(float(bbox_value.get("top"))))
                x2 = int(round(float(bbox_value.get("right"))))
                y2 = int(round(float(bbox_value.get("bottom"))))
            elif all(key in bbox_value for key in ("x", "y", "width", "height")):
                x1 = int(round(float(bbox_value.get("x"))))
                y1 = int(round(float(bbox_value.get("y"))))
                x2 = x1 + int(round(float(bbox_value.get("width"))))
                y2 = y1 + int(round(float(bbox_value.get("height"))))
            else:
                return None
            return x1, y1, x2, y2
        except (TypeError, ValueError):
            return None
    if (
        isinstance(bbox_value, list | tuple)
        and len(bbox_value) == 2
        and all(isinstance(item, list | tuple) and len(item) == 2 for item in bbox_value)
    ):
        try:
            (x1, y1), (x2, y2) = bbox_value
            return int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
        except (TypeError, ValueError):
            return None
    if not isinstance(bbox_value, list | tuple) or len(bbox_value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in bbox_value]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def _coerce_quad(location_value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(location_value, list | tuple):
        return None
    if len(location_value) == 8:
        try:
            flat = [float(item) for item in location_value]
        except (TypeError, ValueError):
            return None
        return [(flat[index], flat[index + 1]) for index in range(0, 8, 2)]
    if len(location_value) == 4 and all(isinstance(item, list | tuple) and len(item) == 2 for item in location_value):
        points: list[tuple[float, float]] = []
        try:
            for point in location_value:
                points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None
        return points
    return None


def _rotate_rect_to_bbox(rotate_rect_value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(rotate_rect_value, list | tuple) or len(rotate_rect_value) != 5:
        return None
    try:
        cx, cy, width, height, angle = [float(item) for item in rotate_rect_value]
    except (TypeError, ValueError):
        return None
    half_width = width / 2.0
    half_height = height / 2.0
    radians = math.radians(angle)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ]
    points = []
    for offset_x, offset_y in corners:
        x = cx + offset_x * cos_a - offset_y * sin_a
        y = cy + offset_x * sin_a + offset_y * cos_a
        points.append((x, y))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))


def _bbox_to_pixels(bbox_value: Any, page: RenderedPage) -> dict[str, int] | None:
    bbox = _coerce_bbox(bbox_value)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000:
            x1 = int(round(max(0, min(1000, x1)) / 1000 * page.rendered_width))
            y1 = int(round(max(0, min(1000, y1)) / 1000 * page.rendered_height))
            x2 = int(round(max(0, min(1000, x2)) / 1000 * page.rendered_width))
            y2 = int(round(max(0, min(1000, y2)) / 1000 * page.rendered_height))
        x1, x2 = sorted((max(0, min(page.rendered_width, x1)), max(0, min(page.rendered_width, x2))))
        y1, y2 = sorted((max(0, min(page.rendered_height, y1)), max(0, min(page.rendered_height, y2))))
        return {"x": x1, "y": y1, "width": max(1, x2 - x1), "height": max(1, y2 - y1)}

    quad = _coerce_quad(bbox_value)
    if quad is not None:
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        return {
            "x": int(round(max(0, min(page.rendered_width, min(xs))))),
            "y": int(round(max(0, min(page.rendered_height, min(ys))))),
            "width": max(1, int(round(max(xs) - min(xs)))),
            "height": max(1, int(round(max(ys) - min(ys)))),
        }

    bbox = _rotate_rect_to_bbox(bbox_value)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted((max(0, min(page.rendered_width, x1)), max(0, min(page.rendered_width, x2))))
        y1, y2 = sorted((max(0, min(page.rendered_height, y1)), max(0, min(page.rendered_height, y2))))
        return {"x": x1, "y": y1, "width": max(1, x2 - x1), "height": max(1, y2 - y1)}

    return None


def _normalize_value_box_payload(raw_field: Any) -> dict[str, Any]:
    if isinstance(raw_field, dict):
        value = raw_field.get("value")
        bbox_value = raw_field.get("bbox_2d")
        if bbox_value is None:
            bbox_value = raw_field.get("bbox") or raw_field.get("box") or raw_field.get("location") or raw_field.get("rotate_rect")
    elif raw_field is None:
        value = None
        bbox_value = None
    else:
        value = raw_field
        bbox_value = None
    return {
        "value": None if value in (None, "", "null") else str(value),
        "bbox_raw": bbox_value,
    }


def _normalize_amount_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    matches = VISIBLE_AMOUNT_RE.findall(text.replace(" ", ""))
    if len(matches) != 1:
        return None
    amount = matches[0]
    if not re.fullmatch(r"-?\d[\d,]*(?:\.\d{1,2})?", amount):
        return None
    if "." in amount:
        whole, fraction = amount.split(".", 1)
        amount = f"{whole}.{fraction[:2].ljust(2, '0')}"
    return amount


def _normalize_quantity_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    matches = VISIBLE_QUANTITY_RE.findall(text)
    if len(matches) != 1:
        return None
    return matches[0]


def _normalize_currency_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = re.sub(r"\s+", " ", text.upper())
    if upper in {"SGD", "S$", "SINGAPORE DOLLAR", "SINGAPORE DOLLARS"}:
        return "SGD"
    if upper in {"USD", "US$", "EUR", "GBP", "MYR", "RM", "HKD", "HK$", "$"}:
        return upper
    if len(upper) <= 4 and upper.isalpha():
        return upper
    return None


def _normalize_payment_terms_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if not PAYMENT_TERM_TOKEN_RE.search(text):
        return None
    if PAYMENT_TERM_REJECT_TOKEN_RE.search(text) and not re.search(r"(?i)\b\d+\s*days?\b", text):
        return None
    return text


def _normalize_scalar_field_value(field_name: str, value: Any) -> str | None:
    if value is None:
        return None
    if field_name in SCALAR_AMOUNT_FIELDS:
        return _normalize_amount_text(value)
    if field_name == "currency":
        return _normalize_currency_text(value)
    if field_name == "payment_terms":
        return _normalize_payment_terms_text(value)
    return str(value).strip() or None


def _normalize_line_item_field_value(field_name: str, value: Any) -> str | None:
    if value is None:
        return None
    if field_name in LINE_ITEM_AMOUNT_FIELDS:
        return _normalize_amount_text(value)
    if field_name == "quantity":
        return _normalize_quantity_text(value)
    return str(value).strip() or None


def _same_bbox(a: Any, b: Any) -> bool:
    bbox_a = _coerce_bbox(a)
    bbox_b = _coerce_bbox(b)
    return bbox_a is not None and bbox_a == bbox_b


def _apply_scalar_consistency_rules(fields: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    due_date = fields.get("due_date")
    invoice_date = fields.get("invoice_date")
    payment_terms = fields.get("payment_terms")
    if (
        due_date
        and invoice_date
        and due_date.get("value")
        and invoice_date.get("value")
        and str(due_date["value"]).strip() == str(invoice_date["value"]).strip()
        and _same_bbox(due_date.get("bbox_raw"), invoice_date.get("bbox_raw"))
    ):
        due_date["value"] = None
        due_date["bbox_raw"] = None
    elif (
        due_date
        and invoice_date
        and payment_terms
        and due_date.get("value")
        and invoice_date.get("value")
        and str(due_date["value"]).strip() == str(invoice_date["value"]).strip()
        and re.search(r"(?i)\b\d+\s*days?\s+from\s+invoice\s+date\b", str(payment_terms.get("value") or ""))
    ):
        due_date["value"] = None
        due_date["bbox_raw"] = None

    account_number = fields.get("account_number")
    if account_number and account_number.get("value"):
        account_digits = re.sub(r"\W+", "", str(account_number["value"]))
        if len(account_digits) < 6:
            account_number["value"] = None
            account_number["bbox_raw"] = None

    bank_branch_code = fields.get("bank_branch_code")
    if bank_branch_code and bank_branch_code.get("value"):
        branch_code = re.sub(r"\W+", "", str(bank_branch_code["value"]))
        account_digits = re.sub(r"\W+", "", str((account_number or {}).get("value") or ""))
        if not (3 <= len(branch_code) <= 8) or (account_digits and branch_code == account_digits):
            bank_branch_code["value"] = None
            bank_branch_code["bbox_raw"] = None
    return fields


def _line_item_bbox_px(line_item: dict[str, Any]) -> dict[str, int] | None:
    row_bbox = line_item.get("row_bbox_px")
    if row_bbox:
        return row_bbox
    x_values: list[int] = []
    y_values: list[int] = []
    x2_values: list[int] = []
    y2_values: list[int] = []
    for field_data in line_item.get("fields", {}).values():
        bbox = field_data.get("bbox_px")
        if not bbox:
            continue
        x_values.append(int(bbox["x"]))
        y_values.append(int(bbox["y"]))
        x2_values.append(int(bbox["x"] + bbox["width"]))
        y2_values.append(int(bbox["y"] + bbox["height"]))
    if not x_values:
        return None
    return {
        "x": min(x_values),
        "y": min(y_values),
        "width": max(1, max(x2_values) - min(x_values)),
        "height": max(1, max(y2_values) - min(y_values)),
    }


def _select_extracted_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    extracted = payload.get("extracted")
    if isinstance(extracted, dict):
        payload = extracted
    if isinstance(payload.get("fields"), dict):
        payload = payload["fields"]
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object keyed by field name.")
    return payload


def _normalize_scalar_field_payload(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, list):
        mapped: dict[str, dict[str, Any]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            field_name = str(item.get("field") or item.get("name") or "").strip()
            if field_name:
                mapped[field_name] = item
        payload = mapped
    payload = _select_extracted_payload(payload)

    bank_details = payload.get("bank_details")
    if isinstance(bank_details, dict):
        for field_name in ("account_name", "bank_name", "account_number", "swift_code", "bank_branch_code"):
            if field_name not in payload and field_name in bank_details:
                payload[field_name] = bank_details.get(field_name)

    normalized: dict[str, dict[str, Any]] = {}
    for field_name in SCALAR_TARGET_FIELDS:
        field_data = _normalize_value_box_payload(payload.get(field_name))
        field_data["value"] = _normalize_scalar_field_value(field_name, field_data.get("value"))
        normalized[field_name] = field_data
    return _apply_scalar_consistency_rules(normalized)


def _normalize_line_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return []
    try:
        payload = _select_extracted_payload(payload)
    except ValueError:
        return []
    raw_items = payload.get("line_items")
    if not isinstance(raw_items, list):
        return []

    normalized_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            continue
        row_bbox_raw = (
            raw_item.get("row_bbox_2d")
            or raw_item.get("row_bbox")
            or raw_item.get("bbox_2d")
            or raw_item.get("bbox")
            or raw_item.get("box")
            or raw_item.get("location")
            or raw_item.get("rotate_rect")
        )
        normalized_fields: dict[str, dict[str, Any]] = {}
        for field_name in LINE_ITEM_FIELDS:
            field_data = _normalize_value_box_payload(raw_item.get(field_name))
            field_data["value"] = _normalize_line_item_field_value(field_name, field_data.get("value"))
            normalized_fields[field_name] = field_data
        normalized_items.append(
            {
                "index": index,
                "row_bbox_raw": row_bbox_raw,
                "fields": normalized_fields,
            }
        )
    return normalized_items


def _extract_doc_type(payload: Any) -> str:
    if isinstance(payload, dict):
        doc_type = payload.get("doc_type")
        if isinstance(doc_type, str) and doc_type.strip():
            return doc_type.strip().upper()
    return "INVOICE"


def _field_envelope_entry(field_data: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"evidence": []}
    if field_data.get("value") is not None:
        entry["value"] = field_data["value"]
    return entry


def _build_envelope(
    *,
    doc_type: str,
    config: RunConfig,
    summary: dict[str, Any],
    merged_fields: dict[str, dict[str, Any]],
    merged_line_items: list[dict[str, Any]],
    page_errors: dict[str, str],
    page_model_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    warnings: list[str] = []
    if any(bool(metadata.get("used_thinking_fallback")) for metadata in page_model_metadata.values()):
        warnings.append("At least one page used thinking fallback because Ollama returned empty content.")
    warnings.append("This direct bench does not emit evidence objects. Use parsed_output.json for bbox_2d debug data.")

    extracted = {
        "seller_name": _field_envelope_entry(merged_fields["seller_name"]),
        "buyer_name": _field_envelope_entry(merged_fields["buyer_name"]),
        "invoice_number": _field_envelope_entry(merged_fields["invoice_number"]),
        "invoice_date": _field_envelope_entry(merged_fields["invoice_date"]),
        "due_date": _field_envelope_entry(merged_fields["due_date"]),
        "payment_terms": _field_envelope_entry(merged_fields["payment_terms"]),
        "currency": _field_envelope_entry(merged_fields["currency"]),
        "gross_amount": _field_envelope_entry(merged_fields["gross_amount"]),
        "tax_amount": _field_envelope_entry(merged_fields["tax_amount"]),
        "total_amount": _field_envelope_entry(merged_fields["total_amount"]),
        "net_amount": _field_envelope_entry(merged_fields["net_amount"]),
        "outstanding_balance": _field_envelope_entry(merged_fields["outstanding_balance"]),
        "line_items": [],
        "bank_details": {
            "account_name": _field_envelope_entry(merged_fields["account_name"]),
            "bank_name": _field_envelope_entry(merged_fields["bank_name"]),
            "account_number": _field_envelope_entry(merged_fields["account_number"]),
            "swift_code": _field_envelope_entry(merged_fields["swift_code"]),
            "bank_branch_code": _field_envelope_entry(merged_fields["bank_branch_code"]),
        },
    }

    for line_item in merged_line_items:
        extracted["line_items"].append(
            {
                "description": _field_envelope_entry(line_item["fields"]["description"]),
                "quantity": _field_envelope_entry(line_item["fields"]["quantity"]),
                "unit_price": _field_envelope_entry(line_item["fields"]["unit_price"]),
                "line_total": _field_envelope_entry(line_item["fields"]["line_total"]),
            }
        )

    return {
        "request_id": str(uuid.uuid4()),
        "schema_version": "0.1-direct-vl",
        "doc_type": doc_type,
        "extracted": extracted,
        "clauses": [],
        "eligibility": {
            "status": "UNKNOWN",
            "reason": "Eligibility is not evaluated in the direct VL bench.",
            "supporting_clauses": [],
        },
        "warnings": warnings,
        "errors": [
            {
                "code": "page_parse_error",
                "message": f"page {page}: {message}",
            }
            for page, message in sorted(page_errors.items(), key=lambda item: int(item[0]))
        ],
    }


def _call_ollama(page: RenderedPage, config: RunConfig) -> tuple[str, dict[str, Any]]:
    image_data_url = _page_to_data_url(page, config.jpeg_quality)
    payload = {
        "model": config.model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [{"role": "user", "content": config.prompt_text, "images": [image_data_url.split(",", 1)[1]]}],
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }
    response = requests.post(f"{config.base_url.rstrip('/')}/api/chat", json=payload, timeout=config.timeout_sec)
    response.raise_for_status()
    data = response.json()
    message = data.get("message", {}) if isinstance(data, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    thinking = message.get("thinking", "") if isinstance(message, dict) else ""
    content_text = str(content or "").strip()
    thinking_text = str(thinking or "").strip()
    used_thinking_fallback = bool(config.allow_thinking_fallback and not content_text and thinking_text)
    metadata = {
        "backend": "ollama",
        "model": config.model,
        "base_url": config.base_url,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
        "done_reason": data.get("done_reason"),
        "created_at": data.get("created_at"),
        "content_char_count": len(content_text),
        "thinking_char_count": len(thinking_text),
        "used_thinking_fallback": used_thinking_fallback,
    }
    raw_output = thinking_text if used_thinking_fallback else content_text
    return raw_output, metadata


def _call_openai_compatible(page: RenderedPage, config: RunConfig) -> tuple[str, dict[str, Any]]:
    image_data_url = _page_to_data_url(page, config.jpeg_quality)
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": config.prompt_text},
                ],
            }
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    response = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=config.timeout_sec,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        content = "\n".join(part for part in text_parts if part)
    metadata = {
        "backend": "openai-compatible",
        "model": config.model,
        "base_url": config.base_url,
        "usage": data.get("usage"),
        "id": data.get("id"),
        "created": data.get("created"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
    }
    return str(content or "").strip(), metadata


def _load_pages(config: RunConfig) -> list[RenderedPage]:
    file_bytes = config.input_path.read_bytes()
    return load_document_pages(
        file_bytes=file_bytes,
        filename=config.input_path.name,
        content_type=None,
        dpi=config.dpi,
        max_pages=config.max_pages,
        max_long_side=config.max_long_side,
        max_pixels=config.max_pixels,
    )


def _render_overlays(
    *,
    pages: list[RenderedPage],
    fields: dict[str, dict[str, Any]],
    output_dir: Path,
    font_size: int = 18,
    line_width: int = 4,
) -> list[str]:
    font = _font(font_size)
    output_paths: list[str] = []
    by_page: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for field_name, field_data in fields.items():
        if not field_data.get("bbox_px") or not field_data.get("page"):
            continue
        by_page.setdefault(int(field_data["page"]), []).append((field_name, field_data))

    for index, page in enumerate(pages, start=1):
        matches = by_page.get(index, [])
        if not matches:
            continue
        image = Image.fromarray(page.image).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item_index, (field_name, field_data) in enumerate(matches):
            bbox = field_data["bbox_px"]
            color = PALETTE[item_index % len(PALETTE)]
            x0 = bbox["x"]
            y0 = bbox["y"]
            x1 = x0 + bbox["width"]
            y1 = y0 + bbox["height"]
            draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
            label = field_name
            text_box = draw.textbbox((x0, y0), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((x0, y0), label, fill="white", font=font)
        output_path = output_dir / f"page_{index:02d}_overlay.png"
        image.save(output_path)
        output_paths.append(str(output_path))
    return output_paths


def _render_line_item_overlays(
    *,
    pages: list[RenderedPage],
    line_items: list[dict[str, Any]],
    output_dir: Path,
    font_size: int = 18,
    line_width: int = 4,
) -> list[str]:
    font = _font(font_size)
    output_paths: list[str] = []
    by_page: dict[int, list[dict[str, Any]]] = {}
    for line_item in line_items:
        page = line_item.get("page")
        if page is None:
            continue
        bbox = _line_item_bbox_px(line_item)
        if not bbox:
            continue
        by_page.setdefault(int(page), []).append({"index": line_item["index"], "bbox": bbox})

    for index, page in enumerate(pages, start=1):
        matches = by_page.get(index, [])
        if not matches:
            continue
        image = Image.fromarray(page.image).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item_index, match in enumerate(matches):
            bbox = match["bbox"]
            color = PALETTE[item_index % len(PALETTE)]
            x0 = bbox["x"]
            y0 = bbox["y"]
            x1 = x0 + bbox["width"]
            y1 = y0 + bbox["height"]
            draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
            label = f"line_item_{match['index']:02d}"
            text_box = draw.textbbox((x0, y0), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((x0, y0), label, fill="white", font=font)
        output_path = output_dir / f"page_{index:02d}_line_items_overlay.png"
        image.save(output_path)
        output_paths.append(str(output_path))
    return output_paths


def run(config: RunConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pages = _load_pages(config)
    page_raw_outputs: dict[str, str] = {}
    page_model_metadata: dict[str, dict[str, Any]] = {}
    per_page_fields: dict[str, dict[str, dict[str, Any]]] = {}
    per_page_line_items: dict[str, list[dict[str, Any]]] = {}
    page_errors: dict[str, str] = {}
    merged_fields = {
        field_name: {"value": None, "page": None, "bbox_px": None, "bbox_raw": None}
        for field_name in SCALAR_TARGET_FIELDS
    }
    merged_line_items: list[dict[str, Any]] = []
    doc_type = "INVOICE"
    started_at = time.time()

    for page in pages:
        page_started_at = time.time()
        if config.backend == "ollama":
            raw_output, metadata = _call_ollama(page, config)
        elif config.backend == "openai-compatible":
            raw_output, metadata = _call_openai_compatible(page, config)
        else:
            raise RuntimeError(f"Unsupported backend: {config.backend}")
        metadata["elapsed_ms"] = int(round((time.time() - page_started_at) * 1000))
        page_key = str(page.page_index)
        page_raw_outputs[page_key] = raw_output
        page_model_metadata[page_key] = metadata
        try:
            payload = _load_json_payload(raw_output)
            doc_type = _extract_doc_type(payload)
            normalized = _normalize_scalar_field_payload(payload)
            for field_name, field_data in normalized.items():
                bbox_px = _bbox_to_pixels(field_data.get("bbox_raw"), page)
                normalized[field_name] = {
                    "value": field_data.get("value"),
                    "bbox_raw": field_data.get("bbox_raw"),
                    "bbox_px": bbox_px,
                    "page": page.page_index if field_data.get("value") else None,
                }
                if merged_fields[field_name]["value"] is None and field_data.get("value") is not None:
                    merged_fields[field_name] = {
                        "value": field_data.get("value"),
                        "page": page.page_index,
                        "bbox_px": bbox_px,
                        "bbox_raw": field_data.get("bbox_raw"),
                    }
            per_page_fields[page_key] = normalized
            normalized_line_items = _normalize_line_items(payload)
            for line_item in normalized_line_items:
                row_bbox_px = _bbox_to_pixels(line_item.get("row_bbox_raw"), page)
                normalized_line_item = {
                    "index": line_item["index"],
                    "page": page.page_index,
                    "row_bbox_raw": line_item.get("row_bbox_raw"),
                    "row_bbox_px": row_bbox_px,
                    "fields": {},
                }
                for field_name, field_data in line_item["fields"].items():
                    normalized_line_item["fields"][field_name] = {
                        "value": field_data.get("value"),
                        "bbox_raw": field_data.get("bbox_raw"),
                        "bbox_px": _bbox_to_pixels(field_data.get("bbox_raw"), page),
                    }
                merged_line_items.append(normalized_line_item)
            per_page_line_items[page_key] = normalized_line_items
        except Exception as exc:
            page_errors[page_key] = str(exc)
            per_page_fields[page_key] = {
                field_name: {"value": None, "bbox_raw": None, "bbox_px": None, "page": None}
                for field_name in SCALAR_TARGET_FIELDS
            }
            per_page_line_items[page_key] = []

    overlay_paths = _render_overlays(pages=pages, fields=merged_fields, output_dir=config.output_dir)
    line_item_overlay_paths = _render_line_item_overlays(
        pages=pages,
        line_items=merged_line_items,
        output_dir=config.output_dir,
    )
    elapsed_ms = int(round((time.time() - started_at) * 1000))

    parsed_output = {
        "backend": config.backend,
        "model": config.model,
        "input": str(config.input_path),
        "prompt": config.prompt_text,
        "doc_type": doc_type,
        "page_count": len(pages),
        "fields": merged_fields,
        "per_page_fields": per_page_fields,
        "line_items": merged_line_items,
        "per_page_line_items": per_page_line_items,
        "page_errors": page_errors,
        "page_dimensions": build_page_dimension_metadata(pages),
    }
    summary = {
        "backend": config.backend,
        "model": config.model,
        "input": str(config.input_path),
        "doc_type": doc_type,
        "page_count": len(pages),
        "elapsed_ms": elapsed_ms,
        "fields_found": sum(1 for item in merged_fields.values() if item["value"] is not None),
        "line_item_count": len(merged_line_items),
        "error_count": len(page_errors),
        "artifacts": {
            "prompt": str(config.output_dir / "prompt.txt"),
            "page_raw_outputs": str(config.output_dir / "page_raw_outputs.json"),
            "page_model_metadata": str(config.output_dir / "page_model_metadata.json"),
            "parsed_output": str(config.output_dir / "parsed_output.json"),
            "envelope": str(config.output_dir / "envelope.json"),
            "overlay_images": overlay_paths,
            "line_item_overlay_images": line_item_overlay_paths,
            "run_summary": str(config.output_dir / "run_summary.json"),
        },
    }
    envelope = _build_envelope(
        doc_type=doc_type,
        config=config,
        summary=summary,
        merged_fields=merged_fields,
        merged_line_items=merged_line_items,
        page_errors=page_errors,
        page_model_metadata=page_model_metadata,
    )

    (config.output_dir / "prompt.txt").write_text(config.prompt_text, encoding="utf-8")
    (config.output_dir / "page_raw_outputs.json").write_text(
        json.dumps({"page_raw_outputs": page_raw_outputs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "page_model_metadata.json").write_text(
        json.dumps({"page_model_metadata": page_model_metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "parsed_output.json").write_text(
        json.dumps(parsed_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "envelope.json").write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        return os.getenv(args.api_key_env)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-model direct invoice VLM bench.")
    parser.add_argument("--backend", choices=("ollama", "openai-compatible"), default="ollama")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--max-long-side", type=int, default=2200)
    parser.add_argument("--max-pixels", type=int, default=6000000)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--allow-thinking-fallback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    prompt_text = load_prompt_text(args.prompt_file)

    if args.backend == "ollama":
        base_url = args.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        api_key = None
    else:
        base_url = args.base_url or os.getenv("QWEN_API_BASE_URL", "").strip()
        api_key = _resolve_api_key(args)
        if not base_url:
            parser.error("QWEN_API_BASE_URL or --base-url is required for openai-compatible backend.")
        if not api_key:
            parser.error("QWEN_API_KEY or --api-key is required for openai-compatible backend.")

    config = RunConfig(
        backend=args.backend,
        model=args.model,
        input_path=input_path,
        output_dir=Path(args.output_root).resolve() / args.run_name,
        prompt_text=prompt_text,
        base_url=base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        dpi=args.dpi,
        max_pages=args.max_pages,
        max_long_side=args.max_long_side,
        max_pixels=args.max_pixels,
        jpeg_quality=args.jpeg_quality,
        allow_thinking_fallback=args.allow_thinking_fallback,
    )
    summary = run(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

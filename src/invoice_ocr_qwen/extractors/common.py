from __future__ import annotations

import json
import re
from typing import Any

from invoice_ocr_qwen.models.schema import OCRPage

SCHEMA_GUIDE = """{
  "request_id": "REQUEST_ID",
  "doc_type": "INVOICE",
  "language": "en",
  "extracted": {
    "seller_name": {"value": null, "confidence": null, "evidence": []},
    "buyer_name": {"value": null, "confidence": null, "evidence": []},
    "invoice_number": {"value": null, "confidence": null, "evidence": []},
    "invoice_date": {"value": null, "confidence": null, "evidence": []},
    "due_date": {"value": null, "confidence": null, "evidence": []},
    "currency": {"value": null, "confidence": null, "evidence": []},
    "totals": {
      "gross_amount": {"value": null, "confidence": null, "evidence": []},
      "tax_amount": {"value": null, "confidence": null, "evidence": []},
      "net_amount": {"value": null, "confidence": null, "evidence": []}
    },
    "line_items": [
      {
        "description": {"value": null, "confidence": null, "evidence": []},
        "quantity": {"value": null, "confidence": null, "evidence": []},
        "unit_price": {"value": null, "confidence": null, "evidence": []},
        "line_total": {"value": null, "confidence": null, "evidence": []}
      }
    ]
  },
  "clauses": [
    {
      "type": "CLAUSE_TYPE",
      "severity": "LOW|MEDIUM|HIGH",
      "text": "verbatim clause text",
      "page": 1,
      "confidence": 0.0
    }
  ],
  "eligibility": {
    "result": "ELIGIBLE|CONDITIONALLY_ELIGIBLE|INELIGIBLE|UNKNOWN",
    "reason": "",
    "supporting_clauses": []
  },
  "warnings": [],
  "errors": []
}"""

FORMATTER_PASS_MARKER = "--- JSON_FORMATTER_PASS ---"


def build_ocr_text(ocr_pages: list[OCRPage]) -> str:
    chunks = []
    for page in ocr_pages:
        page_lines = [line.text.strip() for line in page.lines if line.text.strip()]
        chunks.append(f"PAGE {page.page_number}\n" + "\n".join(page_lines))
    return "\n\n".join(chunks)


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

    raw_text = text.strip()
    sections = [raw_text]
    if FORMATTER_PASS_MARKER in raw_text:
        first_pass, formatter_pass = raw_text.split(FORMATTER_PASS_MARKER, 1)
        sections = [formatter_pass.strip(), first_pass.strip(), raw_text]

    seen: set[str] = set()
    last_error: Exception | None = None

    for section in sections:
        for candidate in _iter_json_candidates(section):
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
        raise ValueError(f"Model output did not contain a parseable JSON object: {last_error}") from last_error
    raise ValueError("Model output did not contain a JSON object.")

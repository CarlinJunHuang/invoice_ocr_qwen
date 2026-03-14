from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from invoice_ocr_glm_paddle_eval.schema import GroundedEvidence, InvoiceEnvelope, OCRLine, OCRPage


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _best_line_match(needle: str, page: OCRPage) -> tuple[OCRLine | None, float]:
    needle_norm = _normalize_text(needle)
    best_line = None
    best_score = -1.0
    for line in page.lines:
        score = float(fuzz.partial_ratio(needle_norm, _normalize_text(line.text)))
        if score > best_score:
            best_line = line
            best_score = score
    return best_line, best_score


def _collect_grounded(field_path: str, field_payload: Any, pages_by_number: dict[int, OCRPage], threshold: int) -> list[GroundedEvidence]:
    if not isinstance(field_payload, dict):
        return []

    grounded: list[GroundedEvidence] = []
    for evidence in field_payload.get("evidence") or []:
        text = evidence.get("text")
        page_number = evidence.get("page")
        if not text or page_number not in pages_by_number:
            continue
        best_line, score = _best_line_match(text, pages_by_number[page_number])
        if best_line is None or score < threshold:
            continue
        grounded.append(
            GroundedEvidence(
                field_path=field_path,
                text=text,
                page=page_number,
                bbox=best_line.bbox,
                score=score,
            )
        )
    return grounded


def ground_envelope_evidence(envelope: InvoiceEnvelope, ocr_pages: list[OCRPage], threshold: int) -> list[GroundedEvidence]:
    pages_by_number = {page.page_number: page for page in ocr_pages}
    payload = envelope.model_dump(mode="json")
    extracted = payload.get("extracted", {})

    grounded: list[GroundedEvidence] = []
    for field_name in (
        "seller_name",
        "buyer_name",
        "invoice_number",
        "invoice_date",
        "due_date",
        "payment_terms",
        "currency",
        "bank_details",
    ):
        grounded.extend(_collect_grounded(f"extracted.{field_name}", extracted.get(field_name), pages_by_number, threshold))

    totals = extracted.get("totals", {})
    for field_name in ("gross_amount", "tax_amount", "net_amount"):
        grounded.extend(
            _collect_grounded(f"extracted.totals.{field_name}", totals.get(field_name), pages_by_number, threshold)
        )

    for index, item in enumerate(extracted.get("line_items") or []):
        for field_name in ("description", "quantity", "unit_price", "line_total"):
            grounded.extend(
                _collect_grounded(
                    f"extracted.line_items[{index}].{field_name}",
                    item.get(field_name),
                    pages_by_number,
                    threshold,
                )
            )
    return grounded

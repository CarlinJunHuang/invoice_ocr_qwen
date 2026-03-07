from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz

from invoice_ocr_qwen.models.schema import Envelope, GroundedEvidence, OCRLine, OCRPage


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^0-9a-z$]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _score_match(evidence_text: str, line_text: str) -> float:
    normalized_evidence = _normalize_text(evidence_text)
    normalized_line = _normalize_text(line_text)
    if not normalized_evidence or not normalized_line:
        return 0.0
    if normalized_evidence in normalized_line or normalized_line in normalized_evidence:
        return 100.0
    return float(
        max(
            fuzz.partial_ratio(normalized_evidence, normalized_line),
            fuzz.token_set_ratio(normalized_evidence, normalized_line),
        )
    )


def _iter_candidate_lines(ocr_pages: list[OCRPage], page_number: int) -> Iterable[tuple[int, OCRLine]]:
    for page in ocr_pages:
        if page.page_number != page_number:
            continue
        for line in page.lines:
            yield page.page_number, line


def _iter_envelope_evidence(envelope: Envelope) -> Iterable[tuple[str, str, int]]:
    scalar_fields = {
        "extracted.seller_name": envelope.extracted.seller_name,
        "extracted.buyer_name": envelope.extracted.buyer_name,
        "extracted.invoice_number": envelope.extracted.invoice_number,
        "extracted.invoice_date": envelope.extracted.invoice_date,
        "extracted.due_date": envelope.extracted.due_date,
        "extracted.currency": envelope.extracted.currency,
        "extracted.totals.gross_amount": envelope.extracted.totals.gross_amount,
        "extracted.totals.tax_amount": envelope.extracted.totals.tax_amount,
        "extracted.totals.net_amount": envelope.extracted.totals.net_amount,
    }
    for field_path, field in scalar_fields.items():
        for evidence in field.evidence:
            yield field_path, evidence.text, evidence.page

    for index, line_item in enumerate(envelope.extracted.line_items):
        item_fields = {
            f"extracted.line_items[{index}].description": line_item.description,
            f"extracted.line_items[{index}].quantity": line_item.quantity,
            f"extracted.line_items[{index}].unit_price": line_item.unit_price,
            f"extracted.line_items[{index}].line_total": line_item.line_total,
        }
        for field_path, field in item_fields.items():
            for evidence in field.evidence:
                yield field_path, evidence.text, evidence.page

    for index, clause in enumerate(envelope.clauses):
        yield f"clauses[{index}]", clause.text, clause.page


def ground_envelope_evidence(envelope: Envelope, ocr_pages: list[OCRPage], threshold: int = 82) -> list[GroundedEvidence]:
    grounded: list[GroundedEvidence] = []
    for field_path, evidence_text, page_number in _iter_envelope_evidence(envelope):
        best_score = 0.0
        best_line: OCRLine | None = None
        for _, line in _iter_candidate_lines(ocr_pages, page_number):
            score = _score_match(evidence_text, line.text)
            if score > best_score:
                best_score = score
                best_line = line
        if best_line is None or best_score < threshold:
            continue
        grounded.append(
            GroundedEvidence(
                field_path=field_path,
                text=evidence_text,
                page=page_number,
                bbox=best_line.bbox,
                score=round(best_score, 2),
            )
        )
    return grounded


def build_normalized_document(ocr_pages: list[OCRPage]) -> dict[str, object]:
    pages = []
    for page in ocr_pages:
        lines = [line.text for line in page.lines]
        pages.append(
            {
                "page_number": page.page_number,
                "image_path": page.image_path,
                "width": page.width,
                "height": page.height,
                "text": "\n".join(lines),
                "lines": page.model_dump(mode="json")["lines"],
            }
        )
    return {"pages": pages}

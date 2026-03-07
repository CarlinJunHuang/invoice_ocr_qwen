from __future__ import annotations

import re
from datetime import datetime

from invoice_ocr_qwen.extractors.base import ExtractionContext, ExtractionResult
from invoice_ocr_qwen.models.schema import Clause, Envelope, EvidenceItem, ExtractedField, LineItem, OCRLine, OCRPage, build_empty_envelope


INVOICE_NUMBER_RE = re.compile(
    r"(?:invoice|bill)\s*(?:no|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]+)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
)
MONEY_RE = re.compile(r"(?<!\d)([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+\.[0-9]{2})(?!\d)")
LINE_ITEM_RE = re.compile(
    r"^(?P<description>.+?)\s+(?P<quantity>\d+(?:\.\d+)?)\s+(?P<unit_price>\d+\.\d{2})\s+(?P<line_total>\d+\.\d{2})$"
)

CURRENCY_PATTERNS = {
    "SGD": re.compile(r"\bSGD\b|S\$", re.IGNORECASE),
    "USD": re.compile(r"\bUSD\b|US\$", re.IGNORECASE),
    "EUR": re.compile(r"\bEUR\b|€", re.IGNORECASE),
    "GBP": re.compile(r"\bGBP\b|£", re.IGNORECASE),
    "HKD": re.compile(r"\bHKD\b", re.IGNORECASE),
    "AUD": re.compile(r"\bAUD\b", re.IGNORECASE),
    "CNY": re.compile(r"\bCNY\b|\bRMB\b|¥", re.IGNORECASE),
}

CLAUSE_RULES = [
    ("NO_ASSIGNMENT_WITHOUT_CONSENT", "HIGH", re.compile(r"assign(?:ed|ment)? .*consent", re.IGNORECASE)),
    ("SET_OFF", "HIGH", re.compile(r"set[- ]?off|offset|counterclaim|deduct", re.IGNORECASE)),
]


def _make_evidence(text: str, page: int) -> list[EvidenceItem]:
    return [EvidenceItem(text=text, page=page)] if text else []


def _make_field(value: object, line: OCRLine | None, page: int | None, confidence_floor: float = 0.72) -> ExtractedField:
    if value is None:
        return ExtractedField(value=None, confidence=None, evidence=[])
    confidence = confidence_floor
    evidence: list[EvidenceItem] = []
    if line and page is not None:
        confidence = round(max(confidence_floor, min(0.99, line.confidence or confidence_floor)), 2)
        evidence = _make_evidence(line.text, page)
    return ExtractedField(value=value, confidence=confidence, evidence=evidence)


def _iter_lines(ocr_pages: list[OCRPage]) -> list[tuple[int, OCRLine]]:
    return [(page.page_number, line) for page in ocr_pages for line in page.lines]


def _parse_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    candidate = match.group(1).strip()
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_money(text: str) -> float | None:
    matches = MONEY_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def _coerce_number(raw: str) -> int | float:
    numeric = float(raw)
    return int(numeric) if numeric.is_integer() else numeric


def _extract_seller_name(ocr_pages: list[OCRPage]) -> tuple[str | None, int | None, OCRLine | None]:
    if not ocr_pages:
        return None, None, None
    for line in ocr_pages[0].lines[:6]:
        text = line.text.strip()
        lowered = text.lower()
        if not text or ":" in text:
            continue
        if any(token in lowered for token in ("bill to", "invoice", "date", "due", "currency", "description", "amount", "qty")):
            continue
        return text, ocr_pages[0].page_number, line
    return None, None, None


def _extract_buyer_name(ocr_pages: list[OCRPage]) -> tuple[str | None, int | None, OCRLine | None]:
    labels = ("bill to", "invoice to", "customer")
    for page_number, line in _iter_lines(ocr_pages):
        lowered = line.text.lower()
        if any(label in lowered for label in labels):
            if ":" in line.text:
                return line.text.split(":", 1)[1].strip(), page_number, line
            return line.text.strip(), page_number, line
    return None, None, None


def _extract_invoice_number(ocr_pages: list[OCRPage]) -> tuple[str | None, int | None, OCRLine | None]:
    for page_number, line in _iter_lines(ocr_pages):
        match = INVOICE_NUMBER_RE.search(line.text)
        if match:
            return match.group(1).strip(), page_number, line
    return None, None, None


def _extract_labeled_date(ocr_pages: list[OCRPage], labels: tuple[str, ...]) -> tuple[str | None, int | None, OCRLine | None]:
    for page_number, line in _iter_lines(ocr_pages):
        lowered = line.text.lower()
        if any(label in lowered for label in labels):
            parsed = _parse_date(line.text)
            if parsed:
                return parsed, page_number, line
    return None, None, None


def _extract_currency(ocr_pages: list[OCRPage]) -> tuple[str | None, int | None, OCRLine | None]:
    for page_number, line in _iter_lines(ocr_pages):
        for code, pattern in CURRENCY_PATTERNS.items():
            if pattern.search(line.text):
                return code, page_number, line
    return None, None, None


def _extract_totals(ocr_pages: list[OCRPage]) -> dict[str, tuple[float | None, int | None, OCRLine | None]]:
    gross_match: tuple[float | None, int | None, OCRLine | None] = (None, None, None)
    net_match: tuple[float | None, int | None, OCRLine | None] = (None, None, None)
    tax_match: tuple[float | None, int | None, OCRLine | None] = (None, None, None)

    for page_number, line in _iter_lines(ocr_pages):
        lowered = line.text.lower()
        amount = _parse_money(line.text)
        if amount is None:
            continue
        if "tax" in lowered or "gst" in lowered or "vat" in lowered:
            tax_match = (amount, page_number, line)
        if any(token in lowered for token in ("total", "outstanding", "balance", "amount due", "grand total", "net total")):
            gross_match = (amount, page_number, line)
            net_match = (amount, page_number, line)

    return {"gross_amount": gross_match, "tax_amount": tax_match, "net_amount": net_match}


def _extract_line_items(ocr_pages: list[OCRPage]) -> list[LineItem]:
    collected: list[LineItem] = []
    header_seen = False

    for page_number, line in _iter_lines(ocr_pages):
        lowered = line.text.lower()
        if "description" in lowered and ("qty" in lowered or "quantity" in lowered):
            header_seen = True
            continue
        if not header_seen:
            continue
        if any(token in lowered for token in ("total", "balance", "amount due", "grand total")):
            break
        match = LINE_ITEM_RE.match(line.text.strip())
        if not match:
            continue
        collected.append(
            LineItem(
                description=_make_field(match.group("description").strip(), line, page_number, confidence_floor=0.75),
                quantity=_make_field(_coerce_number(match.group("quantity")), line, page_number, confidence_floor=0.75),
                unit_price=_make_field(float(match.group("unit_price")), line, page_number, confidence_floor=0.75),
                line_total=_make_field(float(match.group("line_total")), line, page_number, confidence_floor=0.75),
            )
        )

    return collected


def _extract_clauses(ocr_pages: list[OCRPage]) -> list[Clause]:
    clauses: list[Clause] = []
    seen_types: set[tuple[str, int]] = set()
    for page_number, line in _iter_lines(ocr_pages):
        for clause_type, severity, pattern in CLAUSE_RULES:
            if pattern.search(line.text):
                marker = (clause_type, page_number)
                if marker in seen_types:
                    continue
                seen_types.add(marker)
                clauses.append(
                    Clause(
                        type=clause_type,
                        severity=severity,
                        text=line.text,
                        page=page_number,
                        confidence=round(max(0.72, min(0.99, line.confidence or 0.72)), 2),
                    )
                )
    return clauses


def extract_invoice_from_ocr_pages(request_id: str, ocr_pages: list[OCRPage]) -> Envelope:
    envelope = build_empty_envelope(request_id=request_id)

    seller_name, seller_page, seller_line = _extract_seller_name(ocr_pages)
    buyer_name, buyer_page, buyer_line = _extract_buyer_name(ocr_pages)
    invoice_number, invoice_page, invoice_line = _extract_invoice_number(ocr_pages)
    invoice_date, invoice_date_page, invoice_date_line = _extract_labeled_date(ocr_pages, ("date of bill", "invoice date", "bill date"))
    due_date, due_date_page, due_date_line = _extract_labeled_date(ocr_pages, ("due date", "payment due"))
    currency, currency_page, currency_line = _extract_currency(ocr_pages)
    totals = _extract_totals(ocr_pages)

    envelope.extracted.seller_name = _make_field(seller_name, seller_line, seller_page, confidence_floor=0.88)
    envelope.extracted.buyer_name = _make_field(buyer_name, buyer_line, buyer_page, confidence_floor=0.88)
    envelope.extracted.invoice_number = _make_field(invoice_number, invoice_line, invoice_page, confidence_floor=0.90)
    envelope.extracted.invoice_date = _make_field(invoice_date, invoice_date_line, invoice_date_page, confidence_floor=0.85)
    envelope.extracted.due_date = _make_field(due_date, due_date_line, due_date_page, confidence_floor=0.82)
    envelope.extracted.currency = _make_field(currency, currency_line, currency_page, confidence_floor=0.90)

    for field_name, (amount, page_number, line) in totals.items():
        setattr(envelope.extracted.totals, field_name, _make_field(amount, line, page_number, confidence_floor=0.76))

    envelope.extracted.line_items = _extract_line_items(ocr_pages)
    envelope.clauses = _extract_clauses(ocr_pages)

    if envelope.clauses:
        clause_types = [clause.type for clause in envelope.clauses]
        envelope.eligibility.result = "CONDITIONALLY_ELIGIBLE"
        envelope.eligibility.reason = "Detected potential restrictive clause; requires manual review"
        envelope.eligibility.supporting_clauses = clause_types
    else:
        envelope.eligibility.result = "ELIGIBLE"
        envelope.eligibility.reason = "No adverse clause detected in the provided pages"
        envelope.eligibility.supporting_clauses = []

    return envelope


class RuleBasedExtractor:
    def extract(self, context: ExtractionContext) -> ExtractionResult:
        envelope = extract_invoice_from_ocr_pages(request_id=context.request_id, ocr_pages=context.ocr_pages)
        return ExtractionResult(envelope=envelope, parsed_output=envelope.model_dump(mode="json"))

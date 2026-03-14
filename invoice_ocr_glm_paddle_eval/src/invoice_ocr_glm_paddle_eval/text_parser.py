from __future__ import annotations

import re
from datetime import datetime
from typing import Any


INVOICE_NUMBER_PATTERNS = [
    re.compile(r"(?:invoice|bill|credit note)\s*(?:no|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]+)", re.IGNORECASE),
    re.compile(r"\b(?:inv|cn)\s*#\s*([A-Z0-9][A-Z0-9\-\/]+)", re.IGNORECASE),
]
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
    "EUR": re.compile(r"\bEUR\b", re.IGNORECASE),
    "GBP": re.compile(r"\bGBP\b", re.IGNORECASE),
    "HKD": re.compile(r"\bHKD\b", re.IGNORECASE),
    "AUD": re.compile(r"\bAUD\b", re.IGNORECASE),
    "CNY": re.compile(r"\bCNY\b|\bRMB\b", re.IGNORECASE),
}

PAYMENT_TERMS_RE = re.compile(
    r"(?:payment terms?|terms?)\s*[:\-]?\s*(.+)|\b(net\s*\d+|due on receipt|cash on delivery|direct debit)\b",
    re.IGNORECASE,
)
BANK_DETAILS_RE = re.compile(
    r"(?:account\s*(?:no|number)|acct\s*(?:no|number)|iban|swift|bic|bank)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)
COMPANY_MARKERS = ("pte", "ltd", "limited", "llc", "inc", "corp", "corporation", "co.", "company", "consulting")
GENERIC_HEADER_TOKENS = (
    "invoice",
    "credit note",
    "billing",
    "billings",
    "bill to",
    "invoice to",
    "customer",
    "date",
    "due",
    "currency",
    "qty",
    "quantity",
    "amount",
    "total",
    "charges",
)


def _normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\t", " ").strip()
    return re.sub(r"\s+", " ", line)


def _normalize_raw_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("<|assistant|>", "").replace("<|user|>", "").replace("<image>", "")
    text = text.replace("```text", "").replace("```markdown", "").replace("```", "")
    return text.strip()


def split_text_lines(raw_text: str) -> list[str]:
    text = _normalize_raw_text(raw_text)
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.split("\n"):
        line = _normalize_line(raw_line)
        if not line:
            continue
        if set(line) <= {"-", "|", ":"}:
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [_normalize_line(cell) for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            line = " | ".join(cells)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def _field(value: Any, evidence_text: str | None, confidence: float | None) -> dict[str, Any]:
    evidence = [{"text": evidence_text, "page": 1}] if evidence_text else []
    return {"value": value, "confidence": confidence if value is not None else None, "evidence": evidence}


def _parse_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    candidate = match.group(1).strip()
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
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


def _looks_like_company(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in COMPANY_MARKERS)


def _find_by_labels(lines: list[str], labels: tuple[str, ...], parser) -> tuple[Any, str | None]:
    for index, line in enumerate(lines):
        lowered = line.lower()
        if not any(label in lowered for label in labels):
            continue
        value = parser(line)
        if value is not None:
            return value, line
        if index + 1 < len(lines):
            value = parser(lines[index + 1])
            if value is not None:
                return value, lines[index + 1]
    return None, None


def _find_currency(lines: list[str]) -> tuple[str | None, str | None]:
    for line in lines:
        for code, pattern in CURRENCY_PATTERNS.items():
            if pattern.search(line):
                return code, line
    full_text = "\n".join(lines).lower()
    if "$" in full_text and "singapore" in full_text:
        return "SGD", "$"
    return None, None


def _find_payment_terms(lines: list[str]) -> tuple[str | None, str | None]:
    for line in lines:
        match = PAYMENT_TERMS_RE.search(line)
        if not match:
            continue
        value = match.group(1) or match.group(2)
        if value:
            return _normalize_line(value), line
    return None, None


def _find_bank_details(lines: list[str]) -> tuple[str | None, str | None]:
    for line in lines:
        match = BANK_DETAILS_RE.search(line)
        if not match:
            continue
        value = _normalize_line(match.group(1))
        if value:
            return value, line
    return None, None


def _find_buyer(lines: list[str], seller_name: str | None) -> tuple[str | None, str | None]:
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(label in lowered for label in ("bill to", "invoice to", "customer", "buyer")):
            if ":" in line:
                return _normalize_line(line.split(":", 1)[1]), line
            return line, line
        if "invoice for" in lowered:
            for offset in range(1, 4):
                if index - offset >= 0:
                    candidate = lines[index - offset]
                    if candidate != seller_name and _looks_like_company(candidate):
                        return candidate, candidate
            for offset in range(1, 4):
                if index + offset < len(lines):
                    candidate = lines[index + offset]
                    if candidate != seller_name and _looks_like_company(candidate):
                        return candidate, candidate

    for line in lines[:12]:
        if line != seller_name and _looks_like_company(line):
            return line, line
    return None, None


def _find_seller(lines: list[str]) -> tuple[str | None, str | None]:
    for line in lines[:8]:
        lowered = line.lower()
        if ":" in line:
            continue
        if any(token in lowered for token in GENERIC_HEADER_TOKENS):
            continue
        if _looks_like_company(line) or lowered.startswith("spgroup"):
            return line, line

    for line in lines:
        lowered = line.lower()
        if "payable to" not in lowered:
            continue
        seller = re.split(r"payable to", line, maxsplit=1, flags=re.IGNORECASE)[1]
        seller = seller.split(".", 1)[0].strip(" :;")
        if seller:
            return seller, line
    return None, None


def _find_invoice_number(lines: list[str]) -> tuple[str | None, str | None]:
    for line in lines:
        for pattern in INVOICE_NUMBER_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(1).strip(), line
    return None, None


def _find_totals(lines: list[str]) -> dict[str, tuple[float | None, str | None]]:
    totals = {
        "gross_amount": (None, None),
        "tax_amount": (None, None),
        "net_amount": (None, None),
    }
    for line in lines:
        lowered = line.lower()
        amount = _parse_money(line)
        if amount is None:
            continue
        if any(token in lowered for token in ("tax", "gst", "vat")):
            totals["tax_amount"] = (amount, line)
        if any(token in lowered for token in ("gross total", "grand total", "total amount payable", "total payable")):
            totals["gross_amount"] = (amount, line)
            totals["net_amount"] = (amount, line)
        if any(token in lowered for token in ("net total", "amount due", "outstanding", "balance")):
            totals["net_amount"] = (amount, line)

    gross_value, _ = totals["gross_amount"]
    net_value, _ = totals["net_amount"]
    if gross_value is None and net_value is not None:
        totals["gross_amount"] = totals["net_amount"]
    if net_value is None and gross_value is not None:
        totals["net_amount"] = totals["gross_amount"]
    return totals


def _find_line_items(lines: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    header_seen = False
    for line in lines:
        lowered = line.lower()
        if "description" in lowered and ("qty" in lowered or "quantity" in lowered):
            header_seen = True
            continue
        if not header_seen:
            continue
        if any(token in lowered for token in ("grand total", "amount due", "outstanding", "balance")):
            break

        normalized = line.replace(" | ", " ")
        match = LINE_ITEM_RE.match(normalized)
        if not match:
            continue
        items.append(
            {
                "description": _field(match.group("description").strip(), line, 0.75),
                "quantity": _field(_coerce_number(match.group("quantity")), line, 0.74),
                "unit_price": _field(float(match.group("unit_price")), line, 0.74),
                "line_total": _field(float(match.group("line_total")), line, 0.74),
            }
        )
    return items


def parse_invoice_like_text(raw_text: str) -> dict[str, Any]:
    lines = split_text_lines(raw_text)
    full_text = "\n".join(lines)
    doc_type = "CREDIT_NOTE" if re.search(r"\bcredit note\b|\bcredit memo\b", full_text, re.IGNORECASE) else "INVOICE"

    seller_name, seller_evidence = _find_seller(lines)
    buyer_name, buyer_evidence = _find_buyer(lines, seller_name)
    invoice_number, invoice_number_evidence = _find_invoice_number(lines)
    invoice_date, invoice_date_evidence = _find_by_labels(lines, ("invoice date", "bill date", "date"), _parse_date)
    due_date, due_date_evidence = _find_by_labels(lines, ("due date", "payment due"), _parse_date)
    payment_terms, payment_terms_evidence = _find_payment_terms(lines)
    currency, currency_evidence = _find_currency(lines)
    bank_details, bank_details_evidence = _find_bank_details(lines)
    totals = _find_totals(lines)
    line_items = _find_line_items(lines)

    result = {
        "request_id": "",
        "doc_type": doc_type,
        "language": "en",
        "extracted": {
            "seller_name": _field(seller_name, seller_evidence, 0.86),
            "buyer_name": _field(buyer_name, buyer_evidence, 0.88),
            "invoice_number": _field(invoice_number, invoice_number_evidence, 0.9),
            "invoice_date": _field(invoice_date, invoice_date_evidence, 0.84),
            "due_date": _field(due_date, due_date_evidence, 0.82),
            "payment_terms": _field(payment_terms, payment_terms_evidence, 0.8),
            "currency": _field(currency, currency_evidence, 0.86),
            "bank_details": _field(bank_details, bank_details_evidence, 0.75),
            "totals": {
                "gross_amount": _field(totals["gross_amount"][0], totals["gross_amount"][1], 0.84),
                "tax_amount": _field(totals["tax_amount"][0], totals["tax_amount"][1], 0.72),
                "net_amount": _field(totals["net_amount"][0], totals["net_amount"][1], 0.84),
            },
            "line_items": line_items,
        },
        "warnings": [],
        "errors": [],
    }

    non_null_fields = [
        result["extracted"][name]["value"]
        for name in (
            "seller_name",
            "buyer_name",
            "invoice_number",
            "invoice_date",
            "due_date",
            "payment_terms",
            "currency",
            "bank_details",
        )
    ]
    non_null_fields.extend(
        [
            result["extracted"]["totals"]["gross_amount"]["value"],
            result["extracted"]["totals"]["tax_amount"]["value"],
            result["extracted"]["totals"]["net_amount"]["value"],
        ]
    )
    if all(value is None for value in non_null_fields) and not line_items:
        result["warnings"].append("Model OCR text could not be mapped to invoice fields with the current lightweight parser.")

    return result

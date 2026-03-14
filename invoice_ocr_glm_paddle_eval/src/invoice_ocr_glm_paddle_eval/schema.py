from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    page: int


class ExtractedField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: Any = None
    confidence: float | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _default_evidence(cls, value: Any) -> list[EvidenceItem] | Any:
        return [] if value is None else value


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: ExtractedField = Field(default_factory=ExtractedField)
    quantity: ExtractedField = Field(default_factory=ExtractedField)
    unit_price: ExtractedField = Field(default_factory=ExtractedField)
    line_total: ExtractedField = Field(default_factory=ExtractedField)

    @model_validator(mode="before")
    @classmethod
    def _coerce_compact_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        item = dict(value)
        shared_confidence = item.get("confidence")
        shared_evidence = item.get("evidence") or []
        alias_map = {
            "description": ("description", "item_description"),
            "quantity": ("quantity", "qty"),
            "unit_price": ("unit_price", "price"),
            "line_total": ("line_total", "amount", "total"),
        }
        normalized: dict[str, Any] = {}
        for target_field, aliases in alias_map.items():
            raw_value = None
            for alias in aliases:
                if alias in item:
                    raw_value = item[alias]
                    break
            if isinstance(raw_value, dict):
                raw_value.setdefault("confidence", shared_confidence)
                raw_value.setdefault("evidence", shared_evidence)
                normalized[target_field] = raw_value
            else:
                normalized[target_field] = {
                    "value": raw_value,
                    "confidence": shared_confidence,
                    "evidence": shared_evidence,
                }
        return normalized


class TotalsBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gross_amount: ExtractedField = Field(default_factory=ExtractedField)
    tax_amount: ExtractedField = Field(default_factory=ExtractedField)
    net_amount: ExtractedField = Field(default_factory=ExtractedField)


class ExtractedPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seller_name: ExtractedField = Field(default_factory=ExtractedField)
    buyer_name: ExtractedField = Field(default_factory=ExtractedField)
    invoice_number: ExtractedField = Field(default_factory=ExtractedField)
    invoice_date: ExtractedField = Field(default_factory=ExtractedField)
    due_date: ExtractedField = Field(default_factory=ExtractedField)
    payment_terms: ExtractedField = Field(default_factory=ExtractedField)
    currency: ExtractedField = Field(default_factory=ExtractedField)
    bank_details: ExtractedField = Field(default_factory=ExtractedField)
    totals: TotalsBlock = Field(default_factory=TotalsBlock)
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("totals", mode="before")
    @classmethod
    def _default_totals(cls, value: Any) -> dict[str, Any] | Any:
        return {} if value is None else value

    @field_validator("line_items", mode="before")
    @classmethod
    def _default_line_items(cls, value: Any) -> list[LineItem] | Any:
        return [] if value is None else value


class InvoiceEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    doc_type: str = "INVOICE"
    language: str = "en"
    extracted: ExtractedPayload = Field(default_factory=ExtractedPayload)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("warnings", "errors", mode="before")
    @classmethod
    def _default_lists(cls, value: Any) -> list[str] | Any:
        return [] if value is None else value

    @field_validator("extracted", mode="before")
    @classmethod
    def _default_extracted(cls, value: Any) -> dict[str, Any] | Any:
        return {} if value is None else value


class OCRLine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    confidence: float | None = None
    bbox: list[tuple[float, float]]


class OCRPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_number: int
    image_path: str
    width: int
    height: int
    lines: list[OCRLine] = Field(default_factory=list)


class GroundedEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field_path: str
    text: str
    page: int
    bbox: list[tuple[float, float]]
    score: float


SCHEMA_GUIDE = """{
  "request_id": "REQUEST_ID",
  "doc_type": "INVOICE or CREDIT_NOTE",
  "language": "en",
  "extracted": {
    "seller_name": {"value": null, "confidence": null, "evidence": []},
    "buyer_name": {"value": null, "confidence": null, "evidence": []},
    "invoice_number": {"value": null, "confidence": null, "evidence": []},
    "invoice_date": {"value": null, "confidence": null, "evidence": []},
    "due_date": {"value": null, "confidence": null, "evidence": []},
    "payment_terms": {"value": null, "confidence": null, "evidence": []},
    "currency": {"value": null, "confidence": null, "evidence": []},
    "bank_details": {"value": null, "confidence": null, "evidence": []},
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
  "warnings": [],
  "errors": []
}"""


def build_empty_envelope(request_id: str, language: str = "en") -> InvoiceEnvelope:
    return InvoiceEnvelope(request_id=request_id, doc_type="INVOICE", language=language)


def _wrap_scalar_field(value: Any) -> dict[str, Any]:
    return {"value": value, "confidence": None, "evidence": []}


def normalize_envelope(payload: dict[str, Any] | None, request_id: str, language: str = "en") -> InvoiceEnvelope:
    candidate = dict(payload or {})
    candidate["request_id"] = request_id
    candidate.setdefault("doc_type", "INVOICE")
    candidate.setdefault("language", language)
    extracted = candidate.get("extracted")
    if isinstance(extracted, dict):
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
            field_value = extracted.get(field_name)
            if field_value is not None and not isinstance(field_value, dict):
                extracted[field_name] = _wrap_scalar_field(field_value)
        totals = extracted.get("totals")
        if isinstance(totals, dict):
            for field_name in ("gross_amount", "tax_amount", "net_amount"):
                field_value = totals.get(field_name)
                if field_value is not None and not isinstance(field_value, dict):
                    totals[field_name] = _wrap_scalar_field(field_value)
    return InvoiceEnvelope.model_validate(candidate)

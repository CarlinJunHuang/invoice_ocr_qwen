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
    def _coerce_item_shapes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        item = dict(value)
        if "line_total" not in item:
            for alias in ("amount", "line_amount", "total"):
                if alias in item:
                    item["line_total"] = item[alias]
                    break

        shared_evidence = item.get("evidence") or []
        shared_confidence = item.get("confidence")
        for field_name in ("description", "quantity", "unit_price", "line_total"):
            if field_name not in item:
                continue
            field_value = item[field_name]
            if isinstance(field_value, dict):
                field_value.setdefault("evidence", shared_evidence)
                if shared_confidence is not None:
                    field_value.setdefault("confidence", shared_confidence)
                item[field_name] = field_value
                continue
            item[field_name] = {
                "value": field_value,
                "confidence": shared_confidence,
                "evidence": shared_evidence,
            }

        return item


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
    currency: ExtractedField = Field(default_factory=ExtractedField)
    totals: TotalsBlock = Field(default_factory=TotalsBlock)
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("line_items", mode="before")
    @classmethod
    def _default_line_items(cls, value: Any) -> list[LineItem] | Any:
        return [] if value is None else value

    @field_validator("totals", mode="before")
    @classmethod
    def _default_totals(cls, value: Any) -> dict[str, Any] | Any:
        return {} if value is None else value


class Clause(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    severity: str
    text: str
    page: int
    confidence: float | None = None


class Eligibility(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: str = "UNKNOWN"
    reason: str = ""
    supporting_clauses: list[str] = Field(default_factory=list)

    @field_validator("supporting_clauses", mode="before")
    @classmethod
    def _default_supporting_clauses(cls, value: Any) -> list[str] | Any:
        return [] if value is None else value


class Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    doc_type: str = "INVOICE"
    language: str = "en"
    extracted: ExtractedPayload = Field(default_factory=ExtractedPayload)
    clauses: list[Clause] = Field(default_factory=list)
    eligibility: Eligibility = Field(default_factory=Eligibility)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("clauses", "warnings", "errors", mode="before")
    @classmethod
    def _default_lists(cls, value: Any) -> list[Any] | Any:
        return [] if value is None else value

    @field_validator("extracted", mode="before")
    @classmethod
    def _default_extracted(cls, value: Any) -> dict[str, Any] | Any:
        return {} if value is None else value

    @field_validator("eligibility", mode="before")
    @classmethod
    def _default_eligibility(cls, value: Any) -> dict[str, Any] | Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _normalize_defaults(self) -> "Envelope":
        if not self.doc_type:
            self.doc_type = "INVOICE"
        if not self.language:
            self.language = "en"
        return self


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


def build_empty_envelope(request_id: str, language: str = "en") -> Envelope:
    return Envelope(request_id=request_id, doc_type="INVOICE", language=language)


def normalize_envelope(payload: dict[str, Any] | None, request_id: str, language: str = "en") -> Envelope:
    candidate = dict(payload or {})
    candidate["request_id"] = request_id
    candidate.setdefault("doc_type", "INVOICE")
    candidate.setdefault("language", language)
    extracted = candidate.get("extracted")
    if isinstance(extracted, dict):
        for field_name in ("seller_name", "buyer_name", "invoice_number", "invoice_date", "due_date", "currency"):
            field_value = extracted.get(field_name)
            if field_value is not None and not isinstance(field_value, dict):
                extracted[field_name] = {"value": field_value, "confidence": None, "evidence": []}
        totals = extracted.get("totals")
        if isinstance(totals, dict):
            for field_name in ("gross_amount", "tax_amount", "net_amount"):
                field_value = totals.get(field_name)
                if field_value is not None and not isinstance(field_value, dict):
                    totals[field_name] = {"value": field_value, "confidence": None, "evidence": []}
    return Envelope.model_validate(candidate)

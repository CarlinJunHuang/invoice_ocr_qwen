from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from invoice_ocr_glm_paddle_eval.schema import InvoiceEnvelope

_BG_COLOR = "#F7F4EA"
_HEADER_BG = "#1F3A5F"
_HEADER_FG = "#FFFFFF"
_CARD_BORDER = "#D8D0BF"
_TEXT_COLOR = "#1E1E1E"


def _read_json(path: str | None) -> dict[str, Any] | list[Any] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def _read_text(path: str | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8")


def _safe_relpath(target: str | None, base_dir: Path) -> str | None:
    if not target:
        return None
    try:
        return Path(os.path.relpath(Path(target).resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return Path(target).resolve().as_posix()


def _get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _value_at_path(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _field_value(payload: dict[str, Any], *path: str) -> Any:
    field = _value_at_path(payload, list(path))
    if isinstance(field, dict):
        return field.get("value")
    return field


def _line_item_count(payload: dict[str, Any]) -> int:
    items = _value_at_path(payload, ["extracted", "line_items"])
    return len(items) if isinstance(items, list) else 0


def _collect_result_detail(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    envelope_payload = _read_json(artifacts.get("invoice_fields")) or {}
    if isinstance(envelope_payload, dict):
        try:
            envelope = InvoiceEnvelope.model_validate(envelope_payload)
            envelope_payload = envelope.model_dump(mode="json")
        except Exception:
            pass

    grounded_boxes = _read_json(artifacts.get("grounded_boxes")) or []
    raw_output = _read_text(artifacts.get("raw_model_output")) or ""
    parser_output = _read_text(artifacts.get("parser_output")) or ""

    metrics = result.get("metrics", {})
    main_model = metrics.get("main_model") or {}
    parser_model = metrics.get("parser_model") or {}
    errors = envelope_payload.get("errors") if isinstance(envelope_payload, dict) else []
    warnings = envelope_payload.get("warnings") if isinstance(envelope_payload, dict) else []

    return {
        "image": result.get("image"),
        "mode": result.get("mode"),
        "run_name": result.get("run_name"),
        "request_id": result.get("request_id"),
        "overlay_images": artifacts.get("overlay_images") or [],
        "raw_output": raw_output,
        "parser_output": parser_output,
        "envelope": envelope_payload,
        "grounded_box_count": len(grounded_boxes) if isinstance(grounded_boxes, list) else 0,
        "errors": errors or [],
        "warnings": warnings or [],
        "total_elapsed_seconds": metrics.get("total_elapsed_seconds"),
        "main_model": main_model,
        "parser_model": parser_model,
        "seller_name": _field_value(envelope_payload, "extracted", "seller_name"),
        "buyer_name": _field_value(envelope_payload, "extracted", "buyer_name"),
        "invoice_number": _field_value(envelope_payload, "extracted", "invoice_number"),
        "currency": _field_value(envelope_payload, "extracted", "currency"),
        "gross_amount": _field_value(envelope_payload, "extracted", "totals", "gross_amount"),
        "line_item_count": _line_item_count(envelope_payload),
        "artifact_paths": artifacts,
    }


def _markdown_table_rows(details: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| Image | Mode | Seller | Buyer | Invoice No. | Currency | Gross | Lines | Boxes | Total(s) | Main Tokens | Parser Tokens | Errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for detail in details:
        main_tokens = detail["main_model"].get("total_tokens") if isinstance(detail["main_model"], dict) else None
        parser_tokens = detail["parser_model"].get("total_tokens") if isinstance(detail["parser_model"], dict) else None
        rows.append(
            "| {image} | {mode} | {seller} | {buyer} | {invoice} | {currency} | {gross} | {lines} | {boxes} | {elapsed} | {main_tokens} | {parser_tokens} | {errors} |".format(
                image=Path(str(detail["image"])).name,
                mode=detail["mode"] or "",
                seller=str(detail["seller_name"] or ""),
                buyer=str(detail["buyer_name"] or ""),
                invoice=str(detail["invoice_number"] or ""),
                currency=str(detail["currency"] or ""),
                gross=str(detail["gross_amount"] or ""),
                lines=detail["line_item_count"],
                boxes=detail["grounded_box_count"],
                elapsed=detail["total_elapsed_seconds"] or "",
                main_tokens=main_tokens or "",
                parser_tokens=parser_tokens or "",
                errors="; ".join(detail["errors"]) if detail["errors"] else "",
            )
        )
    return rows


def _render_markdown(details: list[dict[str, Any]], report_dir: Path, montage_paths: dict[str, str]) -> str:
    parts = [
        "# OCR Compare Report",
        "",
        "## Summary",
        "",
        *_markdown_table_rows(details),
        "",
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail["image"])].append(detail)

    for image_path, items in grouped.items():
        image_name = Path(image_path).name
        parts.extend(
            [
                f"## Image: {image_name}",
                "",
            ]
        )
        montage_path = montage_paths.get(image_path)
        if montage_path:
            parts.extend(
                [
                    f"![{image_name} montage]({Path(montage_path).name})",
                    "",
                ]
            )
        for detail in items:
            parts.extend(
                [
                    f"### Mode: {detail['mode']}",
                    "",
                    f"- Total elapsed: `{detail['total_elapsed_seconds']}` seconds",
                    f"- Main model tokens: `{detail['main_model'].get('input_tokens', '')}` in / `{detail['main_model'].get('output_tokens', '')}` out",
                    f"- Parser tokens: `{detail['parser_model'].get('input_tokens', '')}` in / `{detail['parser_model'].get('output_tokens', '')}` out",
                    f"- Grounded boxes: `{detail['grounded_box_count']}`",
                    f"- Errors: `{'; '.join(detail['errors']) if detail['errors'] else 'None'}`",
                    "",
                ]
            )
            overlay_images = detail["overlay_images"] or []
            if overlay_images:
                overlay_rel = _safe_relpath(overlay_images[0], report_dir)
                if overlay_rel:
                    parts.extend(
                        [
                            f"![{detail['mode']} overlay]({overlay_rel})",
                            "",
                        ]
                    )

            parts.extend(
                [
                    "#### Invoice JSON",
                    "",
                    "```json",
                    json.dumps(detail["envelope"], indent=2, ensure_ascii=False),
                    "```",
                    "",
                    "<details>",
                    "<summary>Raw model output</summary>",
                    "",
                    "```text",
                    detail["raw_output"] or "",
                    "```",
                    "</details>",
                    "",
                ]
            )
            if detail["parser_output"]:
                parts.extend(
                    [
                        "<details>",
                        "<summary>Parser output</summary>",
                        "",
                        "```text",
                        detail["parser_output"],
                        "```",
                        "</details>",
                        "",
                    ]
                )
    return "\n".join(parts).strip() + "\n"


def _html_table_rows(details: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for detail in details:
        rows.append(
            "<tr>"
            f"<td>{html.escape(Path(str(detail['image'])).name)}</td>"
            f"<td>{html.escape(str(detail['mode'] or ''))}</td>"
            f"<td>{html.escape(str(detail['seller_name'] or ''))}</td>"
            f"<td>{html.escape(str(detail['buyer_name'] or ''))}</td>"
            f"<td>{html.escape(str(detail['invoice_number'] or ''))}</td>"
            f"<td>{html.escape(str(detail['currency'] or ''))}</td>"
            f"<td>{html.escape(str(detail['gross_amount'] or ''))}</td>"
            f"<td>{detail['line_item_count']}</td>"
            f"<td>{detail['grounded_box_count']}</td>"
            f"<td>{detail['total_elapsed_seconds'] or ''}</td>"
            f"<td>{detail['main_model'].get('total_tokens', '')}</td>"
            f"<td>{detail['parser_model'].get('total_tokens', '')}</td>"
            f"<td>{html.escape('; '.join(detail['errors']) if detail['errors'] else '')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_html(details: list[dict[str, Any]], report_dir: Path, montage_paths: dict[str, str]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail["image"])].append(detail)

    sections: list[str] = []
    for image_path, items in grouped.items():
        image_name = Path(image_path).name
        montage_path = montage_paths.get(image_path)
        montage_html = ""
        if montage_path:
            rel = _safe_relpath(montage_path, report_dir)
            if rel:
                montage_html = f'<img class="montage" src="{html.escape(rel)}" alt="{html.escape(image_name)} montage" />'

        cards: list[str] = []
        for detail in items:
            overlay_html = ""
            overlay_images = detail["overlay_images"] or []
            if overlay_images:
                rel = _safe_relpath(overlay_images[0], report_dir)
                if rel:
                    overlay_html = f'<img class="overlay" src="{html.escape(rel)}" alt="{html.escape(detail["mode"])} overlay" />'

            parser_block = ""
            if detail["parser_output"]:
                parser_block = (
                    '<details><summary>Parser output</summary>'
                    f'<pre>{html.escape(detail["parser_output"])}</pre>'
                    '</details>'
                )

            cards.append(
                '<section class="card">'
                f'<h3>{html.escape(str(detail["mode"]))}</h3>'
                f'<p><strong>Total elapsed:</strong> {html.escape(str(detail["total_elapsed_seconds"] or ""))} seconds<br>'
                f'<strong>Main tokens:</strong> {html.escape(str(detail["main_model"].get("input_tokens", "")))} in / {html.escape(str(detail["main_model"].get("output_tokens", "")))} out<br>'
                f'<strong>Parser tokens:</strong> {html.escape(str(detail["parser_model"].get("input_tokens", "")))} in / {html.escape(str(detail["parser_model"].get("output_tokens", "")))} out<br>'
                f'<strong>Grounded boxes:</strong> {detail["grounded_box_count"]}<br>'
                f'<strong>Errors:</strong> {html.escape("; ".join(detail["errors"]) if detail["errors"] else "None")}</p>'
                f'{overlay_html}'
                '<h4>Invoice JSON</h4>'
                f'<pre>{html.escape(json.dumps(detail["envelope"], indent=2, ensure_ascii=False))}</pre>'
                '<details><summary>Raw model output</summary>'
                f'<pre>{html.escape(detail["raw_output"] or "")}</pre>'
                '</details>'
                f'{parser_block}'
                '</section>'
            )

        sections.append(
            '<section class="image-group">'
            f'<h2>{html.escape(image_name)}</h2>'
            f'{montage_html}'
            + "".join(cards)
            + '</section>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>OCR Compare Report</title>
  <style>
    body {{
      background: {_BG_COLOR};
      color: {_TEXT_COLOR};
      font-family: Segoe UI, Arial, sans-serif;
      margin: 0;
      padding: 24px;
    }}
    h1, h2, h3, h4 {{
      margin-top: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 24px;
      background: #fff;
    }}
    th, td {{
      border: 1px solid {_CARD_BORDER};
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: {_HEADER_BG};
      color: {_HEADER_FG};
    }}
    .image-group {{
      margin-bottom: 40px;
    }}
    .card {{
      background: #fff;
      border: 1px solid {_CARD_BORDER};
      padding: 16px;
      margin-bottom: 20px;
    }}
    .overlay, .montage {{
      max-width: 100%;
      display: block;
      margin: 12px 0;
      border: 1px solid {_CARD_BORDER};
    }}
    pre {{
      background: #f4f1e8;
      padding: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <h1>OCR Compare Report</h1>
  <h2>Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Image</th>
        <th>Mode</th>
        <th>Seller</th>
        <th>Buyer</th>
        <th>Invoice No.</th>
        <th>Currency</th>
        <th>Gross</th>
        <th>Lines</th>
        <th>Boxes</th>
        <th>Total(s)</th>
        <th>Main Tokens</th>
        <th>Parser Tokens</th>
        <th>Errors</th>
      </tr>
    </thead>
    <tbody>
      {_html_table_rows(details)}
    </tbody>
  </table>
  {''.join(sections)}
</body>
</html>
"""


def _build_montage(image_path: str, items: list[dict[str, Any]], report_dir: Path) -> str | None:
    panels: list[tuple[str, Image.Image]] = []
    for item in items:
        overlay_images = item.get("overlay_images") or []
        if not overlay_images:
            continue
        overlay_path = Path(overlay_images[0])
        if not overlay_path.exists():
            continue
        with Image.open(overlay_path) as overlay:
            panels.append((str(item["mode"]), overlay.convert("RGB").copy()))

    if not panels:
        return None

    title_font = _get_font(22)
    label_height = 40
    panel_spacing = 18
    max_height = max(image.height for _, image in panels)
    total_width = sum(image.width for _, image in panels) + panel_spacing * (len(panels) - 1)
    canvas = Image.new("RGB", (total_width, max_height + label_height), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, panel in panels:
        draw.rectangle([(x, 0), (x + panel.width, label_height)], fill=_HEADER_BG)
        draw.text((x + 12, 9), label, fill=_HEADER_FG, font=title_font)
        canvas.paste(panel, (x, label_height))
        x += panel.width + panel_spacing

    montage_path = report_dir / f"{Path(image_path).stem}_overlay_montage.png"
    canvas.save(montage_path)
    return str(montage_path)


def build_compare_report(run_name: str, results: list[dict[str, Any]], report_root: Path) -> dict[str, Any]:
    report_dir = report_root / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    details = [_collect_result_detail(result) for result in results]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail["image"])].append(detail)

    montage_paths: dict[str, str] = {}
    for image_path, items in grouped.items():
        montage_path = _build_montage(image_path, items, report_dir)
        if montage_path:
            montage_paths[image_path] = montage_path

    markdown = _render_markdown(details, report_dir, montage_paths)
    html_report = _render_html(details, report_dir, montage_paths)

    markdown_path = report_dir / "report.md"
    html_path = report_dir / "report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")

    summary_path = report_dir / "report_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "result_count": len(details),
                "report_markdown": str(markdown_path),
                "report_html": str(html_path),
                "montages": montage_paths,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "report_markdown": str(markdown_path),
        "report_html": str(html_path),
        "report_summary": str(summary_path),
        "montages": montage_paths,
    }

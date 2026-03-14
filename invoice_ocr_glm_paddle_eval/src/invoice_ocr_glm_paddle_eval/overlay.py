from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from invoice_ocr_glm_paddle_eval.schema import GroundedEvidence

_COLORS = [
    "#E85D04",
    "#0A9396",
    "#BB3E03",
    "#005F73",
    "#9B2226",
    "#CA6702",
    "#3A86FF",
    "#6A994E",
]


def _get_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def render_overlay_images(
    image_paths: list[Path],
    grounded_evidence: list[GroundedEvidence],
    output_dir: Path,
    line_width: int,
    font_size: int,
) -> list[str]:
    font = _get_font(font_size)
    overlay_paths: list[str] = []
    grouped: dict[int, list[GroundedEvidence]] = {}
    for item in grounded_evidence:
        grouped.setdefault(item.page, []).append(item)

    for page_number, image_path in enumerate(image_paths, start=1):
        page_items = grouped.get(page_number, [])
        if not page_items:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for index, item in enumerate(page_items):
                color = _COLORS[index % len(_COLORS)]
                xs = [point[0] for point in item.bbox]
                ys = [point[1] for point in item.bbox]
                bounds = [(min(xs), min(ys)), (max(xs), max(ys))]
                draw.rectangle(bounds, outline=color, width=line_width)
                label = item.field_path.split(".")[-1]
                draw.text((bounds[0][0], max(bounds[0][1] - font_size - 2, 0)), label, fill=color, font=font)

            overlay_path = output_dir / f"page_{page_number:02d}_overlay.png"
            image.save(overlay_path)
            overlay_paths.append(str(overlay_path))
    return overlay_paths

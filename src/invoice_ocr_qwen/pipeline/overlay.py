from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from invoice_ocr_qwen.models.schema import GroundedEvidence


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


def _get_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def render_overlay_images(
    image_paths: list[Path],
    grounded_evidence: list[GroundedEvidence],
    output_dir: Path,
    line_width: int = 3,
    font_size: int = 18,
) -> list[str]:
    if not grounded_evidence:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    font = _get_font(font_size)
    output_paths: list[str] = []

    for page_number, image_path in enumerate(image_paths, start=1):
        page_matches = [item for item in grounded_evidence if item.page == page_number]
        if not page_matches or not image_path.exists():
            continue

        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for index, item in enumerate(page_matches):
                color = PALETTE[index % len(PALETTE)]
                flat_points = [tuple(point) for point in item.bbox]
                draw.line(flat_points + [flat_points[0]], fill=color, width=line_width)
                x0, y0 = flat_points[0]
                label = item.field_path
                text_box = draw.textbbox((x0, y0), label, font=font)
                draw.rectangle(text_box, fill=color)
                draw.text((x0, y0), label, fill="white", font=font)

            output_path = output_dir / f"page_{page_number:02d}_overlay.png"
            image.save(output_path)
            output_paths.append(str(output_path))

    return output_paths

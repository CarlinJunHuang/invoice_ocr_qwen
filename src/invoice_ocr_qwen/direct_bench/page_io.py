from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
from PIL import Image
import pypdfium2 as pdfium


@dataclass
class RenderedPage:
    """Standard page payload passed into provider pipelines."""

    page_index: int
    image: np.ndarray
    source_width: int
    source_height: int
    rendered_width: int
    rendered_height: int


def is_pdf(filename: str | None, content_type: str | None, file_bytes: bytes) -> bool:
    """Best-effort PDF detection for multipart uploads."""

    if content_type and "application/pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return file_bytes.startswith(b"%PDF")


def read_image_page(file_bytes: bytes) -> list[RenderedPage]:
    """Load a single image upload as page 1."""

    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    width, height = image.size
    return [
        RenderedPage(
            page_index=1,
            image=np.array(image),
            source_width=width,
            source_height=height,
            rendered_width=width,
            rendered_height=height,
        )
    ]


def read_pdf_pages(file_bytes: bytes, dpi: int = 300, max_pages: int | None = None) -> list[RenderedPage]:
    """Render PDF pages into RGB numpy arrays using pdfium."""

    document = pdfium.PdfDocument(file_bytes)
    page_total = len(document)
    upper = page_total if not max_pages or max_pages < 1 else min(page_total, max_pages)

    pages: list[RenderedPage] = []
    scale = max(72, dpi) / 72.0
    for idx in range(upper):
        page = document[idx]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil().convert("RGB")
        width, height = pil_image.size
        pages.append(
            RenderedPage(
                page_index=idx + 1,
                image=np.array(pil_image),
                source_width=width,
                source_height=height,
                rendered_width=width,
                rendered_height=height,
            )
        )
        bitmap.close()
        page.close()

    document.close()
    return pages


def resize_page_image(
    image: np.ndarray,
    max_long_side: int | None = None,
    max_pixels: int | None = None,
) -> np.ndarray:
    """Resize page image with conservative limits to avoid runtime memory spikes."""

    if image is None:
        return image

    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    if width <= 0 or height <= 0:
        return image

    scale = 1.0
    long_side = max(width, height)
    if max_long_side and max_long_side > 0 and long_side > max_long_side:
        scale = min(scale, float(max_long_side) / float(long_side))

    if max_pixels and max_pixels > 0:
        pixels = width * height
        if pixels > max_pixels:
            scale = min(scale, math.sqrt(float(max_pixels) / float(pixels)))

    if scale >= 0.999:
        return image

    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = pil_image.resize((resized_w, resized_h), resample=Image.Resampling.LANCZOS)
    return np.array(resized)


def load_document_pages(
    file_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    dpi: int = 300,
    max_pages: int | None = None,
    max_long_side: int | None = None,
    max_pixels: int | None = None,
) -> list[RenderedPage]:
    """Unified loader for image/PDF uploads."""

    if is_pdf(filename=filename, content_type=content_type, file_bytes=file_bytes):
        pages = read_pdf_pages(file_bytes=file_bytes, dpi=dpi, max_pages=max_pages)
    else:
        pages = read_image_page(file_bytes=file_bytes)

    if not max_long_side and not max_pixels:
        return pages

    resized_pages: list[RenderedPage] = []
    for page in pages:
        resized_image = resize_page_image(page.image, max_long_side=max_long_side, max_pixels=max_pixels)
        resized_pages.append(
            RenderedPage(
                page_index=page.page_index,
                image=resized_image,
                source_width=page.source_width,
                source_height=page.source_height,
                rendered_width=int(resized_image.shape[1]),
                rendered_height=int(resized_image.shape[0]),
            )
        )
    return resized_pages


def build_page_dimension_metadata(pages: list[RenderedPage]) -> dict[str, str]:
    """Flatten per-page coordinate space details into metadata.extra."""

    extra: dict[str, str] = {}
    for page in pages:
        prefix = f"page_{page.page_index}"
        extra[f"{prefix}_source_width"] = str(page.source_width)
        extra[f"{prefix}_source_height"] = str(page.source_height)
        extra[f"{prefix}_render_width"] = str(page.rendered_width)
        extra[f"{prefix}_render_height"] = str(page.rendered_height)
    return extra

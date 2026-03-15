from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import requests


LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal vLLM client for dots.ocr 1.5 raw-output tryout")
    parser.add_argument("--input-image", required=True, help="Path to a local image file")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions", help="OpenAI-compatible vLLM endpoint")
    parser.add_argument("--model-name", default="model", help="Served model name")
    parser.add_argument("--run-name", default=None, help="Optional output directory name")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Response token cap")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    output_root = project_root / "outputs" / "vllm"
    output_root.mkdir(parents=True, exist_ok=True)

    input_image = Path(args.input_image).resolve()
    if not input_image.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_image}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_b64 = base64.b64encode(input_image.read_bytes()).decode("ascii")
    payload = {
        "model": args.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LAYOUT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{input_image.suffix.lstrip('.').lower()};base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": args.max_tokens,
    }
    _write_json(output_dir / "request.json", payload)

    started_at = perf_counter()
    try:
        response = requests.post(args.endpoint, json=payload, timeout=1800)
        response.raise_for_status()
        data = response.json()
        _write_json(output_dir / "response.json", data)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            raw_output = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            raw_output = str(content)
        (output_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
        _write_json(
            output_dir / "status.json",
            {
                "status": "success",
                "elapsed_seconds": round(perf_counter() - started_at, 3),
                "raw_output_empty": not raw_output.strip(),
            },
        )
        return 0
    except Exception as exc:
        error_message = str(exc)
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            error_message = f"{error_message}\n\n{exc.response.text}"
        (output_dir / "raw_output.txt").write_text(error_message, encoding="utf-8")
        (output_dir / "error.txt").write_text(error_message, encoding="utf-8")
        _write_json(
            output_dir / "status.json",
            {
                "status": "failed",
                "elapsed_seconds": round(perf_counter() - started_at, 3),
                "error": error_message,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

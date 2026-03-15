from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter


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
    parser = argparse.ArgumentParser(description="Minimal HF raw-output tryout for dots.ocr 1.5")
    parser.add_argument("--input-image", required=True, help="Path to a local image file")
    parser.add_argument("--model-id", default="kristaller486/dots.ocr-1.5", help="Model source to download")
    parser.add_argument("--run-name", default=None, help="Optional output directory name")
    parser.add_argument("--max-new-tokens", type=int, default=24000, help="Generation cap")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    runtime_root = project_root / ".local_runtime"
    hf_home = runtime_root / "hf-cache"
    weights_root = runtime_root / "weights"
    output_root = project_root / "outputs" / "hf"
    for path in (runtime_root, hf_home, weights_root, output_root):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
    os.environ["HF_MODULES_CACHE"] = str(hf_home / "modules")
    os.environ["XDG_CACHE_HOME"] = str(runtime_root)

    input_image = Path(args.input_image).resolve()
    if not input_image.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_image}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input_image": str(input_image),
        "model_id": args.model_id,
        "prompt": LAYOUT_PROMPT,
        "max_new_tokens": args.max_new_tokens,
    }
    _write_json(output_dir / "metadata.json", metadata)

    started_at = perf_counter()
    try:
        import torch
        from huggingface_hub import snapshot_download
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForCausalLM, AutoProcessor

        local_model_dir = weights_root / "DotsOCR_1_5"
        if not local_model_dir.exists():
            snapshot_download(
                repo_id=args.model_id,
                local_dir=str(local_model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            str(local_model_dir),
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(str(local_model_dir), trust_remote_code=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(input_image)},
                    {"type": "text", "text": LAYOUT_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

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
        (output_dir / "raw_output.txt").write_text(str(exc), encoding="utf-8")
        (output_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        _write_json(
            output_dir / "status.json",
            {
                "status": "failed",
                "elapsed_seconds": round(perf_counter() - started_at, 3),
                "error": str(exc),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

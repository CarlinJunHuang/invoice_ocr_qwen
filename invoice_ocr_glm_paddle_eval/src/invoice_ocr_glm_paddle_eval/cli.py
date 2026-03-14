from __future__ import annotations

import argparse
import json
from pathlib import Path

from invoice_ocr_glm_paddle_eval.config import load_config
from invoice_ocr_glm_paddle_eval.pipeline import run_compare, run_extract


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple local evaluation for GLM-OCR, PaddleOCR-VL, and FireRed-OCR invoice extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Run one image with one mode")
    extract_parser.add_argument("--config", required=True, help="Path to YAML config")
    extract_parser.add_argument("--mode", required=True, help="Mode name from config")
    extract_parser.add_argument("--input", nargs=1, required=True, help="Single input image path")
    extract_parser.add_argument("--run-name", default=None, help="Optional output run directory name")

    compare_parser = subparsers.add_parser("compare", help="Run one or more images across multiple modes")
    compare_parser.add_argument("--config", required=True, help="Path to YAML config")
    compare_parser.add_argument("--modes", nargs="+", required=True, help="Mode names from config")
    compare_parser.add_argument("--input", nargs="+", required=True, help="One or more input image paths")
    compare_parser.add_argument("--run-name", default=None, help="Optional output run directory name")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "extract":
        summary = run_extract(
            image_path=Path(args.input[0]),
            mode_name=args.mode,
            config=config,
            run_name=args.run_name,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "compare":
        summary = run_compare(
            image_paths=[Path(path) for path in args.input],
            mode_names=list(args.modes),
            config=config,
            run_name=args.run_name,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

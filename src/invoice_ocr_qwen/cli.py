from __future__ import annotations

import argparse
import json
from pathlib import Path

from invoice_ocr_qwen.config import load_config
from invoice_ocr_qwen.pipeline.benchmark import run_benchmark
from invoice_ocr_qwen.pipeline.orchestrator import run_extraction


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local invoice extraction benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Run one extraction mode")
    extract_parser.add_argument("--config", required=True, help="Path to the YAML config")
    extract_parser.add_argument("--mode", required=True, help="Mode name from the config")
    extract_parser.add_argument("--input", nargs="+", required=True, help="One or more page images")
    extract_parser.add_argument("--request-id", default=None, help="Optional fixed request id")
    extract_parser.add_argument("--run-name", default=None, help="Optional output run folder name")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run multiple extraction modes")
    benchmark_parser.add_argument("--config", required=True, help="Path to the YAML config")
    benchmark_parser.add_argument("--modes", nargs="+", required=True, help="One or more mode names")
    benchmark_parser.add_argument("--input", nargs="+", required=True, help="One or more page images")
    benchmark_parser.add_argument("--run-name", default=None, help="Optional output run folder name")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "extract":
        summary = run_extraction(
            image_paths=[Path(path) for path in args.input],
            mode_name=args.mode,
            config=config,
            request_id=args.request_id,
            run_name=args.run_name,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "benchmark":
        summary = run_benchmark(
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

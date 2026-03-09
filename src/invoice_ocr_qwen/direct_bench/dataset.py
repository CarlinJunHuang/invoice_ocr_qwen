from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from invoice_ocr_qwen.direct_bench.app import OUTPUT_ROOT, RunConfig, load_prompt_text, run


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "run"


def _nested_get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        return os.getenv(args.api_key_env)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch runner for the direct invoice VLM bench.")
    parser.add_argument("--backend", choices=("ollama", "openai-compatible"), default="ollama")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--model", action="append", required=True, dest="models")
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--prompt-file")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--timeout-sec", type=float, default=420.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--max-long-side", type=int, default=2200)
    parser.add_argument("--max-pixels", type=int, default=6000000)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--allow-thinking-fallback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    dataset_dir = output_root / args.run_prefix
    dataset_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = load_prompt_text(args.prompt_file)
    if args.backend == "ollama":
        base_url = args.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        api_key = None
    else:
        base_url = args.base_url or os.getenv("QWEN_API_BASE_URL", "").strip()
        api_key = _resolve_api_key(args)
        if not base_url:
            parser.error("QWEN_API_BASE_URL or --base-url is required for openai-compatible backend.")
        if not api_key:
            parser.error("QWEN_API_KEY or --api-key is required for openai-compatible backend.")

    rows: list[dict[str, Any]] = []
    for model in args.models:
        for raw_input in args.inputs:
            input_path = Path(raw_input).resolve()
            if not input_path.exists():
                parser.error(f"Input file does not exist: {input_path}")
            run_name = f"{args.run_prefix}-{_sanitize(model)}-{_sanitize(input_path.name)}"
            config = RunConfig(
                backend=args.backend,
                model=model,
                input_path=input_path,
                output_dir=output_root / run_name,
                prompt_text=prompt_text,
                base_url=base_url,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                dpi=args.dpi,
                max_pages=args.max_pages,
                max_long_side=args.max_long_side,
                max_pixels=args.max_pixels,
                jpeg_quality=args.jpeg_quality,
                allow_thinking_fallback=args.allow_thinking_fallback,
            )
            summary = run(config)
            parsed_path = Path(summary["artifacts"]["parsed_output"])
            parsed_output = json.loads(parsed_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model,
                    "input": input_path.name,
                    "input_path": str(input_path),
                    "elapsed_ms": int(summary.get("elapsed_ms", 0)),
                    "fields_found": int(summary.get("fields_found", 0)),
                    "line_item_count": int(summary.get("line_item_count", 0)),
                    "error_count": int(summary.get("error_count", 0)),
                    "invoice_number": _nested_get(parsed_output, "fields", "invoice_number", "value"),
                    "invoice_date": _nested_get(parsed_output, "fields", "invoice_date", "value"),
                    "due_date": _nested_get(parsed_output, "fields", "due_date", "value"),
                    "total_amount": _nested_get(parsed_output, "fields", "total_amount", "value"),
                    "currency": _nested_get(parsed_output, "fields", "currency", "value"),
                    "bank_name": _nested_get(parsed_output, "fields", "bank_name", "value"),
                    "envelope": summary["artifacts"]["envelope"],
                }
            )

    summary_tsv = dataset_dir / "dataset-summary.tsv"
    summary_jsonl = dataset_dir / "dataset-summary.jsonl"
    aggregate_tsv = dataset_dir / "dataset-aggregate.tsv"

    header = [
        "model",
        "input",
        "elapsed_ms",
        "fields_found",
        "line_item_count",
        "error_count",
        "invoice_number",
        "invoice_date",
        "due_date",
        "total_amount",
        "currency",
        "bank_name",
    ]
    summary_lines = ["\t".join(header)]
    for row in rows:
        summary_lines.append(
            "\t".join(
                str(row.get(column, "") or "")
                for column in header
            )
        )
    summary_tsv.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    summary_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    aggregates: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        run_count = len(model_rows)
        aggregates.append(
            {
                "model": model,
                "runs": run_count,
                "avg_elapsed_ms": int(sum(row["elapsed_ms"] for row in model_rows) / max(1, run_count)),
                "total_fields_found": sum(row["fields_found"] for row in model_rows),
                "total_line_items": sum(row["line_item_count"] for row in model_rows),
                "error_runs": sum(1 for row in model_rows if row["error_count"] > 0),
            }
        )

    aggregate_header = [
        "model",
        "runs",
        "avg_elapsed_ms",
        "total_fields_found",
        "total_line_items",
        "error_runs",
    ]
    aggregate_lines = ["\t".join(aggregate_header)]
    for row in aggregates:
        aggregate_lines.append("\t".join(str(row[column]) for column in aggregate_header))
    aggregate_tsv.write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "run_prefix": args.run_prefix,
                "backend": args.backend,
                "dataset_dir": str(dataset_dir),
                "summary_tsv": str(summary_tsv),
                "summary_jsonl": str(summary_jsonl),
                "aggregate_tsv": str(aggregate_tsv),
                "runs": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

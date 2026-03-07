from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from invoice_ocr_qwen.config import AppConfig
from invoice_ocr_qwen.pipeline.orchestrator import run_extraction


def run_benchmark(
    image_paths: list[Path],
    mode_names: list[str],
    config: AppConfig,
    run_name: str | None = None,
) -> dict[str, object]:
    run_name = run_name or f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    benchmark_root = config.output_root / run_name
    benchmark_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_extraction(
            image_paths=image_paths,
            mode_name=mode_name,
            config=config,
            run_name=run_name,
        )
        for mode_name in mode_names
    ]

    benchmark_summary = {
        "run_name": run_name,
        "modes": mode_names,
        "documents": [str(path) for path in image_paths],
        "runs": summaries,
    }
    summary_path = benchmark_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(benchmark_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    benchmark_summary["summary_path"] = str(summary_path)
    return benchmark_summary

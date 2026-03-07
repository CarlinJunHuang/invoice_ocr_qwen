from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModeConfig:
    name: str
    kind: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    config_path: Path
    repo_root: Path
    project_root: Path
    output_root: Path
    hf_home: Path
    torch_home: Path
    logs_root: Path
    ocr_languages: list[str]
    ocr_gpu: bool
    ocr_min_confidence: float
    overlay_enabled: bool
    overlay_line_width: int
    overlay_font_size: int
    fuzzy_threshold: int
    resource_poll_seconds: float
    modes: dict[str, ModeConfig]

    def require_mode(self, mode_name: str) -> ModeConfig:
        try:
            return self.modes[mode_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.modes))
            raise KeyError(f"Unknown mode '{mode_name}'. Available modes: {available}") from exc


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_runtime_dirs(config: AppConfig) -> None:
    for path in (config.output_root, config.hf_home, config.torch_home, config.logs_root):
        path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    project_root = config_path.parent.parent.resolve()
    repo_root = project_root.parent.resolve()

    paths = raw.get("paths", {})
    ocr = raw.get("ocr", {})
    overlay = raw.get("overlay", {})
    grounding = raw.get("grounding", {})
    benchmark = raw.get("benchmark", {})

    modes = {
        name: ModeConfig(
            name=name,
            kind=payload["kind"],
            options={key: value for key, value in payload.items() if key != "kind"},
        )
        for name, payload in raw.get("modes", {}).items()
    }

    config = AppConfig(
        config_path=config_path,
        repo_root=repo_root,
        project_root=project_root,
        output_root=_resolve_path(project_root, paths.get("output_root", "./output")),
        hf_home=_resolve_path(project_root, paths.get("hf_home", "../runtime/hf-cache")),
        torch_home=_resolve_path(project_root, paths.get("torch_home", "../runtime/torch-cache")),
        logs_root=_resolve_path(project_root, paths.get("logs_root", "../runtime/logs")),
        ocr_languages=list(ocr.get("languages", ["en"])),
        ocr_gpu=bool(ocr.get("gpu", True)),
        ocr_min_confidence=float(ocr.get("min_confidence", 0.2)),
        overlay_enabled=bool(overlay.get("enabled", True)),
        overlay_line_width=int(overlay.get("line_width", 3)),
        overlay_font_size=int(overlay.get("font_size", 18)),
        fuzzy_threshold=int(grounding.get("fuzzy_threshold", 82)),
        resource_poll_seconds=float(benchmark.get("resource_poll_seconds", 0.25)),
        modes=modes,
    )

    ensure_runtime_dirs(config)
    return config

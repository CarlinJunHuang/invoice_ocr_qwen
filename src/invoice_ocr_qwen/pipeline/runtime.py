from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import psutil

from invoice_ocr_qwen.config import AppConfig


def apply_runtime_environment(config: AppConfig) -> None:
    os.environ["HF_HOME"] = str(config.hf_home)
    os.environ["HF_HUB_CACHE"] = str(config.hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(config.hf_home)
    os.environ["TORCH_HOME"] = str(config.torch_home)


@dataclass(slots=True)
class ResourceMetrics:
    wall_seconds: float
    peak_rss_mb: float
    peak_cuda_mb: float | None


class ResourceMonitor:
    def __init__(self, poll_seconds: float = 0.25) -> None:
        self.poll_seconds = poll_seconds
        self.process = psutil.Process()
        self.peak_rss = 0
        self._peak_cuda_bytes = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> "ResourceMonitor":
        self._started_at = time.perf_counter()
        self._running = True
        self._reset_torch_peak_memory()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds * 2)
        self._poll_once()

    def snapshot(self) -> ResourceMetrics:
        peak_cuda_mb = None
        if self._peak_cuda_bytes > 0:
            peak_cuda_mb = round(self._peak_cuda_bytes / (1024 * 1024), 2)
        return ResourceMetrics(
            wall_seconds=round(time.perf_counter() - self._started_at, 3),
            peak_rss_mb=round(self.peak_rss / (1024 * 1024), 2),
            peak_cuda_mb=peak_cuda_mb,
        )

    def _poll_loop(self) -> None:
        while self._running:
            self._poll_once()
            time.sleep(self.poll_seconds)

    def _poll_once(self) -> None:
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        self._peak_cuda_bytes = max(self._peak_cuda_bytes, self._read_cuda_peak_bytes())

    def _read_cuda_peak_bytes(self) -> int:
        try:
            import torch
        except ImportError:
            return 0

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated())

    def _reset_torch_peak_memory(self) -> None:
        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

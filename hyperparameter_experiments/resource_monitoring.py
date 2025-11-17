"""Resource monitoring utilities shared across experiment scripts."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import psutil

try:  # Optional dependency
    import GPUtil  # type: ignore

    _GPU_AVAILABLE = True
except ImportError:  # pragma: no cover - optional path
    GPUtil = None  # type: ignore
    _GPU_AVAILABLE = False


class ResourceMonitor:
    """Background resource monitor that samples CPU/GPU usage."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.monitoring = False
        self.samples: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.monitoring:
            return
        self.monitoring = True
        self.samples = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.monitoring = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _monitor_loop(self) -> None:
        while self.monitoring:
            self.samples.append(self._collect_sample())
            time.sleep(self.interval)

    def _collect_sample(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'cpu_per_core': psutil.cpu_percent(interval=None, percpu=True),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024 ** 3),
        }

        if _GPU_AVAILABLE and GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    snapshot['gpu_utilization'] = gpu.load * 100
                    snapshot['vram_used_mb'] = gpu.memoryUsed
                    snapshot['vram_total_mb'] = gpu.memoryTotal
                    snapshot['gpu_temp'] = gpu.temperature
            except Exception:  # pragma: no cover - defensive
                pass
        return snapshot

    def get_stats(self) -> Dict[str, Any]:
        if not self.samples:
            return {}

        import numpy as np

        cpu_percents = [s.get('cpu_percent', 0) for s in self.samples]
        ram_percents = [s.get('ram_percent', 0) for s in self.samples]

        stats: Dict[str, Any] = {
            'cpu_avg': float(np.mean(cpu_percents)),
            'cpu_max': float(np.max(cpu_percents)),
            'cpu_min': float(np.min(cpu_percents)),
            'ram_avg': float(np.mean(ram_percents)),
            'ram_max': float(np.max(ram_percents)),
            'sample_count': len(self.samples),
        }

        gpu_utils = [s.get('gpu_utilization') for s in self.samples if 'gpu_utilization' in s]
        if gpu_utils:
            stats['gpu_avg'] = float(np.mean(gpu_utils))
            stats['gpu_max'] = float(np.max(gpu_utils))
            stats['gpu_min'] = float(np.min(gpu_utils))

            vram_usages = [s.get('vram_used_mb', 0) for s in self.samples if 'vram_used_mb' in s]
            stats['vram_avg_mb'] = float(np.mean(vram_usages))
            stats['vram_max_mb'] = float(np.max(vram_usages))

        first_core_sample = self.samples[0].get('cpu_per_core')
        if first_core_sample:
            cpu_per_core_samples = [s['cpu_per_core'] for s in self.samples if 'cpu_per_core' in s]
            core_array = np.array(cpu_per_core_samples)
            stats['cpu_per_core_avg'] = [float(x) for x in core_array.mean(axis=0)]
            stats['cpu_per_core_max'] = [float(x) for x in core_array.max(axis=0)]

        return stats


def get_snapshot_metrics() -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        'cpu_percent': psutil.cpu_percent(interval=None),
        'cpu_count': psutil.cpu_count(),
        'ram_percent': psutil.virtual_memory().percent,
        'ram_used_gb': psutil.virtual_memory().used / (1024 ** 3),
        'ram_total_gb': psutil.virtual_memory().total / (1024 ** 3),
    }

    if _GPU_AVAILABLE and GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['vram_used_mb'] = gpu.memoryUsed
                metrics['vram_total_mb'] = gpu.memoryTotal
                metrics['vram_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
        except Exception:  # pragma: no cover - defensive
            metrics['gpu_available'] = False
    else:
        metrics['gpu_available'] = False

    return metrics


def aggregate_resource_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    filtered = [s for s in samples if s]
    if not filtered:
        return {}

    import numpy as np

    def _collect(key: str) -> List[float]:
        return [s[key] for s in filtered if key in s]

    aggregated: Dict[str, Any] = {}

    cpu_avgs = _collect('cpu_avg')
    cpu_maxes = _collect('cpu_max')
    if cpu_avgs:
        aggregated['cpu_avg'] = float(np.mean(cpu_avgs))
    if cpu_maxes:
        aggregated['cpu_max'] = float(np.max(cpu_maxes))

    ram_avgs = _collect('ram_avg')
    ram_maxes = _collect('ram_max')
    if ram_avgs:
        aggregated['ram_avg'] = float(np.mean(ram_avgs))
    if ram_maxes:
        aggregated['ram_max'] = float(np.max(ram_maxes))

    gpu_avgs = _collect('gpu_avg')
    gpu_maxes = _collect('gpu_max')
    if gpu_avgs:
        aggregated['gpu_avg'] = float(np.mean(gpu_avgs))
    if gpu_maxes:
        aggregated['gpu_max'] = float(np.max(gpu_maxes))

    vram_avgs = _collect('vram_avg_mb')
    vram_maxes = _collect('vram_max_mb')
    if vram_avgs:
        aggregated['vram_avg_mb'] = float(np.mean(vram_avgs))
    if vram_maxes:
        aggregated['vram_max_mb'] = float(np.max(vram_maxes))

    sample_counts = _collect('sample_count')
    if sample_counts:
        aggregated['sample_count'] = int(sum(sample_counts))

    return aggregated
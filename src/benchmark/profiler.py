"""Hardware profiling utilities.

Collects CPU/GPU utilization, memory usage patterns, and thermal data
for mobile scenario simulation.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ProfileSample:
    """A single profiling sample at a point in time."""

    timestamp: float
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class ProfileResult:
    """Complete profiling result for a function execution."""

    samples: list[ProfileSample] = field(default_factory=list)
    duration_sec: float = 0.0
    peak_memory_mb: float = 0.0
    mean_cpu_percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "duration_sec": self.duration_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "mean_cpu_percent": self.mean_cpu_percent,
            "num_samples": len(self.samples),
            "timeline": [
                {
                    "timestamp": s.timestamp,
                    "cpu_percent": s.cpu_percent,
                    "memory_mb": s.memory_mb,
                }
                for s in self.samples
            ],
        }


class HardwareProfiler:
    """Profiles CPU, memory, and GPU usage during function execution.

    Args:
        interval_sec: Sampling interval in seconds.

    Usage:
        profiler = HardwareProfiler(interval_sec=0.5)
        result = profiler.profile(my_function, arg1, arg2)
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self._interval = interval_sec
        self._samples: list[ProfileSample] = []
        self._running = False

    def profile(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> ProfileResult:
        """Profile a function's hardware usage.

        Args:
            func: Function to profile.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            ProfileResult with timeline and summary statistics.
        """
        self._samples = []
        self._running = True

        # Start sampling thread
        sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        sample_thread.start()

        start_time = time.perf_counter()
        try:
            func(*args, **kwargs)
        finally:
            self._running = False
            sample_thread.join(timeout=self._interval * 2)

        duration = time.perf_counter() - start_time

        # Compute summary
        result = ProfileResult(
            samples=self._samples,
            duration_sec=duration,
        )

        if self._samples:
            result.peak_memory_mb = max(s.memory_mb for s in self._samples)
            result.mean_cpu_percent = (
                sum(s.cpu_percent for s in self._samples) / len(self._samples)
            )

        logger.info(
            "Profile complete: %.1fs, peak memory %.1f MB, mean CPU %.1f%%",
            result.duration_sec,
            result.peak_memory_mb,
            result.mean_cpu_percent,
        )
        return result

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        import os

        start = time.perf_counter()
        while self._running:
            sample = ProfileSample(timestamp=time.perf_counter() - start)

            try:
                import psutil

                process = psutil.Process(os.getpid())
                sample.cpu_percent = process.cpu_percent()
                sample.memory_mb = process.memory_info().rss / (1024 * 1024)
            except ImportError:
                # Fallback: read from /proc on Linux
                sample.memory_mb = self._read_proc_memory()

            self._samples.append(sample)
            time.sleep(self._interval)

    @staticmethod
    def _read_proc_memory() -> float:
        """Read current process memory from /proc/self/status (Linux only)."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # KB to MB
        except (FileNotFoundError, ValueError):
            pass
        return 0.0

# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# --- Memory sampler: CPU resident set size (RSS) + GPU VRAM (NVML) -------------------
import os, time, threading
import psutil
import drjit as dr

try:
    import pynvml as nvml

    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


class MemSampler:
    """Poll CPU resident set size (RSS) and (optionally) GPU VRAM; records peaks."""

    def __init__(
        self,
        gpu_index: int = 0,
        interval: float = 0.05,
    ):
        self.gpu_index = gpu_index
        self.interval = interval
        self.pid = os.getpid()
        self.rss_peak_mb = 0.0
        self.vram_peak_mb = None
        self._stop = threading.Event()
        self._t = None
        self._proc = psutil.Process(self.pid)
        self._handle = None

    def __enter__(self):
        if _HAS_NVML:
            try:
                nvml.nvmlInit()
                self._handle = nvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:
                self._handle = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            # CPU RSS (MB)
            try:
                rss_mb = self._proc.memory_info().rss / (1024**2)
                if rss_mb > self.rss_peak_mb:
                    self.rss_peak_mb = rss_mb
            except psutil.Error:
                pass

            # GPU VRAM (MB)
            if _HAS_NVML and self._handle is not None:
                try:
                    mem = nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    mb = mem.used / (1024**2)
                    self.vram_peak_mb = max(self.vram_peak_mb or 0.0, mb)
                except Exception:
                    pass

            time.sleep(self.interval)

    def __exit__(self, exc_type, exc, tb):
        # Ensure GPU work finished before stopping the sampler
        try:
            dr.sync_thread()
        except Exception:
            pass
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        if _HAS_NVML and self._handle is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def render_with_mem(fn, *, gpu_index: int = 0):
    """
    Execute `fn()` while sampling memory. Returns (result, cpu_rss_peak_mb, gpu_vram_peak_mb or None).
    """
    with MemSampler(gpu_index=gpu_index) as ms:
        result = fn()
    return result, ms.rss_peak_mb, ms.vram_peak_mb

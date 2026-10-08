"""通用工具函数。"""
from __future__ import annotations

import gc
import sys


def free_memory() -> None:
    """回收已释放的 Python 对象，并清空 CUDA 缓存，把显存/内存还给系统。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def cuda_allocated_mb() -> int:
    """返回当前进程占用的 CUDA 显存（MB），非 CUDA 环境返回 0。"""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() // (1024 * 1024))
    except Exception:  # noqa: BLE001
        pass
    return 0


class ProgressBar:
    """简单的控制台进度条（ASCII 字符，避免 Windows GBK 编码问题）。

    用法：
        bar = ProgressBar(total=100, label="编码")
        for i in range(100):
            ...
            bar.update(i + 1)
        bar.close()
    """

    def __init__(self, total: int, label: str = "", width: int = 40) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.width = max(10, int(width))
        self.done = 0

    def update(self, done: int = None, total: int = None) -> None:
        if total is not None:
            self.total = max(1, int(total))
        if done is None:
            self.done += 1
        else:
            self.done = int(done)
        frac = min(1.0, self.done / self.total)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(
            f"\r{self.label} [{bar}] {self.done}/{self.total} ({frac * 100:5.1f}%)"
        )
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

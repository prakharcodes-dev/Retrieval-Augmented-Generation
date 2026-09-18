"""
Centralized Resource Guard for resource-aware, crash-free execution.
Monitors system memory (RAM), process memory (RSS), and GPU VRAM before expensive operations.
Prevents OS-level OOM kills by halting un-viable operations gracefully.
"""

import gc
import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
from app.core.exceptions import ResourceUnsafeError

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


class ResourceGuard:
    """
    Centralized monitor for RAM, RSS, VRAM, and workload parameters.
    Configurable thresholds enforce pre-flight safety checks.
    """

    MIN_AVAILABLE_RAM_MB: float = 300.0  # Minimum required free system RAM in MB
    MAX_RAM_PERCENT_THRESHOLD: float = 92.0  # Maximum system RAM usage percentage allowed
    MIN_AVAILABLE_VRAM_MB: float = 500.0  # Minimum required GPU VRAM in MB for local GPU model
    MAX_FILE_SIZE_MB: float = 100.0  # Hard upper boundary for single file upload

    @classmethod
    def get_memory_stats(cls) -> Dict[str, Any]:
        """Returns current system RAM, RSS, and GPU VRAM statistics."""
        stats: Dict[str, Any] = {
            "available_ram_mb": 1024.0,
            "total_ram_mb": 4096.0,
            "ram_percent_used": 50.0,
            "process_rss_mb": 100.0,
            "gpu_available": False,
            "free_vram_mb": 0.0,
            "total_vram_mb": 0.0,
        }

        if psutil is not None:
            try:
                mem = psutil.virtual_memory()
                stats["available_ram_mb"] = round(mem.available / (1024 * 1024), 2)
                stats["total_ram_mb"] = round(mem.total / (1024 * 1024), 2)
                stats["ram_percent_used"] = round(mem.percent, 2)

                proc = psutil.Process(os.getpid())
                stats["process_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Failed to fetch psutil memory stats: {e}")

        if torch is not None and torch.cuda.is_available():
            try:
                stats["gpu_available"] = True
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                stats["free_vram_mb"] = round(free_bytes / (1024 * 1024), 2)
                stats["total_vram_mb"] = round(total_bytes / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Failed to fetch CUDA VRAM stats: {e}")

        return stats

    @classmethod
    def check_memory(
        cls,
        min_available_mb: Optional[float] = None,
        max_ram_percent: Optional[float] = None,
        context: str = "operation"
    ) -> Dict[str, Any]:
        """
        Validates system memory before performing an operation.
        Raises ResourceUnsafeError if system memory is dangerously low.
        """
        min_ram = min_available_mb if min_available_mb is not None else cls.MIN_AVAILABLE_RAM_MB
        max_pct = max_ram_percent if max_ram_percent is not None else cls.MAX_RAM_PERCENT_THRESHOLD

        stats = cls.get_memory_stats()
        avail_ram = stats["available_ram_mb"]
        ram_pct = stats["ram_percent_used"]

        if avail_ram < min_ram or ram_pct > max_pct:
            cls.collect_garbage()
            # Re-check after garbage collection
            stats = cls.get_memory_stats()
            avail_ram = stats["available_ram_mb"]
            ram_pct = stats["ram_percent_used"]
            if avail_ram < min_ram or ram_pct > max_pct:
                raise ResourceUnsafeError(
                    f"Insufficient system memory for {context}. Available RAM: {avail_ram:.1f} MB "
                    f"(required >= {min_ram:.1f} MB, usage: {ram_pct:.1f}%)."
                )

        return stats

    @classmethod
    def check_pdf_workload(
        cls,
        file_size_bytes: int,
        estimated_pages: Optional[int] = None,
        filename: str = "Document"
    ) -> None:
        """Checks PDF file size and page workload against available system resources."""
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > cls.MAX_FILE_SIZE_MB:
            raise ResourceUnsafeError(
                f"File '{filename}' size ({file_size_mb:.1f} MB) exceeds maximum safe file threshold "
                f"of {cls.MAX_FILE_SIZE_MB:.1f} MB."
            )

        stats = cls.get_memory_stats()
        avail_ram = stats["available_ram_mb"]

        # Check if file size is dangerously close to total available RAM
        if file_size_mb > (avail_ram * 0.4):
            raise ResourceUnsafeError(
                f"File '{filename}' ({file_size_mb:.1f} MB) requires more RAM than currently safe "
                f"(Available: {avail_ram:.1f} MB)."
            )

    @classmethod
    def calculate_safe_batch_size(
        cls,
        total_items: int,
        default_batch_size: int = 20,
        min_batch_size: int = 5,
        max_batch_size: int = 100
    ) -> int:
        """
        Dynamically calculates a safe batch size based on available system memory.
        """
        stats = cls.get_memory_stats()
        avail_ram = stats["available_ram_mb"]

        if avail_ram < 500:
            batch_size = min_batch_size
        elif avail_ram < 1500:
            batch_size = min(default_batch_size, 20)
        elif avail_ram > 4000:
            batch_size = min(max_batch_size, 50)
        else:
            batch_size = default_batch_size

        return max(min_batch_size, min(batch_size, total_items if total_items > 0 else default_batch_size))

    @classmethod
    def check_llm_loading_workload(cls, prefer_gpu: bool = True) -> Tuple[bool, str]:
        """
        Checks system resources before loading an LLM model.
        Returns tuple: (use_gpu, reason_message)
        """
        stats = cls.get_memory_stats()
        avail_ram = stats["available_ram_mb"]

        if prefer_gpu and stats["gpu_available"]:
            free_vram = stats["free_vram_mb"]
            if free_vram >= cls.MIN_AVAILABLE_VRAM_MB:
                return True, f"GPU available with {free_vram:.1f} MB VRAM."
            else:
                logger.warning(f"GPU VRAM low ({free_vram:.1f} MB free). Falling back to CPU.")

        if avail_ram < 1000.0:
            raise ResourceUnsafeError(
                f"Insufficient memory to load LLM model (Available RAM: {avail_ram:.1f} MB, required >= 1000 MB)."
            )

        return False, f"CPU deployment selected (Available RAM: {avail_ram:.1f} MB)."

    @classmethod
    def collect_garbage(cls) -> None:
        """Executes garbage collection safely to release unreferenced objects."""
        try:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during garbage collection: {e}")

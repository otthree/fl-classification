"""Memory monitoring utilities for federated learning."""

import gc
import os
import time
from typing import Dict, List, Optional

import psutil
import torch


class MemoryMonitor:
    """Monitor GPU and CPU memory usage during federated learning."""

    def __init__(self, client_id: Optional[str] = None):
        """Initialize memory monitor.

        Args:
            client_id: Optional client ID for logging
        """
        self.client_id = client_id or "unknown"
        self.memory_history: List[Dict] = []
        self.start_time = time.time()

    def log_memory_usage(self, stage: str, round_num: Optional[int] = None, epoch: Optional[int] = None):
        """Log current memory usage.

        Args:
            stage: Description of current stage (e.g., "before_training", "after_epoch")
            round_num: Current FL round number
            epoch: Current epoch number
        """
        timestamp = time.time() - self.start_time

        memory_info = {
            "timestamp": timestamp,
            "stage": stage,
            "client_id": self.client_id,
            "round": round_num,
            "epoch": epoch,
        }

        # GPU memory info
        if torch.cuda.is_available():
            memory_info.update(
                {
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
                }
            )

        # CPU memory info
        process = psutil.Process(os.getpid())
        memory_info.update(
            {"cpu_memory_gb": process.memory_info().rss / 1024**3, "cpu_memory_percent": process.memory_percent()}
        )

        self.memory_history.append(memory_info)

        # Print summary
        gpu_str = ""
        if torch.cuda.is_available():
            gpu_str = f"GPU: {memory_info['gpu_allocated_gb']:.2f}/{memory_info['gpu_reserved_gb']:.2f}GB, "

        print(
            f"Memory [{self.client_id}] {stage}: "
            f"{gpu_str}CPU: {memory_info['cpu_memory_gb']:.2f}GB ({memory_info['cpu_memory_percent']:.1f}%)"
        )

    def reset_peak_memory_stats(self):
        """Reset peak memory statistics for PyTorch."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_memory_summary(self) -> Dict:
        """Get summary of memory usage throughout monitoring.

        Returns:
            Dictionary with memory usage statistics
        """
        if not self.memory_history:
            return {}

        summary = {
            "client_id": self.client_id,
            "total_entries": len(self.memory_history),
            "duration_seconds": self.memory_history[-1]["timestamp"] - self.memory_history[0]["timestamp"],
        }

        if torch.cuda.is_available():
            gpu_allocated = [entry["gpu_allocated_gb"] for entry in self.memory_history]
            gpu_reserved = [entry["gpu_reserved_gb"] for entry in self.memory_history]

            summary.update(
                {
                    "gpu_allocated_min_gb": min(gpu_allocated),
                    "gpu_allocated_max_gb": max(gpu_allocated),
                    "gpu_allocated_final_gb": gpu_allocated[-1],
                    "gpu_reserved_min_gb": min(gpu_reserved),
                    "gpu_reserved_max_gb": max(gpu_reserved),
                    "gpu_reserved_final_gb": gpu_reserved[-1],
                }
            )

        cpu_memory = [entry["cpu_memory_gb"] for entry in self.memory_history]
        summary.update(
            {
                "cpu_memory_min_gb": min(cpu_memory),
                "cpu_memory_max_gb": max(cpu_memory),
                "cpu_memory_final_gb": cpu_memory[-1],
            }
        )

        return summary

    def force_cleanup(self):
        """Force memory cleanup."""
        try:
            # Clear gradients if possible
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.log_memory_usage("after_force_cleanup")

        except Exception as e:
            print(f"Warning: Force cleanup failed: {e}")

    def save_memory_history(self, filename: str):
        """Save memory history to file.

        Args:
            filename: Path to save the memory history
        """
        try:
            import json

            with open(filename, "w") as f:
                json.dump(self.memory_history, f, indent=2)
            print(f"Saved memory history to {filename}")
        except Exception as e:
            print(f"Error saving memory history: {e}")


def create_memory_monitor(client_id: Optional[str] = None) -> MemoryMonitor:
    """Create a memory monitor instance.

    Args:
        client_id: Optional client ID for logging

    Returns:
        MemoryMonitor instance
    """
    return MemoryMonitor(client_id)


def log_gpu_memory_summary(prefix: str = "GPU Memory"):
    """Log a quick GPU memory summary.

    Args:
        prefix: Prefix for the log message
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{prefix}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
    else:
        print(f"{prefix}: CUDA not available")

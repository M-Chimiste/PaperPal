"""Shared helpers for task handlers (extracted from TaskManager, B6)."""
from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
import json

from ...data_access import SettingsRepository
from ..tasks import TaskStatus

if TYPE_CHECKING:
    from ..tasks import TaskManager


def progress_callback(task_manager: "TaskManager", task_id: str):
    """Create a progress callback function for TheseusInsight."""
    loop = asyncio.get_event_loop()

    def callback(stage: str, progress: float, message: str = "", metadata=None):
        coro = task_manager.update_task_status(
            task_id=task_id,
            status=TaskStatus.PROCESSING,
            message=f"{stage}: {message}",
            progress=progress,
            current_step=stage,
            metadata=metadata,
        )
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            # This case might occur if the main loop is stopped or not accessible.
            # For robustness, you could log this or handle it differently.
            # For now, we attempt to run it in a new loop, though this has implications.
            try:
                asyncio.run(coro) # Fallback, but be cautious with this approach.
            except RuntimeError as e:
                print(f"RuntimeError in progress_callback fallback: {e}. Status update for '{stage}' might be lost.")
    return callback


def get_orchestration_config(verbose: bool = False) -> dict:
    """Compat shim — canonical implementation moved to theseus_insight.config (B10)."""
    from ...config import get_orchestration_config as _impl
    return _impl(verbose=verbose)

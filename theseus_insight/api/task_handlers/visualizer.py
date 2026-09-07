"""Task handler(s) extracted from TaskManager (refactor B6): run_visualizer_task."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, List
import asyncio
import json
import os
from datetime import datetime

from ..tasks import TaskStatus
from ...data_access import (
    TaskRepository, LogsRepository, SettingsRepository,
    PaperRepository, PaperFulltextRepository
)
from ._common import get_orchestration_config, progress_callback

if TYPE_CHECKING:
    from ..tasks import TaskManager


async def run(task_manager: "TaskManager", task_id: str):
    """Run the audio visualization task."""
    try:
        task = TaskRepository.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        config = task["config_json"]
        audio_file_path = config.get("audio_file_path")
        visualizer_params_dict = config.get("visualizer_params")
        output_dir_base = config.get("output_dir_base", "data/visualizations")

        if not audio_file_path or not os.path.exists(audio_file_path):
            raise ValueError(f"Audio file not found at path: {audio_file_path}")
        if not visualizer_params_dict:
            raise ValueError("Visualizer parameters are missing from task config.")

        await task_manager.update_task_status(
            task_id,
            TaskStatus.PROCESSING,
            "Starting visualization generation",
            current_step="visualizer_init",
        )

        output_dir = os.path.join(output_dir_base, task_id)
        os.makedirs(output_dir, exist_ok=True)

        final_video_filename = f"visualization_{task_id}.mp4"
        final_video_path = os.path.join(output_dir, final_video_filename)

        # Import locally to avoid circular dependencies if any, and keep generator specific
        from ...podcast.generator import generate_visualizer_video

        # Convert visualizer_params_dict to appropriate arguments for generate_visualizer_video
        # The generate_visualizer_video function expects individual arguments.
        vis_params = {
            "audio_filepath": audio_file_path,
            "output_filepath": final_video_path,
            "resolution": (visualizer_params_dict.get('resolution_width', 1920), visualizer_params_dict.get('resolution_height', 1080)),
            "fps": visualizer_params_dict.get('fps', 30),
            "matrix_count": visualizer_params_dict.get('matrix_count', 150),
            "matrix_head_color": visualizer_params_dict.get('matrix_head_color', "#e0ffe7"),
            "matrix_tail_color": visualizer_params_dict.get('matrix_tail_color', "#00b000"),
            "matrix_char_size": visualizer_params_dict.get('matrix_char_size', 24),
            "head_step_time": visualizer_params_dict.get('head_step_time', 0.3),
            "random_x_jitter": visualizer_params_dict.get('random_x_jitter', 3.0),
            "fade_time": visualizer_params_dict.get('fade_time', 3.0),
            "head_glow_passes": visualizer_params_dict.get('head_glow_passes', 3),
            "head_glow_alpha_decay": visualizer_params_dict.get('head_glow_alpha_decay', 50),
            "head_spawn_delay_range": (
                visualizer_params_dict.get('head_spawn_delay_range_min', 1.0),
                visualizer_params_dict.get('head_spawn_delay_range_max', 3.0)
            ),
            "head_saw_period": visualizer_params_dict.get('head_saw_period', 1.5),
            "line_width": visualizer_params_dict.get('line_width', 3),
            "wave_color": visualizer_params_dict.get('wave_color', "#d703fc"),
            "trail_colors": [
                visualizer_params_dict.get('trail_color_1', "#fc03b6"),
                visualizer_params_dict.get('trail_color_2', "#ba03fc"),
                visualizer_params_dict.get('trail_color_3', "#ce6bf2")
            ],
            "glow_passes": visualizer_params_dict.get('glow_passes', 3),
            "glow_alpha_decay": visualizer_params_dict.get('glow_alpha_decay', 40),
            "font_path": visualizer_params_dict.get('font_path', "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")
        }

        # Progress callback for generate_visualizer_video if it supports it
        # For now, just running it in a thread.
        await asyncio.to_thread(generate_visualizer_video, **vis_params)

        result = {
            "visualizer_file": final_video_path # Ensure this key matches what main.py expects for video downloads
        }

        await task_manager.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            "Visualization generated successfully",
            progress=100,
            current_step="visualizer_complete",
            result=result,
        )

    except Exception as e:
        import traceback
        error_details = f"Error in visualizer task: {str(e)}\n{traceback.format_exc()}"
        await task_manager.update_task_status(
            task_id,
            TaskStatus.FAILED,
            "Visualization generation failed",
            error=error_details,
            current_step="visualizer_failed",
        )
        print(error_details) # Log to server console
        raise

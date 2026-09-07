"""Task handler(s) extracted from TaskManager (refactor B6): run_podcast_task."""
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
    """Run the podcast generation task."""
    try:
        task = TaskRepository.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        config = task["config_json"]

        # Initialize podcast generator
        # The text_model parameter expects the entire model configuration dictionary.
        # The key for this dictionary in the config is "podcast_model_config".
        podcast_model_configuration = config.get("podcast_model_config")
        if not podcast_model_configuration:
            raise ValueError("Podcast model configuration (podcast_model_config) is missing from task config.")

        from ...podcast.generator import PodcastGenerator
        podcast_gen = PodcastGenerator(
            text_model=podcast_model_configuration, # Use the correct key here
            tts_provider=config.get("tts_model_config", {}).get("tts_provider", "openai"),
            speaker_1_voice=config.get("tts_model_config", {}).get("speaker_1_voice", "sage"),
            speaker_2_voice=config.get("tts_model_config", {}).get("speaker_2_voice", "ash"),
            speaker_1_speed=config.get("tts_model_config", {}).get("speaker_1_speed", 1.0),
            speaker_2_speed=config.get("tts_model_config", {}).get("speaker_2_speed", 1.0),
            intro_music_path=config.get("intro_music_path", None),
            verbose=config.get("verbose", True), # Added verbose from config
            db_url=config.get("data_path", None)  # Pass the database URL
        )

        await task_manager.update_task_status(
            task_id,
            TaskStatus.PROCESSING,
            "Starting podcast generation",
            current_step="podcast_init",
        )

        # Get input sources from config
        input_type = config.get("input_type") # Should be present, validated by Pydantic model
        if not input_type: # Defensive check
            raise ValueError("input_type is missing from task config.")

        urls_to_process = config.get("urls", [])
        input_pdf_paths = config.get("input_pdf_paths", [])

        final_input_paths = []
        if input_type == "URLs": # Matches what api_client sends
            if not urls_to_process:
                # It's valid to have URLs type with no URLs yet, user might be about to input them
                # or it's an error if pipeline starts. For now, assume it's an error if empty at execution.
                raise ValueError("No URLs provided for podcast generation when input_type is 'URLs'.")
            final_input_paths = urls_to_process
        elif input_type == "pdfs": # Matches what api_client sends
            if not input_pdf_paths:
                raise ValueError("No PDF paths provided for podcast generation when input_type is 'pdfs'.")
            final_input_paths = input_pdf_paths
        else:
            # This case should ideally not be reached if PodcastGenerationParams validates input_type
            raise ValueError(f"Unsupported input_type for podcast received in task: {input_type}")

        # Create output directories
        # Use task_id in output_dir_base for better organization as done in main.py
        output_dir_base = config.get("output_dir_base", "data/podcasts")
        output_dir = os.path.join(output_dir_base, task_id) 
        os.makedirs(output_dir, exist_ok=True)

        # Generate podcast with progress tracking
        # Ensure the progress_callback in PodcastGenerator is compatible or adapt here
        # For now, assuming PodcastGenerator.generate_podcast uses a simple callback structure if any.
        # The callback here is for the TaskManager, not directly for PodcastGenerator stages if it has finer-grained ones.

        visualizer_settings = config.get("visualizer_settings")
        create_visualization = config.get("create_visualization", False)

        # Run podcast generation
        # The generate_podcast method in the stub takes pdf_paths. 
        # We should ensure it can handle URLs or pre-processed PDF paths based on input_type.
        # For now, passing final_input_paths which could be URLs or local PDF paths.
        # PodcastGenerator needs to handle this logic internally.
        result = await asyncio.to_thread(
            podcast_gen.generate_podcast,
            pdf_paths=final_input_paths, # This is now a list of URLs or local PDF paths
            output_dir=output_dir,
            prefix=f"podcast_{task_id}", 
            final_filename=f"podcast_{task_id}_final",
            visualizer=create_visualization,
            # Pass specific visualizer settings if available and visualizer is True
            **(visualizer_settings if create_visualization and visualizer_settings else {}),
            progress_callback=progress_callback(task_manager, task_id) # If PodcastGenerator takes a callback
        )

        # Result will be stored via update_task_status call
        await task_manager.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            "Podcast generated successfully",
            progress=100,
            current_step="podcast_complete",
            result=result,
        )

    except Exception as e:
        await task_manager.update_task_status(
            task_id,
            TaskStatus.FAILED,
            "Podcast generation failed",
            error=str(e),
            current_step="podcast_failed",
        )
        raise

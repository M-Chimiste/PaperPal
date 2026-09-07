"""Task handler(s) extracted from TaskManager (refactor B6): run_newsletter_task."""
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
from ...theseus_insight import TheseusInsight

if TYPE_CHECKING:
    from ..tasks import TaskManager


async def run(task_manager: "TaskManager", task_id: str):
    """Run the newsletter generation task."""
    try:
        task = TaskRepository.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        config = task["config_json"]
        email_recipients = config.get("emailRecipients", None)
        research_interests_override = config.get("researchInterests", None)

        # Profile filtering parameters
        profile_id = config.get("profile_id")
        profile_ids = config.get("profile_ids")
        profile_tag = config.get("profile_tag")
        profile_tags = config.get("profile_tags")
        use_profile_recipients = config.get("use_profile_recipients", False)

        # Resolve profile context
        resolved_profile_ids = None
        profile_recipients = None

        if profile_id or profile_ids or profile_tag or profile_tags:
            from ...data_access import ProfileRepository

            # Resolve profile IDs from tags if provided
            if profile_tag or profile_tags:
                tag_list = []
                if profile_tag:
                    tag_list.append(profile_tag)
                if profile_tags:
                    tag_list.extend(profile_tags)

                tag_profiles = ProfileRepository.get_by_tags(tag_list)
                tag_profile_ids = [p['id'] for p in tag_profiles]

                if profile_ids:
                    # Combine explicit profile_ids with tag-resolved IDs
                    resolved_profile_ids = list(set(profile_ids + tag_profile_ids))
                else:
                    resolved_profile_ids = tag_profile_ids
            elif profile_ids:
                resolved_profile_ids = profile_ids
            elif profile_id:
                resolved_profile_ids = [profile_id]

            # Get profile-specific email recipients if requested
            if use_profile_recipients and resolved_profile_ids:
                profile_recipients = []
                for pid in resolved_profile_ids:
                    profile = ProfileRepository.get_by_id(pid)
                    if profile and profile.get('email_recipients'):
                        profile_recipients.extend(profile['email_recipients'])

                # Remove duplicates while preserving order
                profile_recipients = list(dict.fromkeys(profile_recipients))

                # Use profile recipients if available, otherwise fall back to config recipients
                if profile_recipients:
                    email_recipients = profile_recipients

        orchestration_config = SettingsRepository.get("orchestration")
        if not orchestration_config:
            orchestration_config = "config/orchestration.json"

        # Create TheseusInsight instance for newsletter generation
        ti = TheseusInsight(
            research_interests_override=research_interests_override,
            orchestration_config=orchestration_config,
            task_id=task_id,
            checkpoint_dir=os.path.join("data", "checkpoints", task_id),
            progress_callback=progress_callback(task_manager, task_id),
            profile_ids_override=resolved_profile_ids,  # Pass resolved profile IDs
            top_n=config.get("num_sections", config.get("max_papers_to_select", 5)),
            start_date_override=config.get("start_date_override", config.get("start_date")),
            end_date_override=config.get("end_date_override", config.get("end_date")),
            generate_podcast=False,  # We handle podcast generation separately for now
            data_path=os.getenv("DATABASE_URL", "postgresql://theseus:theseus@localhost:5432/theseusdb"),
            generate_email=config.get("send_email", True),
            receiver_address_override=email_recipients,
            verbose=True
        )

        # Run the pipeline with progress tracking
        await task_manager.update_task_status(
            task_id,
            TaskStatus.PROCESSING,
            "Starting newsletter generation",
            current_step="initializing",
        )

        # Create progress callback
        callback = progress_callback(task_manager, task_id)

        # Run the pipeline
        result = await asyncio.to_thread(
            ti.run,
            progress_callback=callback
        )

        # Result will be stored via update_task_status call
        await task_manager.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            "Newsletter generated successfully",
            progress=100,
            current_step="newsletter_complete",
            result=result,
        )

    except Exception as e:
        await task_manager.update_task_status(
            task_id,
            TaskStatus.FAILED,
            "Newsletter generation failed",
            error=str(e),
            current_step="newsletter_failed",
        )
        raise

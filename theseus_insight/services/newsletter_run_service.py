"""Custom newsletter pipeline runs (extracted from the
newsletters_and_podcasts router in refactor B7).

Owns everything that is not HTTP-shaped: profile/recipient resolution,
orchestration-config loading, multi-server newsletter-job creation,
TheseusInsight instantiation, and the cross-thread progress callback.
The router keeps request validation and task enqueueing.
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..api.tasks import task_manager, TaskStatus
from ..theseus_insight import TheseusInsight


def resolve_profiles_and_recipients(params) -> Tuple[Optional[List[int]], Optional[List[str]]]:
    """Resolve the run's profile ids and effective email recipients.

    Tag-resolved ids are UNIONED with explicit profile_ids here (unlike the
    papers/trends filters, which intersect) — preserved from the original
    endpoint logic.
    """
    resolved_profile_ids = None
    final_email_recipients = params.email_recipients

    if params.profile_id or params.profile_ids or params.profile_tag or params.profile_tags:
        from ..data_access import ProfileRepository

        if params.profile_tag or params.profile_tags:
            tag_list = []
            if params.profile_tag:
                tag_list.append(params.profile_tag)
            if params.profile_tags:
                tag_list.extend(params.profile_tags)

            tag_profiles = ProfileRepository.get_by_tags(tag_list)
            tag_profile_ids = [p['id'] for p in tag_profiles]

            if params.profile_ids:
                # Combine explicit profile_ids with tag-resolved IDs
                resolved_profile_ids = list(set(params.profile_ids + tag_profile_ids))
            else:
                resolved_profile_ids = tag_profile_ids
        elif params.profile_ids:
            resolved_profile_ids = params.profile_ids
        elif params.profile_id:
            resolved_profile_ids = [params.profile_id]

        # Use profile-specific email recipients if requested
        if params.use_profile_recipients and resolved_profile_ids:
            profile_recipients = []
            for pid in resolved_profile_ids:
                profile = ProfileRepository.get_by_id(pid)
                if profile and profile.get('email_recipients'):
                    profile_recipients.extend(profile['email_recipients'])

            # Remove duplicates while preserving order
            profile_recipients = list(dict.fromkeys(profile_recipients))

            if profile_recipients:
                final_email_recipients = profile_recipients

    return resolved_profile_ids, final_email_recipients


def make_pipeline_progress_callback(task_id: str, loop: asyncio.AbstractEventLoop):
    """Build the stage-progress callback TheseusInsight invokes.

    The pipeline runs in a worker thread (asyncio.to_thread), so status
    updates must hop back to the captured main loop.
    """

    def pipeline_progress_callback(
        stage: str,
        progress_val: float,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        status_detail = f"Stage: {stage} - {message} ({progress_val:.2f}%)"
        overall_status_for_tm = TaskStatus.PROCESSING
        if stage.lower() == "newsletter_complete" and progress_val >= 100.0:
            overall_status_for_tm = TaskStatus.COMPLETED

        async def update_status_async():
            await task_manager.update_task_status(
                task_id,
                overall_status_for_tm,
                message=status_detail,
                progress=progress_val,
                current_step=stage,
                metadata=metadata,
            )

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            asyncio.create_task(update_status_async())
        else:
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(update_status_async(), loop)
            else:
                print("Error: Main event loop is not running, cannot update task status")

    return pipeline_progress_callback


async def run_custom_newsletter(params, task_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """Execute a custom newsletter pipeline run as a background task.

    Note: this runs after the HTTP response was sent; exceptions mark the
    task failed rather than producing an error response.
    """
    run_db_path = os.getenv("DATABASE_URL", "postgresql://theseus:theseus@localhost:5432/theseusdb")
    try:
        await task_manager.update_task_status(
            task_id,
            TaskStatus.PENDING,
            message="Pipeline initialized.",
            current_step="initializing",
        )

        resolved_profile_ids, final_email_recipients = resolve_profiles_and_recipients(params)

        # Load orchestration config from database settings
        from ..data_access.settings import SettingsRepository
        orchestration_config_json = SettingsRepository.get("orchestration")
        if orchestration_config_json:
            orchestration_config = json.loads(orchestration_config_json)
        else:
            # Fallback to config file if not in database
            orchestration_config = "config/orchestration.json"

        # Create newsletter job if using multi-server judge
        newsletter_job_id = None
        judge_servers = None
        if params.use_multi_server_judge:
            from ..data_access.newsletters import NewsletterJobRepository

            if params.judge_server_ids:
                from ..data_access.inference_servers import InferenceServersRepository
                servers = InferenceServersRepository.get_by_ids(params.judge_server_ids)
                enabled_servers = [s for s in servers if s.enabled]
                if not enabled_servers:
                    # Raised post-response: fails the task via the except below.
                    raise ValueError("No enabled servers found in selection")
                judge_servers = enabled_servers

            newsletter_job_id = NewsletterJobRepository.create_job(
                profile_ids=resolved_profile_ids or [],
                use_multi_server=True,
                server_ids=params.judge_server_ids,
                research_interests=params.research_interests,
                date_range_start=params.start_date,
                date_range_end=params.end_date,
            )

        ti_instance = TheseusInsight(
            research_interests_override=params.research_interests,
            start_date_override=params.start_date,
            end_date_override=params.end_date,
            receiver_address_override=final_email_recipients,
            profile_ids_override=resolved_profile_ids,
            orchestration_config=orchestration_config,
            generate_podcast=params.generate_podcast_run,
            top_n=params.num_sections or 5,
            db_saving=True,
            data_path=run_db_path,
            verbose=True,
            task_id=task_id,
            checkpoint_dir=os.path.join("data", "checkpoints", task_id),
            use_multi_server_judge=params.use_multi_server_judge,
            judge_server_ids=[s.id for s in judge_servers] if judge_servers else None,
            newsletter_job_id=newsletter_job_id,
            judge_request_timeout_sec=params.judge_request_timeout_sec,
            judge_max_retries=params.judge_max_retries,
        )
        # Run in a separate thread to avoid blocking the main event loop.
        # The run() method uses asyncio.run() internally for the pipeline.
        await asyncio.to_thread(
            ti_instance.run,
            progress_callback=make_pipeline_progress_callback(task_id, loop),
        )

        # Always mark as completed if we reach here successfully; the
        # progress callback may already have done so, which is fine.
        await task_manager.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            message="Pipeline finished processing.",
            current_step="newsletter_complete",
        )

    except Exception as e:
        error_message = f"Error in newsletter pipeline for task {task_id}: {type(e).__name__} - {str(e)}"
        if task_manager:
            await task_manager.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=error_message,
                message=error_message,
                current_step="newsletter_failed",
            )
        print(error_message)  # Log to server console as well

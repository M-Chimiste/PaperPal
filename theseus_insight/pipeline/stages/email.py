"""Stage 6: Send the newsletter email (extracted from run_async, B9)."""
import datetime
import hashlib
import asyncio
from typing import Callable, Optional, Tuple

from ...data_access import LogsRepository
from ...communication import construct_email_body


async def run(
    ti,
    newsletter_content,
    sections_data,
    progress_callback: Optional[Callable],
) -> Tuple[Optional[str], Optional[dict]]:
    """Send the newsletter email when generate_email is on.

    Loads newsletter_content/sections_data from checkpoints when the
    caller doesn't have them (later-stage resume); returns both so the
    orchestrator keeps the hydrated values for the podcast stage.
    """
    if progress_callback:
        progress_callback("newsletter", 85, "Starting newsletter email sending")
    if ti.generate_email:
        if newsletter_content is None:
            newsletter_content = await ti._load_checkpoint_async('newsletter_content')
            if newsletter_content is None:
                raise ValueError("Cannot send email: no newsletter content found.")

        if sections_data is None:
            sections_data = await ti._load_checkpoint_async('newsletter_sections')
            if sections_data is None:
                raise ValueError("No sections data found to build email links.")

        if ti.verbose:
            print("Sending newsletter email...")

        # Construct a simple bulleted list of links
        if len(sections_data['urls_and_titles']) > 0:
            urls_and_titles_bulleted = "\n".join(
                f"{i+1}. {title}" for i, title in enumerate(sections_data['urls_and_titles'])
            )
        else:
            urls_and_titles_bulleted = "No new papers found for this period."
        email_body = construct_email_body(
            newsletter_content,
            ti.start_date.strftime('%Y-%m-%d'),
            ti.end_date.strftime('%Y-%m-%d'),
            urls_and_titles_bulleted
        )
        from ...data_access.runtime import DeliveryRepository
        delivery_key = hashlib.sha256(f"newsletter:{ti.task_id}".encode()).hexdigest()
        if not await asyncio.to_thread(DeliveryRepository.begin, delivery_key, ti.task_id):
            return newsletter_content, sections_data
        try:
            ti.communication.compose_message(email_body, ti.start_date, ti.end_date)
            await asyncio.to_thread(ti.communication.send_email)
            await asyncio.to_thread(DeliveryRepository.finish, delivery_key, "sent")
            # Log successful email
            LogsRepository.upsert(
                task_id=ti.task_id,
                status=f"EMAIL_SUCCESS: Successfully sent newsletter to {ti.receiver_address}",
                datetime_run=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        except Exception as e:
            await asyncio.to_thread(DeliveryRepository.finish, delivery_key, "uncertain")
            ti._log_error(500, e)
            raise
    if progress_callback and not ti.generate_podcast:
        progress_callback("newsletter", 100, "Newsletter email sending complete")
    if progress_callback and ti.generate_podcast:
        progress_callback("newsletter", 90, "Newsletter email sending complete")

    return newsletter_content, sections_data

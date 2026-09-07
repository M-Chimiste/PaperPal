"""Assemble validated briefs with a source-reviewed local introduction."""
import asyncio

from ...data_access import NewsletterRepository
from ..newsletter_quality import intro_or_excerpts, assemble_newsletter


async def run(ti, sections_data, start_from, progress_callback):
    if start_from is not None and start_from not in ['newsletter_sections', 'newsletter_content']:
        return None, sections_data
    newsletter_content = await ti._load_checkpoint_async('newsletter_content')
    if sections_data is None:
        sections_data = await ti._load_checkpoint_async('newsletter_sections')
    if sections_data is None:
        raise ValueError('No newsletter sections found')
    if newsletter_content is None:
        papers = sections_data.get('papers', [])
        if not sections_data['sections']:
            newsletter_content = sections_data.get('empty_reason', 'No new papers met the newsletter criteria for this period.')
        elif not papers:
            raise ValueError('Newsletter sections lack validated evidence; start a new run')
        else:
            intro, fallback = await asyncio.to_thread(intro_or_excerpts, ti.newsletter_intro_inference, papers)
            if fallback:
                sections_data['intro_fallback'] = 'Validated contribution excerpts; synthesis did not pass review.'
            if sections_data.get('coverage_note'):
                intro += '\n\n' + sections_data['coverage_note']
            newsletter_content = assemble_newsletter(intro, papers)
        await ti._save_checkpoint_async('newsletter_content', newsletter_content)
    # Persist even after recovering the content checkpoint; the repository makes retries idempotent.
    if ti.db_saving:
        NewsletterRepository.save_edition(ti.task_id, newsletter_content, ti.start_date, ti.end_date,
                                          getattr(ti, 'profile_ids_override', None), sections_data)
    if progress_callback:
        progress_callback('newsletter', 80, 'Validated newsletter assembled')
    return newsletter_content, sections_data

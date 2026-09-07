"""Build source-backed local-model briefs in deterministic editorial order."""
import asyncio
import concurrent.futures as cf
import hashlib
import os

from ..newsletter_quality import (
    editorial_candidates, extract_evidence, write_brief, render_brief, paper_key,
    require_local,
)


async def run(ti, top_n_df, start_from, progress_callback):
    if start_from is not None and start_from not in ['papers_ranked', 'newsletter_sections']:
        return None
    cached = await ti._load_checkpoint_async('newsletter_sections')
    if cached is not None:
        return cached
    if top_n_df is None:
        top_n_df = await ti._load_checkpoint_async('papers_ranked')
    if top_n_df is None:
        raise ValueError('No ranked papers found')
    for client in (ti.content_extraction_inference, ti.newsletter_sections_inference):
        require_local(client)
    from ...data_access.newsletters import NewsletterRepository
    from ...api.tasks import record_async

    async def event(stage, status, **details):
        if getattr(ti, 'task_id', None):
            await record_async(ti.task_id, stage, status, details=details)

    rows = top_n_df.to_dict('records')
    shortlist = [{key: row.get(key, '') for key in ('title', 'abstract', 'pdf_url')}
                 for row in rows[:max(ti.top_n * 4, ti.top_n)]]
    previous = NewsletterRepository.recent_paper_keys(getattr(ti, 'profile_ids_override', None)) if ti.db_saving else []
    candidates = await ti._load_checkpoint_async('newsletter_editorial')
    if candidates is None:
        if progress_callback:
            progress_callback('newsletter', 35, f'Editorial review of {len(shortlist)} shortlisted papers with the local model')
        candidates = await asyncio.to_thread(editorial_candidates, shortlist, ti.newsletter_sections_inference, ti.research_interests, previous)
        await ti._save_checkpoint_async('newsletter_editorial', candidates)
    from ...observability import provenance
    result = {'sections': [], 'urls_and_titles': [], 'papers': [], 'rejected': [], 'quality_version': 1,
              'provenance': provenance(getattr(ti, 'orchestration_config', {}))}
    if rows and not candidates:
        # An editorial decision, not a conversion/generation failure.
        result['empty_reason'] = 'No papers passed the editorial relevance threshold.'
    if not candidates:
        await ti._save_checkpoint_async('newsletter_sections', result)
        return result

    def clean(future):
        if future.cancelled():
            return
        try:
            path = future.result()
            if path and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

    workers = min(max(ti.pdf_download_max_workers, 1), len(candidates))
    executor = cf.ThreadPoolExecutor(max_workers=workers)
    pending = {}
    next_index = 0

    def fill():
        nonlocal next_index
        while len(pending) < workers and next_index < len(candidates):
            row = candidates[next_index]
            pending[next_index] = executor.submit(ti._download_pdf_to_temp_file, row['pdf_url'])
            next_index += 1

    fill()
    try:
        for index, row in enumerate(candidates):
            future = pending.pop(index)
            key = hashlib.sha256(paper_key(row).encode()).hexdigest()[:24]
            try:
                evidence = await ti._load_checkpoint_async('evidence_' + key)
                if evidence is None:
                    if progress_callback:
                        progress_callback('newsletter', 40 + 10 * len(result['papers']) / max(ti.top_n, 1),
                                          f"Extracting source evidence: {row['title']}")
                    path = await asyncio.wrap_future(future)
                    markdown = await asyncio.to_thread(ti._parse_downloaded_pdf_to_markdown, path, row['pdf_url'])
                    evidence = await asyncio.to_thread(extract_evidence, ti.content_extraction_inference, markdown)
                    await ti._save_checkpoint_async('evidence_' + key, evidence)
                saved = await ti._load_checkpoint_async('brief_' + key)
                if saved is None:
                    if progress_callback:
                        progress_callback('newsletter', 40 + 10 * len(result['papers']) / max(ti.top_n, 1),
                                          f"Writing and checking evidence: {row['title']}")
                    featured = len(result['papers']) < 2
                    brief = await asyncio.to_thread(write_brief, ti.newsletter_sections_inference, ti.content_extraction_inference, evidence, ti.research_interests, featured)
                    saved = {'brief': brief.model_dump(), 'featured': featured}
                    await ti._save_checkpoint_async('brief_' + key, saved)
                from ..newsletter_quality import Brief
                brief = Brief.model_validate(saved['brief'])
                section = render_brief(row['title'], row['pdf_url'], brief, evidence, paper_id=f'P{len(result["papers"])+1}')
                result['sections'].append(section)
                result['urls_and_titles'].append(f"{row['title']}: {row['pdf_url']}")
                result['papers'].append({'key': paper_key(row), 'title': row['title'], 'url': row['pdf_url'],
                                         'editorial': row['editorial'], 'evidence': evidence, **saved})
                await event('paper_quality', 'completed', paper_key=paper_key(row), source_hash=evidence['source_hash'])
            except Exception as exc:
                result['rejected'].append({'key': paper_key(row), 'reason': str(exc)[:500]})
                await event('paper_quality', 'rejected', paper_key=paper_key(row), reason=str(exc)[:500])
            finally:
                # Also clean late downloads when cached evidence bypasses their wait.
                future.add_done_callback(clean)
            if progress_callback:
                progress_callback('newsletter', 40 + 10 * len(result['papers']) / max(ti.top_n, 1),
                                  f"Validated {len(result['papers'])}/{ti.top_n} paper briefs")
            if len(result['papers']) >= ti.top_n:
                break
            fill()
    finally:
        for future in pending.values():
            future.cancel()
            future.add_done_callback(clean)
        executor.shutdown(wait=False, cancel_futures=True)
    if not result['papers']:
        await ti._save_checkpoint_async('newsletter_quality_failures', result['rejected'])
        raise ValueError('No paper passed extraction and draft quality checks; newsletter was not generated')
    if len(result['papers']) < ti.top_n:
        result['coverage_note'] = f"Included {len(result['papers'])} papers that passed relevance and evidence checks (requested {ti.top_n})."
    await ti._save_checkpoint_async('newsletter_sections', result)
    return result

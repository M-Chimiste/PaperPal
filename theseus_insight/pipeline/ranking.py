"""Paper ranking/scoring (lifted verbatim from TheseusInsight, B9).

The historical-scores, single-server, and multi-server paths remain
separate implementations on purpose — unifying them is future work that
needs dedicated verification, not a mechanical move.
"""
import asyncio
import os
import time

import json_repair
import pandas as pd
import yake
from tqdm import tqdm

from ..data_access import PaperRepository
from ..data_model.papers import Paper
from ..prompt import (
    RESEARCH_INTERESTS_SYSTEM_PROMPT,
    ResearchInterestsPromptData,
    research_prompt,
)
from ..utils import TODAY
def rank_papers_with_historical_scores(ti, data_df, return_all_scored=False, progress_callback=None):
    """Optimized ranking that uses existing scores when available, only ranking new papers.

    Args:
        data_df: DataFrame of papers to rank
        return_all_scored: If True, return tuple (top_n_df, all_scored_df) for profile scoring
        progress_callback: Optional callback for progress updates
    """
    try:
        if ti.verbose:
            print(f"\n🏃‍♂️ OPTIMIZED RANKING WITH HISTORICAL SCORES")
            print(f"Total papers to process: {len(data_df)}")
            print("="*60)

        # Bulk-fetch existing rows + aggregated profile scores upfront so we
        # don't issue 3686 round-trips classifying "has score" vs "needs scoring".
        # For profile-aware runs the authoritative score lives in
        # paper_profile_scores, NOT papers.score — the latter is only populated by
        # the legacy single-research-interests path and stays NULL for profile mode.
        urls = [row['pdf_url'] for _, row in data_df.iterrows() if row.get('pdf_url')]
        url_to_paper = PaperRepository.get_url_to_id_and_score_map(urls)

        profile_score_map: dict = {}
        using_profile_scores = bool(getattr(ti, 'profile_ids_override', None))
        if using_profile_scores:
            from ..data_access.profiles import ProfileScoreRepository
            profile_score_map = ProfileScoreRepository.get_aggregated_scores_for_profiles(
                ti.profile_ids_override
            )
            if ti.verbose:
                print(
                    f"📚 Profile-aware resume: found cached scores for "
                    f"{len(profile_score_map)} papers across profiles {ti.profile_ids_override}"
                )

        # Separate papers into those with and without existing scores
        papers_with_scores = []
        papers_without_scores = []

        for idx, row in data_df.iterrows():
            paper_data = row.to_dict()
            pdf_url = row.get('pdf_url')
            existing_paper = url_to_paper.get(pdf_url) if pdf_url else None

            # Profile-aware path: prefer paper_profile_scores. A row in that table
            # means a worker successfully scored this paper for at least one of
            # the active profiles — treat it as historical, no rescore needed.
            profile_score = None
            if using_profile_scores and existing_paper:
                profile_score = profile_score_map.get(existing_paper['id'])

            if profile_score is not None and profile_score.get('score') is not None:
                paper_data['score'] = profile_score['score']
                paper_data['related'] = profile_score.get('related', True)
                paper_data['rationale'] = profile_score.get('rationale') or 'Historical profile score'
                papers_with_scores.append(paper_data)
            elif (
                existing_paper
                and existing_paper.get('score') is not None
                and (existing_paper.get('score') or 0) > 0
            ):
                # Legacy single-interests path: papers.score is populated
                paper_data['score'] = existing_paper['score']
                paper_data['related'] = existing_paper.get('related', True)
                paper_data['rationale'] = existing_paper.get('rationale', 'Historical score')
                papers_with_scores.append(paper_data)
            else:
                # Paper needs to be scored
                papers_without_scores.append(paper_data)

        if ti.verbose:
            print(f"📊 Papers with existing scores: {len(papers_with_scores)}")
            print(f"🔄 Papers needing new scores: {len(papers_without_scores)}")
            if using_profile_scores and papers_with_scores:
                reused_from_profile = sum(
                    1
                    for p in papers_with_scores
                    if isinstance(p.get('rationale'), str)
                    and 'profile' in p['rationale'].lower()
                )
                print(f"   ↳ reused from paper_profile_scores: {reused_from_profile}")

        # Create DataFrames
        scored_papers_df = pd.DataFrame(papers_with_scores) if papers_with_scores else pd.DataFrame()
        unscored_papers_df = pd.DataFrame(papers_without_scores) if papers_without_scores else pd.DataFrame()

        # Score the papers that don't have scores yet
        if not unscored_papers_df.empty:
            if ti.verbose:
                print(f"🧠 Running LLM judge on {len(unscored_papers_df)} unscored papers...")
            newly_scored_df = ti.rank_papers(unscored_papers_df, progress_callback=progress_callback)
        else:
            if ti.verbose:
                print("⏭️ No new papers required judge inference; reusing historical scores from the database.")
            newly_scored_df = pd.DataFrame()

        # Combine all papers
        if not scored_papers_df.empty and not newly_scored_df.empty:
            combined_df = pd.concat([scored_papers_df, newly_scored_df], ignore_index=True)
        elif not scored_papers_df.empty:
            combined_df = scored_papers_df
        elif not newly_scored_df.empty:
            combined_df = newly_scored_df
        else:
            combined_df = pd.DataFrame()

        if combined_df.empty:
            if ti.verbose:
                print("⚠️ No papers available after scoring")
            if return_all_scored:
                return pd.DataFrame(), pd.DataFrame()
            return pd.DataFrame()

        # Sort by score
        combined_df = combined_df.sort_values(by='score', ascending=False)

        # Get more papers than needed to allow for PDF conversion failures
        backup_multiplier = 4
        extended_count = min(len(combined_df), ti.top_n * backup_multiplier)
        top_n_df = combined_df.head(extended_count)

        if ti.verbose:
            print(f"✅ Final ranking complete:")
            print(f"   Total papers ranked: {len(combined_df)}")
            print(f"   Papers with historical scores: {len(papers_with_scores)}")
            print(f"   Papers newly scored: {len(newly_scored_df) if not newly_scored_df.empty else 0}")
            print(f"   Top papers selected: {len(top_n_df)} (target: {ti.top_n})")
            if len(top_n_df) > 0:
                print(f"   Score range: {top_n_df['score'].min():.1f} - {top_n_df['score'].max():.1f}")
            if return_all_scored:
                print(f"   Returning all {len(combined_df)} scored papers for profile saving")

        # Return both limited and full results if requested (for profile scoring)
        if return_all_scored:
            return top_n_df, combined_df

        return top_n_df

    except Exception as e:
        # Surface the underlying error so the caller's traceback isn't the only signal —
        # otherwise a bad fallback shape masks the real failure (e.g. caller unpacks
        # `top_n_df, all_scored_df` and sees only "too many values to unpack").
        print(f"Error in optimized ranking: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to original ranking method. Honor `return_all_scored` so the caller's
        # 2-tuple unpack still works — the fallback's `top_n_df` doubles as "all scored"
        # since we don't have a separate "all" set without re-running the full pipeline.
        fallback_df = ti.rank_papers(data_df, progress_callback=progress_callback)
        if return_all_scored:
            return fallback_df, fallback_df
        return fallback_df


async def rank_papers_async(ti, data_df):
    """Async wrapper for rank_papers that supports both single and multi-server modes."""
    if ti.use_multi_server_judge:
        return await rank_papers_multi_server(ti, data_df)
    else:
        # Run sync rank_papers in thread pool to avoid blocking
        return await asyncio.to_thread(ti._rank_papers_single_server, data_df)


async def rank_papers_multi_server(ti, data_df):
    """Rank papers using multi-server worker pool."""
    from theseus_insight.data_processing.newsletter_scorer import NewsletterScorer

    if ti.verbose:
        print(f"\n🚀 MULTI-SERVER JUDGE SCORING")
        print(f"Total papers to score: {len(data_df)}")
        print(f"Using {len(ti.judge_server_ids)} inference servers")
        print("="*60)

    # Prepare papers for scoring
    papers = []
    for idx, row in data_df.iterrows():
        papers.append({
            'id': row.get('id'),  # Assuming paper has ID from database
            'title': row.get('title', ''),
            'abstract': row.get('abstract', '')
        })

    # Create newsletter scorer
    scorer = NewsletterScorer(ti.orchestration_config)

    # Progress callback
    SCORING_STAGE_START = 20.0
    SCORING_STAGE_END = 30.0

    def progress_callback(status, progress, message=None, metadata=None):
        status_for_callback = status
        adjusted_progress = progress
        metadata_payload = dict(metadata) if isinstance(metadata, dict) else metadata

        if status == 'scoring':
            status_for_callback = 'rank'
            adjusted_progress = SCORING_STAGE_START + (progress / 100.0) * (SCORING_STAGE_END - SCORING_STAGE_START)

        if ti.verbose:
            print(
                f"Scoring progress: raw={progress:.1f}% "
                f"(overall {adjusted_progress:.1f}% between {SCORING_STAGE_START}-{SCORING_STAGE_END})"
            )

        if isinstance(metadata, dict):
            metadata_payload = dict(metadata)
            metadata_payload['scoring_progress_pct'] = progress
            metadata_payload['overall_progress_pct'] = adjusted_progress

        # Forward to main progress callback if available
        print(f"[DEBUG] progress_callback called: status={status_for_callback}, progress={adjusted_progress}, has_callback={ti.progress_callback is not None}")
        if ti.progress_callback:
            print(f"[DEBUG] Calling ti.progress_callback with metadata: {metadata_payload is not None}")
            try:
                # If main callback accepts metadata
                ti.progress_callback(status_for_callback, adjusted_progress, message, metadata_payload)
                print(f"[DEBUG] progress_callback succeeded")
            except TypeError as e:
                # Fallback for simpler signature
                print(f"[DEBUG] TypeError in progress_callback, falling back to 2-arg signature: {e}")
                try:
                    ti.progress_callback(status_for_callback, adjusted_progress)
                except Exception as e2:
                    print(f"[ERROR] Failed to call progress_callback with fallback: {type(e2).__name__}: {e2}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"[ERROR] Failed to call progress_callback: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG] No ti.progress_callback set!")

    # Score papers using multi-server worker pool
    # Uses profile-specific scoring like bulk judge, then aggregates across profiles
    results = await scorer.score_papers_multi_server(
        job_id=ti.newsletter_job_id,
        papers=papers,
        profile_ids=ti.profile_ids_override or [],
        server_ids=ti.judge_server_ids,
        progress_callback=progress_callback,
        request_timeout_sec=ti.judge_request_timeout_sec,
        max_retries=ti.judge_max_retries
    )

    # Convert results back to DataFrame format
    # Create a mapping from paper_id to scores
    score_map = {}
    for result in results:
        score_map[result['paper_id']] = {
            'score': result.get('score', 1),
            'related': result.get('related', False),
            'rationale': result.get('rationale', 'No rationale provided')
        }

    # Add scores to dataframe
    scores, related, rationale = [], [], []
    for idx, row in data_df.iterrows():
        paper_id = row.get('id')
        if paper_id in score_map:
            scores.append(score_map[paper_id]['score'])
            related.append(score_map[paper_id]['related'])
            rationale.append(score_map[paper_id]['rationale'])
        else:
            # Paper not scored (failed) - use defaults
            scores.append(1)
            related.append(False)
            rationale.append('Failed to score')

    data_df['score'] = scores
    data_df['related'] = related
    data_df['rationale'] = rationale
    data_df = data_df.sort_values(by='score', ascending=False)

    # Get more papers than needed to allow for PDF conversion failures
    backup_multiplier = 4
    extended_count = min(len(data_df), ti.top_n * backup_multiplier)
    top_n_df = data_df.head(extended_count)

    if ti.verbose:
        print(f"✅ Multi-server scoring complete:")
        print(f"   Total papers scored: {len(results)}")
        print(f"   Top papers selected: {len(top_n_df)} (target: {ti.top_n})")
        if len(top_n_df) > 0:
            print(f"   Score range: {top_n_df['score'].min():.1f} - {top_n_df['score'].max():.1f}")

    return top_n_df


def rank_papers(ti, data_df, progress_callback=None):
    """Given embedded papers, use judge model to score them (single-server mode)."""
    # If multi-server mode is enabled, delegate to async version
    if ti.use_multi_server_judge:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're inside an event loop - use asyncio.create_task or run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, rank_papers_multi_server(ti, data_df))
                return future.result()
        except RuntimeError:
            # Not in an event loop - safe to create one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(rank_papers_multi_server(ti, data_df))
    else:
        return rank_papers_single_server(ti, data_df, progress_callback=progress_callback)


def rank_papers_single_server(ti, data_df, progress_callback=None):
    """Single-server sequential scoring (original implementation)."""
    try:
        progress_callback = progress_callback or ti.progress_callback
        abstracts = list(data_df['abstract'])
        scores, related, rationale = [], [], []
        failed_papers = []
        consecutive_failures = 0

        # Check for partial checkpoint
        partial_checkpoint = ti._load_checkpoint('ranking_partial')
        start_index = 0
        if partial_checkpoint is not None:
            scores = partial_checkpoint.get('scores', [])
            related = partial_checkpoint.get('related', [])
            rationale = partial_checkpoint.get('rationale', [])
            failed_papers = partial_checkpoint.get('failed_papers', [])
            start_index = len(scores)
            if ti.verbose:
                print(f"Resuming ranking from paper {start_index + 1}/{len(abstracts)}")

        total_papers = len(abstracts)

        def emit_rank_progress(processed_count: int, in_progress_count: int = 0):
            if not progress_callback:
                return

            successful_count = max(processed_count - len(failed_papers), 0)
            pending_count = max(total_papers - processed_count - in_progress_count, 0)
            progress_pct = 20 + (processed_count / total_papers) * 30 if total_papers > 0 else 30
            current_paper_num = min(processed_count + in_progress_count, total_papers)
            message = (
                f"Ranking paper {current_paper_num}/{total_papers}"
                if total_papers > 0 else
                "Ranking papers"
            )
            scoring_summary = {
                "completed": successful_count,
                "failed": len(failed_papers),
                "pending": pending_count,
                "in_progress": in_progress_count,
                "total": total_papers,
                "pending_plus_in_progress": pending_count + in_progress_count,
            }
            metadata = {
                "papers_to_score": total_papers,
                "papers_total": total_papers,
                "papers_scored": successful_count,
                "papers_failed": len(failed_papers),
                "papers_pending": pending_count,
                "papers_in_progress": in_progress_count,
                "scoring_summary": scoring_summary,
            }

            try:
                progress_callback("rank", progress_pct, message, metadata)
            except TypeError:
                progress_callback("rank", progress_pct, message)

        for i, abstract in enumerate(tqdm(abstracts[start_index:], 
                                        disable=not ti.verbose, 
                                        desc="Ranking papers",
                                        initial=start_index, 
                                        total=len(abstracts))):
            actual_index = start_index + i

            # Progress update
            emit_rank_progress(processed_count=actual_index, in_progress_count=1)

            success = False
            attempts = 0
            max_attempts = 3

            while not success and attempts < max_attempts:
                attempts += 1
                try:
                    # Clear cache on second attempt if using Ollama
                    if attempts == 2 and consecutive_failures > 2:
                        ti._clear_judge_model_cache()

                    messages = [
                        {"role": "user", "content": research_prompt(ti.research_interests, abstract)}
                    ]

                    if ti.judge_inference.provider == "ollama":
                        response = ti.judge_inference.invoke(
                            messages=messages,
                            system_prompt=RESEARCH_INTERESTS_SYSTEM_PROMPT,
                            schema=ResearchInterestsPromptData
                        )
                    else:
                        # E.g. Anthropic or OpenAI
                        response = ti.judge_inference.invoke(
                            messages=messages,
                            system_prompt=RESEARCH_INTERESTS_SYSTEM_PROMPT
                        )

                    # Parse and validate JSON response
                    try:
                        response_json = json_repair.loads(response)

                        # Ensure response_json is a dictionary
                        if not isinstance(response_json, dict):
                            if ti.verbose:
                                print(f"Paper ranking expected dict, got {type(response_json)} for paper {actual_index+1}, attempt {attempts}")
                                print(f"Raw response: {response[:200]}...")
                            if attempts == max_attempts:
                                raise TypeError(f"Expected dict from JSON parsing, got {type(response_json)}")
                            continue

                    except Exception as json_error:
                        if ti.verbose:
                            print(f"JSON parsing failed for paper {actual_index+1}, attempt {attempts}: {json_error}")
                            print(f"Raw response: {response[:200]}...")
                        if attempts == max_attempts:
                            raise json_error
                        continue

                    # Validate required keys exist
                    required_keys = ['score', 'related', 'rationale']
                    missing_keys = [key for key in required_keys if key not in response_json]

                    if missing_keys:
                        if ti.verbose:
                            print(f"Missing keys {missing_keys} for paper {actual_index+1}, attempt {attempts}")
                            print(f"Response JSON: {response_json}")
                        if attempts == max_attempts:
                            raise KeyError(f"Missing required keys in response: {missing_keys}")
                        continue

                    # Validate and convert values
                    try:
                        score_val = int(response_json['score'])
                        related_val = bool(response_json['related'])
                        rationale_val = str(response_json['rationale'])

                        # Validate score range
                        if not (1 <= score_val <= 10):
                            if ti.verbose:
                                print(f"Invalid score {score_val} for paper {actual_index+1}, attempt {attempts}")
                            if attempts == max_attempts:
                                score_val = max(1, min(10, score_val))  # Clamp to valid range
                            else:
                                continue

                        scores.append(score_val)
                        related.append(related_val)
                        rationale.append(rationale_val)
                        success = True
                        consecutive_failures = 0  # Reset counter on success

                    except (ValueError, TypeError) as conversion_error:
                        if ti.verbose:
                            print(f"Value conversion failed for paper {actual_index+1}, attempt {attempts}: {conversion_error}")
                            print(f"Response JSON: {response_json}")
                        if attempts == max_attempts:
                            raise conversion_error
                        continue

                except Exception as e:
                    if ti.verbose:
                        print(f"Error processing paper {actual_index+1}, attempt {attempts}: {e}")

                    # Check if this is an LM Studio error - verify model and retry with longer delay
                    provider = getattr(ti.judge_inference, 'provider', None)
                    error_str = str(e).lower()
                    is_lmstudio_error = provider == "lmstudio" and (
                        "lm studio" in error_str or 
                        "inference" in error_str or
                        "websocket" in error_str or
                        error_str.strip() == "" or  # Empty error message
                        ": ." in str(e)  # LMStudio SDK empty error pattern
                    )

                    if is_lmstudio_error and attempts < max_attempts:
                        if ti.verbose:
                            print(f"LM Studio error detected, verifying model availability...")
                        ti._clear_judge_model_cache()  # This verifies LM Studio model
                        time.sleep(3)  # Longer delay for LM Studio recovery
                        continue

                    if attempts == max_attempts:
                        # Use default values for failed paper
                        if ti.verbose:
                            print(f"Using default values for failed paper {actual_index+1}")
                        scores.append(1)  # Default low score
                        related.append(False)  # Default not related
                        rationale.append(f"Failed to process: {str(e)[:100]}")
                        failed_papers.append(actual_index)
                        consecutive_failures += 1
                        success = True
                    else:
                        # Add small delay before retry
                        time.sleep(1)

            emit_rank_progress(processed_count=len(scores), in_progress_count=0)

            # Save partial progress every 50 papers
            if (actual_index + 1) % 50 == 0:
                partial_data = {
                    'scores': scores,
                    'related': related,
                    'rationale': rationale,
                    'failed_papers': failed_papers
                }
                ti._save_checkpoint('ranking_partial', partial_data)

        if failed_papers and ti.verbose:
            print(f"Warning: {len(failed_papers)} papers failed processing and received default scores")

        data_df['score'] = scores
        data_df['related'] = related
        data_df['rationale'] = rationale
        data_df = data_df.sort_values(by='score', ascending=False)

        # Get more papers than needed to allow for PDF conversion failures
        # We'll take 2x the requested amount as backup
        backup_multiplier = 4
        extended_count = min(len(data_df), ti.top_n * backup_multiplier)
        top_n_df = data_df.head(extended_count)

        if ti.verbose:
            print(f"Selected top {extended_count} papers (target: {ti.top_n}) to allow for PDF conversion failures")

        if ti.db_saving:
            print("Saving LLM judge scores to paper_profile_scores table")
            # Save scores to paper_profile_scores for each selected profile
            updated_count = 0
            new_count = 0

            # Get the model name for judge_model field
            judge_model_name = getattr(ti.judge_inference, 'model_name', 'unknown')

            for _, row in data_df.iterrows():
                # Check if paper already exists
                existing_paper = PaperRepository.get_by_url(row['pdf_url'])

                if existing_paper:
                    # Save scores to paper_profile_scores for each selected profile
                    try:
                        from ..db import get_cursor
                        with get_cursor() as cur:
                            # Update date_run in papers table
                            cur.execute("""
                                UPDATE papers
                                SET date_run = %s
                                WHERE url = %s
                            """, (
                                TODAY.strftime('%Y-%m-%d'),
                                row['pdf_url']
                            ))

                            # Write scores to paper_profile_scores for each profile
                            profile_ids = ti.profile_ids_override or []
                            if not profile_ids:
                                print("Warning: No profiles selected for newsletter - scores will not be saved")
                                continue

                            for profile_id in profile_ids:
                                cur.execute("""
                                    INSERT INTO paper_profile_scores
                                        (paper_id, profile_id, score, related, rationale, judge_model, date_scored)
                                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                                    ON CONFLICT (paper_id, profile_id)
                                    DO UPDATE SET
                                        score = EXCLUDED.score,
                                        related = EXCLUDED.related,
                                        rationale = EXCLUDED.rationale,
                                        judge_model = EXCLUDED.judge_model,
                                        date_scored = CURRENT_TIMESTAMP
                                """, (
                                    existing_paper['id'],
                                    profile_id,
                                    row['score'],
                                    row['related'],
                                    row['rationale'],
                                    judge_model_name
                                ))
                        updated_count += 1

                        # Extract and update keywords
                        try:
                            extractor = getattr(ti, '_yake_extractor', None)
                            if extractor is None:
                                extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
                                ti._yake_extractor = extractor
                            text_kw = f"{row['title']} {row['abstract']}"
                            kw_scores = extractor.extract_keywords(text_kw)
                            keywords = [w for w, _ in kw_scores]
                            if keywords:
                                PaperRepository.update_keywords(existing_paper['id'], keywords)
                        except Exception:
                            pass
                    except Exception as e:
                        if ti.verbose:
                            print(f"Failed to update paper {row['title']}: {e}")
                else:
                    # Paper doesn't exist yet (shouldn't happen with new flow, but handle gracefully)
                    embedding = row['abstract_embedding']
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        embedding = list(embedding)

                    paper = Paper(
                        title=row['title'],
                        abstract=row['abstract'],
                        url=row['pdf_url'],
                        date_run=TODAY.strftime('%Y-%m-%d'),
                        date=row['date'].strftime('%Y-%m-%d'),
                        score=None,  # Scores are now stored in paper_profile_scores
                        related=None,
                        rationale=None,
                        cosine_similarity=row['cosine_similarity'],
                        embedding_model=ti.embedding_model_name,
                        embedding=embedding
                    )

                    was_inserted = PaperRepository.insert_paper(paper, skip_duplicates=True)
                    if was_inserted:
                        new_count += 1

                        # Get the inserted paper to get its ID
                        inserted_paper = PaperRepository.get_by_url(row['pdf_url'])
                        if inserted_paper:
                            # Save scores to paper_profile_scores for each profile
                            try:
                                from ..db import get_cursor
                                with get_cursor() as cur:
                                    profile_ids = ti.profile_ids_override or []
                                    for profile_id in profile_ids:
                                        cur.execute("""
                                            INSERT INTO paper_profile_scores
                                                (paper_id, profile_id, score, related, rationale, judge_model, date_scored)
                                            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                                            ON CONFLICT (paper_id, profile_id)
                                            DO UPDATE SET
                                                score = EXCLUDED.score,
                                                related = EXCLUDED.related,
                                                rationale = EXCLUDED.rationale,
                                                judge_model = EXCLUDED.judge_model,
                                                date_scored = CURRENT_TIMESTAMP
                                        """, (
                                            inserted_paper.id,
                                            profile_id,
                                            row['score'],
                                            row['related'],
                                            row['rationale'],
                                            judge_model_name
                                        ))
                            except Exception as e:
                                if ti.verbose:
                                    print(f"Failed to save scores for paper {row['title']}: {e}")

                        # Extract and cache keywords
                        try:
                            extractor = getattr(ti, '_yake_extractor', None)
                            if extractor is None:
                                extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
                                ti._yake_extractor = extractor
                            text_kw = f"{row['title']} {row['abstract']}"
                            kw_scores = extractor.extract_keywords(text_kw)
                            keywords = [w for w, _ in kw_scores]
                            if inserted_paper and keywords:
                                PaperRepository.update_keywords(inserted_paper.id, keywords)
                        except Exception:
                            pass

            if ti.verbose:
                profile_count = len(ti.profile_ids_override) if ti.profile_ids_override else 0
                total_scores = (updated_count + new_count) * profile_count
                print(f"Database update complete: Saved {total_scores} scores ({updated_count} existing + {new_count} new papers × {profile_count} profiles) to paper_profile_scores")

        # Clean up partial checkpoint on success
        partial_checkpoint_path = os.path.join(ti.checkpoint_dir, 'ranking_partial_checkpoint.pkl')
        if os.path.exists(partial_checkpoint_path):
            os.remove(partial_checkpoint_path)

        return top_n_df
    except Exception as e:
        ti._log_error(500, e)
        raise

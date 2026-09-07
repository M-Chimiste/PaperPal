"""Stage 3: Rank papers / score for profiles (extracted from run_async, B9)."""
import gc
from typing import Callable, Optional

import pandas as pd

from ...data_access import PaperRepository


async def run(
    ti,
    embedded_df,
    start_from: Optional[str],
    progress_callback: Optional[Callable],
):
    """Rank papers (or profile-score them) and checkpoint as papers_ranked.

    Returns top_n_df (may be None when start_from skips this stage).
    The caller should drop its own embedded_df reference after this
    returns — the stage del's its local for the same memory-release
    behavior the inline block had.
    """
    top_n_df = None

    if progress_callback:
        progress_callback("rank", 20, "Starting paper ranking")

    if start_from is None or start_from in ['papers_embedded', 'papers_ranked']:
        top_n_df = await ti._load_checkpoint_async('papers_ranked')
        if top_n_df is None:
            # Check if we should use profile-based paper scoring
            if ti.profile_ids_override:
                if ti.verbose:
                    print(f"Profile newsletter generation for profiles: {ti.profile_ids_override}")
                    print("Using freshly downloaded papers and scoring for profile...")
                top_n_df = ti.get_and_score_profile_papers(
                    profile_ids=ti.profile_ids_override,
                    embedded_df=embedded_df,
                    progress_callback=progress_callback
                )
            else:
                # Use traditional embedding-based approach
                if embedded_df is None:
                    embedded_df = ti._load_checkpoint('papers_embedded')
                    if embedded_df is None:
                        raise ValueError("No embedded papers found to rank.")

                # Check if we have any papers to rank
                if len(embedded_df) == 0:
                    if ti.verbose:
                        print("No new papers to rank (all papers already exist in database)")
                        print("Loading existing papers from database for newsletter generation...")

                    # Load existing papers from database within date range for newsletter generation
                    existing_papers = PaperRepository.get_papers_in_date_range(
                        start_date=ti.start_date.strftime('%Y-%m-%d'),
                        end_date=ti.end_date.strftime('%Y-%m-%d')
                    )

                    if existing_papers:
                        # Convert to DataFrame format expected by newsletter generation
                        papers_list = []
                        for paper in existing_papers:
                            papers_list.append({
                                'title': paper['title'],
                                'abstract': paper['abstract'],
                                'pdf_url': paper['url'],
                                'date': paper['date'],
                                'score': paper.get('score', 5.0),  # Use existing score or default
                                'related': paper.get('related', True),
                                'rationale': paper.get('rationale', 'Previously scored paper'),
                                'cosine_similarity': paper.get('cosine_similarity', 0.0),
                                'abstract_embedding': paper.get('embedding', [])
                            })

                        df = pd.DataFrame(papers_list)
                        # Sort by score if available, otherwise by date
                        if 'score' in df.columns:
                            df = df.sort_values('score', ascending=False)
                        else:
                            df = df.sort_values('date', ascending=False)

                        # Get more papers than needed to allow for PDF conversion failures
                        backup_multiplier = 4
                        extended_count = min(len(df), ti.top_n * backup_multiplier)
                        top_n_df = df.head(extended_count)

                        if ti.verbose:
                            print(f"✅ Loaded {len(top_n_df)} existing papers for newsletter generation")
                    else:
                        # No papers found in database for date range
                        top_n_df = embedded_df.copy()  # Empty dataframe with same structure
                        if ti.verbose:
                            print("No existing papers found in database for date range")
                else:
                    if ti.verbose:
                        print("Ranking papers...")
                    top_n_df = ti.rank_papers_with_historical_scores(embedded_df, progress_callback=progress_callback)

            # Only checkpoint a non-empty result. Persisting an empty df here
            # would poison subsequent retries: load_checkpoint returns the
            # (empty) df, the `if top_n_df is None` guard sees a non-None
            # value, and the whole scoring stage gets skipped.
            if top_n_df is not None and len(top_n_df) > 0:
                await ti._save_checkpoint_async('papers_ranked', top_n_df)
            elif ti.verbose:
                print("Skipping papers_ranked checkpoint (empty result — retry will re-attempt scoring)")

        # free memory from embeddings if needed
        del embedded_df
        gc.collect()
    if progress_callback:
        progress_callback("rank", 30, "Paper ranking complete")

    return top_n_df

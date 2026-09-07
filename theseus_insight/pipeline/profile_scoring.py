"""Profile paper retrieval/scoring + the no-papers handler
(lifted verbatim from TheseusInsight, B9)."""
import datetime
from typing import Callable, List, Optional

import pandas as pd
from tqdm import tqdm

from ..data_access import LogsRepository, PaperRepository
from ..data_model.papers import Paper
from ..utils import TODAY
def get_profile_papers(ti, profile_ids: List[int], min_score: float = 0.5) -> pd.DataFrame:
    """
    Retrieve papers scored by specific profiles for newsletter generation.

    Args:
        profile_ids: List of profile IDs to filter by
        min_score: Minimum profile score threshold

    Returns:
        DataFrame with papers formatted for newsletter generation
    """
    try:
        if ti.verbose:
            print(f"\n📋 RETRIEVING PROFILE PAPERS")
            print(f"Profile IDs: {profile_ids}")
            print(f"Min score: {min_score}")
            print("="*60)

        from ..db import get_cursor

        # Get papers with profile scores
        papers_data = []
        with get_cursor() as cur:
            cur.execute("""
                SELECT DISTINCT p.*, pps.score, pps.related, pps.rationale
                FROM papers p
                INNER JOIN paper_profile_scores pps ON p.id = pps.paper_id
                WHERE pps.profile_id = ANY(%s)
                  AND pps.score >= %s
                  AND p.date >= %s
                  AND p.date <= %s
                ORDER BY pps.score DESC, p.date DESC
                LIMIT %s
            """, (profile_ids, min_score, ti.start_date, ti.end_date, ti.top_n * 4))  # Get 4x more than needed for PDF failures

            papers_data = cur.fetchall()

        if not papers_data:
            if ti.verbose:
                print("No papers found for the specified profiles and criteria")
            return pd.DataFrame()

        # Convert to DataFrame format expected by newsletter generation
        papers_list = []
        for paper in papers_data:
            papers_list.append({
                'title': paper['title'],
                'abstract': paper['abstract'],
                'pdf_url': paper['url'],
                'date': paper['date'],
                'score': paper['score'],  # Use profile score
                'related': paper['related'],
                'rationale': paper['rationale'],
                'cosine_similarity': 1.0,  # Set high since profile already filtered
                'abstract_embedding': None  # Not needed for profile-based approach
            })

        df = pd.DataFrame(papers_list)

        # Take more papers than needed to allow for PDF conversion failures
        backup_multiplier = 2
        extended_count = min(len(df), ti.top_n * backup_multiplier)
        top_df = df.head(extended_count)

        if ti.verbose:
            print(f"✅ Retrieved {len(top_df)} profile-scored papers")
            if len(top_df) > 0:
                print(f"Score range: {top_df['score'].min():.2f} - {top_df['score'].max():.2f}")

        return top_df

    except Exception as e:
        if ti.verbose:
            print(f"Error retrieving profile papers: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_and_score_profile_papers(
    ti,
    profile_ids: List[int],
    embedded_df: pd.DataFrame = None,
    progress_callback: Optional[Callable[[str, float, str, dict], None]] = None
) -> pd.DataFrame:
    """
    Get papers from database in date range and score them for the profile.
    This ensures newsletter generation works even if profile hasn't scored papers yet.

    Args:
        profile_ids: List of profile IDs to score for
        embedded_df: Optional DataFrame with freshly downloaded papers to use instead of database query

    Returns:
        DataFrame with top-scored papers for newsletter generation
    """
    try:
        if ti.verbose:
            print(f"\n🎯 SCORING PAPERS FOR PROFILE NEWSLETTER")
            print(f"Profile IDs: {profile_ids}")
            print(f"Date range: {ti.start_date} to {ti.end_date}")
            print("="*60)

        from ..data_access import ProfileRepository, ProfileInterestsRepository
        from ..db import get_cursor

        # Get research interests for the profile
        research_interests_list = []
        for profile_id in profile_ids:
            interests = ProfileInterestsRepository.get_by_profile(profile_id)
            for interest in interests:
                research_interests_list.append(interest["interest_text"])

        if not research_interests_list:
            if ti.verbose:
                print("⚠️ No research interests found for profile(s)")
            return pd.DataFrame()

        research_interests_text = "\n".join(research_interests_list)
        if ti.verbose:
            print(f"Research interests: {research_interests_text[:200]}...")

        # Use embedded_df if provided (freshly downloaded papers), otherwise query database
        if embedded_df is not None and not embedded_df.empty:
            if ti.verbose:
                print(f"📊 Using {len(embedded_df)} freshly downloaded papers from current session")

            # Convert DataFrame to the format expected by scoring
            papers_list = []
            for _, row in embedded_df.iterrows():
                papers_list.append({
                    'id': row.get('id'),  # Include database ID for multi-server scoring
                    'title': row['title'],
                    'abstract': row['abstract'],
                    'pdf_url': row['pdf_url'],
                    'date': row['date'],
                    'cosine_similarity': 1.0,  # Will be set during scoring
                    'abstract_embedding': row['abstract_embedding']  # Use existing embedding
                })

            df = pd.DataFrame(papers_list)
        else:
            # Fall back to database query (existing logic)
            # Get papers from database in date range (all candidates in the requested date window)
            papers_data = []
            with get_cursor() as cur:
                # First, let's check what date range we have in the database
                cur.execute("""
                    SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as total_papers
                    FROM papers
                """)
                date_info = cur.fetchone()
                if ti.verbose and date_info:
                    print(f"📊 Database contains {date_info['total_papers']} papers")
                    print(f"   Date range: {date_info['min_date']} to {date_info['max_date']}")

                cur.execute("""
                    SELECT id, title, abstract, url, date FROM papers
                    WHERE date >= %s AND date <= %s
                    ORDER BY date DESC, id ASC
                """, (ti.start_date, ti.end_date))

                papers_data = cur.fetchall()

            if not papers_data:
                if ti.verbose:
                    print("No papers found in database for the date range")
                    # Let's also check if there are any papers close to this date range
                    with get_cursor() as cur:
                        cur.execute("""
                            SELECT date, COUNT(*) as count 
                            FROM papers 
                            WHERE date >= %s AND date <= %s
                            GROUP BY date 
                            ORDER BY date DESC 
                            LIMIT 10
                        """, (
                            ti.start_date - datetime.timedelta(days=30),
                            ti.end_date + datetime.timedelta(days=30)
                        ))
                        nearby_papers = cur.fetchall()
                        if nearby_papers:
                            print(f"📅 Papers found within ±30 days:")
                            for row in nearby_papers:
                                print(f"   {row['date']}: {row['count']} papers")
                        else:
                            print("📅 No papers found within ±30 days of target range")
                return pd.DataFrame()

            if ti.verbose:
                print(f"📊 Found {len(papers_data)} papers in date range")

            # Convert to DataFrame for scoring
            papers_list = []
            for paper in papers_data:
                papers_list.append({
                    'id': paper.get('id'),  # Include database ID for multi-server scoring
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'pdf_url': paper['url'],
                    'date': paper['date'],
                    'cosine_similarity': 1.0,  # Will be set during scoring
                    'abstract_embedding': None  # Not needed for judge scoring
                })

            df = pd.DataFrame(papers_list)

        if ti.verbose:
            print(f"🧠 Starting scoring process with judge model...")

        # Temporarily override research interests for profile-specific scoring
        original_research_interests = ti.research_interests
        ti.research_interests = research_interests_text

        try:
            # Score papers using the optimized ranking method
            # Get both top_n for newsletter AND all scored papers for profile saving
            top_n_df, all_scored_df = ti.rank_papers_with_historical_scores(
                df,
                return_all_scored=True,
                progress_callback=progress_callback
            )
        finally:
            # Restore original research interests
            ti.research_interests = original_research_interests

        # Save ALL scored papers to paper_profile_scores table (not just top_n)
        if ti.db_saving and not all_scored_df.empty:
            from ..data_access.profiles import ProfileScoreRepository

            if ti.verbose:
                print(f"💾 Saving profile scores for {len(profile_ids)} profile(s)...")

            saved_scores = 0
            papers_inserted = 0
            for profile_id in profile_ids:
                # Save ALL scored papers, not just top_n
                for _, row in all_scored_df.iterrows():
                    # Get the paper ID from database using URL
                    existing_paper = PaperRepository.get_by_url(row['pdf_url'])

                    # If paper doesn't exist, insert it first (shouldn't normally happen but handle gracefully)
                    if not existing_paper:
                        if ti.verbose:
                            print(f"⚠️ Paper not found in database, inserting: {row['title'][:50]}...")

                        # Get or create embedding
                        embedding = row.get('abstract_embedding')
                        if embedding is None:
                            # Generate embedding if not present
                            embedding = ti.embedding_model.invoke(row['abstract'])

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
                            score=row['score'],
                            related=row['related'],
                            rationale=row['rationale'],
                            cosine_similarity=row.get('cosine_similarity', 0.0),
                            embedding_model=ti.embedding_model_name,
                            embedding=embedding
                        )

                        was_inserted = PaperRepository.insert_paper(paper, skip_duplicates=True)
                        if was_inserted:
                            papers_inserted += 1
                            existing_paper = PaperRepository.get_by_url(row['pdf_url'])

                    if existing_paper:
                        # Save profile score
                        success = ProfileScoreRepository.create_or_update_score(
                            paper_id=existing_paper['id'],
                            profile_id=profile_id,
                            score=int(row['score']),
                            related=bool(row['related']),
                            rationale=str(row['rationale']),
                            judge_model=getattr(ti.judge_inference, 'model_name', 'unknown')
                        )
                        if success:
                            saved_scores += 1

            if ti.verbose:
                print(f"✅ Saved {saved_scores} profile scores")
                if papers_inserted > 0:
                    print(f"✅ Inserted {papers_inserted} missing papers into database")

        # Return the limited top_n_df for newsletter generation
        # (all_scored_df was already saved to database above)
        top_df = top_n_df

        if ti.verbose:
            print(f"✅ Returning top {len(top_df)} papers for newsletter generation (target: {ti.top_n})")
            print(f"✅ Saved {len(all_scored_df)} total papers to paper_profile_scores")

        if ti.verbose:
            print(f"✅ Scored and selected top {len(top_df)} papers")
            if len(top_df) > 0:
                print(f"Score range: {top_df['score'].min():.2f} - {top_df['score'].max():.2f}")
                related_count = sum(top_df['related'])
                print(f"Related papers: {related_count}/{len(top_df)}")

        return top_df

    except Exception as e:
        if ti.verbose:
            print(f"Error scoring profile papers: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def store_papers_without_scoring(ti, data_df):
    """Store all papers without LLM judge scoring for profiles feature."""
    try:
        if ti.verbose:
            print(f"\n💾 STORING {len(data_df)} PAPERS WITHOUT SCORING")
            print("="*60)

        # Save all papers to DB with null scores for later profile-specific scoring
        saved_count = 0
        duplicate_count = 0
        duplicate_urls = []

        # Use YAKE keyword extractor
        try:
            extractor = getattr(ti, '_yake_extractor', None)
            if extractor is None:
                import yake
                extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
                ti._yake_extractor = extractor  # cache for reuse
        except ImportError:
            extractor = None
            if ti.verbose:
                print("YAKE not available, skipping keyword extraction")

        # Collect all papers and their embeddings for bulk update
        updates = []
        new_papers = []

        for _, row in tqdm(data_df.iterrows(), total=len(data_df), 
                          desc="Preparing embeddings", disable=not ti.verbose):
            # Check if paper already exists
            existing_paper = PaperRepository.get_by_url(row['pdf_url'])

            # Convert numpy array to list if needed for embedding
            embedding = row['abstract_embedding']
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding)

            if existing_paper:
                # Update existing paper's embedding
                updates.append((existing_paper['id'], embedding))
                duplicate_count += 1
            else:
                # For new papers, create and insert
                paper = Paper(
                    title=row['title'],
                    abstract=row['abstract'],
                    url=row['pdf_url'],
                    date_run='1970-01-01',  # Placeholder date - will be updated when scored
                    date=row['date'].strftime('%Y-%m-%d'),
                    score=0.0,  # Placeholder score - will be updated when scored
                    related=False,  # Placeholder - will be updated when scored
                    rationale='Not yet scored',  # Placeholder - will be updated when scored
                    cosine_similarity=row['cosine_similarity'],
                    embedding_model=ti.embedding_model_name,
                    embedding=embedding
                )
                new_papers.append(paper)

        # Bulk insert new papers
        if new_papers:
            if ti.verbose:
                print(f"\n💾 Inserting {len(new_papers)} new papers...")
            stats = PaperRepository.bulk_insert(new_papers, skip_duplicates=True)
            saved_count += stats.get('imported', 0)

        # Bulk update embeddings for existing papers
        if updates:
            if ti.verbose:
                print(f"\n💾 Updating embeddings for {len(updates)} existing papers...")
            PaperRepository.bulk_update_embeddings(updates, embedding_model=ti.embedding_model_name)
            # Note: These are updates, not new insertions

        if ti.verbose:
            print(f"✅ Storage complete: {saved_count} new papers saved, {len(updates)} papers updated with embeddings")

        return {
            'saved_count': saved_count,
            'duplicate_count': duplicate_count,
            'total_processed': len(data_df)
        }

    except Exception as e:
        ti._log_error(500, e)
        raise


def handle_no_papers_found(ti, reason="no_papers_from_arxiv"):
    """Handle the case where no papers were found from ArXiv or all papers were duplicates."""

    if reason == "all_duplicates":
        if ti.verbose:
            print("All papers already exist in database - no new papers to process.")
        log_status = "NO_NEW_PAPERS_ALL_DUPLICATES"
        email_message = f"""
No New Research Papers - Theseus Insight

Dear Subscriber,

We retrieved research papers from ArXiv for the period {ti.start_date.strftime('%Y-%m-%d')} to {ti.end_date.strftime('%Y-%m-%d')}, but all papers found were already processed in previous runs.

This means:
• ArXiv papers were successfully retrieved for your specified categories
• All papers have been previously analyzed and included in earlier newsletters
• No new research papers were published in your areas of interest during this period

Search Parameters:
• Date Range: {ti.start_date.strftime('%Y-%m-%d')} to {ti.end_date.strftime('%Y-%m-%d')}
• Categories: {getattr(ti, 'arxiv_filter_categories', 'Not specified')}

We'll continue monitoring for new papers in your next scheduled run.

Best regards,
Theseus Insight
        """.strip()
    elif reason == "threshold_not_met":
        if ti.verbose:
            print("No papers met the relevance threshold.")
        log_status = "NO_PAPERS_MEET_THRESHOLD"
        email_message = f"""
No Relevant Research Papers - Theseus Insight

Dear Subscriber,

We retrieved research papers from ArXiv for the period {ti.start_date.strftime('%Y-%m-%d')} to {ti.end_date.strftime('%Y-%m-%d')}, but none of the papers met your relevance criteria.

This means:
• ArXiv papers were successfully retrieved for your specified categories
• After analyzing each paper's relevance to your research interests, none scored above the minimum threshold
• The papers published during this period may not align closely with your specified research focus

Search Parameters:
• Date Range: {ti.start_date.strftime('%Y-%m-%d')} to {ti.end_date.strftime('%Y-%m-%d')}
• Categories: {getattr(ti, 'arxiv_filter_categories', 'Not specified')}
• Relevance Threshold: {getattr(ti, 'cosine_similarity_threshold', 'Not specified')}

Consider lowering your relevance threshold if you'd like to receive papers with broader relevance to your interests.

Best regards,
Theseus Insight
        """.strip()
    else:
        if ti.verbose:
            print("No papers found from ArXiv for the specified date range and categories.")
        log_status = "NO_PAPERS_FOUND"
        email_message = f"""
No Research Papers Found - Theseus Insight

Dear Subscriber,

We attempted to retrieve research papers from ArXiv for the period {ti.start_date.strftime('%Y-%m-%d')} to {ti.end_date.strftime('%Y-%m-%d')}, but no papers were found matching your criteria.

This could be due to:
• ArXiv API temporary unavailability (503 errors)
• No new papers published in your specified categories during this period
• Network connectivity issues

Search Parameters:
• Date Range: {ti.start_date.strftime('%Y-%m-%d')} to {ti.end_date.strftime('%Y-%m-%d')}
• Categories: {getattr(ti, 'arxiv_filter_categories', 'Not specified')}

We'll try again during the next scheduled run. If this issue persists, please check the ArXiv status or contact support.

Best regards,
Theseus Insight Team
        """.strip()

    # Log the event
    LogsRepository.upsert(
        task_id=ti.task_id, 
        status=log_status,
        datetime_run=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # Send notification email if email generation is enabled
    if ti.generate_email and ti.receiver_address:
        try:

            # Compose the email message
            ti.communication.compose_message(
                content=email_message,
                start_date=ti.start_date,
                end_date=ti.end_date
            )
            # Replace the subject to indicate no papers found (remove existing and set new)
            if ti.communication.email_message:
                del ti.communication.email_message['Subject']
                if reason == "all_duplicates":
                    ti.communication.email_message['Subject'] = "Theseus Insight - No New Papers"
                elif reason == "threshold_not_met":
                    ti.communication.email_message['Subject'] = "Theseus Insight - No Relevant Papers"
                else:
                    ti.communication.email_message['Subject'] = "Theseus Insight - No Papers Found"
            ti.communication.send_email()

            if ti.verbose:
                print(f"Sent 'no papers found' notification to {ti.receiver_address}")

            # Log successful email notification
            if reason == "all_duplicates":
                notification_type = "NO_NEW_PAPERS"
            elif reason == "threshold_not_met":
                notification_type = "NO_RELEVANT_PAPERS"
            else:
                notification_type = "NO_PAPERS_FOUND"
            LogsRepository.upsert(
                task_id=ti.task_id, 
                status=f"EMAIL_{notification_type}_NOTIFICATION: Sent to {ti.receiver_address}",
                datetime_run=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

        except Exception as e:
            if ti.verbose:
                print(f"Failed to send 'no papers found' notification: {e}")
            ti._log_error(500, e)

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple
import json

from psycopg import sql

from ..db import get_cursor
from .base import to_pgvector

VECTOR_DIM = 768  # TODO configurable


class PaperRepository:
    """CRUD and search operations for the `papers` table."""

    # ---------------------------------------------------------------------
    # Existence helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def exists_by_url(url: str) -> bool:
        with get_cursor() as cur:
            cur.execute("SELECT 1 FROM papers WHERE url = %s LIMIT 1", (url,))
            return cur.fetchone() is not None

    @staticmethod
    def get_url_to_id_and_score_map(urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """Bulk lookup: url -> {id, score, related, rationale}.

        Avoids N+1 round-trips when classifying a large batch of papers as
        already-scored vs needs-scoring. Returns only rows that exist; callers
        treat missing URLs as "needs ingest".
        """
        if not urls:
            return {}
        with get_cursor() as cur:
            cur.execute(
                "SELECT id, url, score, related, rationale FROM papers WHERE url = ANY(%s)",
                (list(urls),),
            )
            return {
                row["url"]: {
                    "id": row["id"],
                    "score": row["score"],
                    "related": row["related"],
                    "rationale": row["rationale"],
                }
                for row in cur.fetchall()
            }

    @staticmethod
    def exists_by_title(title: str) -> bool:
        with get_cursor() as cur:
            cur.execute("SELECT 1 FROM papers WHERE title = %s LIMIT 1", (title,))
            return cur.fetchone() is not None

    @staticmethod
    def bulk_check_existence(urls: Optional[List[str]] = None, titles: Optional[List[str]] = None) -> Tuple[Set[str], Set[str]]:
        """
        Check existence of multiple papers by URLs and/or titles in a single query.
        
        Args:
            urls: List of URLs to check (optional)
            titles: List of titles to check (optional)
            
        Returns:
            Tuple of (existing_urls, existing_titles)
        """
        existing_urls = set()
        existing_titles = set()
        
        with get_cursor() as cur:
            # Check URLs if provided
            if urls:
                # Use ANY for efficient bulk checking
                cur.execute(
                    "SELECT DISTINCT url FROM papers WHERE url = ANY(%s) AND url IS NOT NULL",
                    (urls,)
                )
                existing_urls = {row["url"] for row in cur.fetchall()}
            
            # Check titles if provided
            if titles:
                cur.execute(
                    "SELECT DISTINCT title FROM papers WHERE title = ANY(%s) AND title IS NOT NULL",
                    (titles,)
                )
                existing_titles = {row["title"] for row in cur.fetchall()}
        
        return existing_urls, existing_titles

    @staticmethod
    def get_all_urls_and_titles() -> Tuple[Set[str], Set[str]]:
        """
        Get all existing URLs and titles for duplicate checking.
        More efficient for large-scale duplicate checking than individual queries.
        
        Returns:
            Tuple of (all_urls, all_titles)
        """
        with get_cursor() as cur:
            cur.execute("SELECT url FROM papers WHERE url IS NOT NULL")
            all_urls = {row["url"] for row in cur.fetchall()}
            
            cur.execute("SELECT title FROM papers WHERE title IS NOT NULL")
            all_titles = {row["title"] for row in cur.fetchall()}
            
        return all_urls, all_titles

    # ---------------------------------------------------------------------
    # Inserts
    # ---------------------------------------------------------------------

    @staticmethod
    def insert(paper: Any, *, skip_duplicates: bool = True) -> bool:
        """Insert a new paper row.

        The `paper` argument may be a dataclass from existing models or any
        object exposing the expected attributes.
        """
        if skip_duplicates and PaperRepository.exists_by_url(paper.url):
            return False

        emb_literal = to_pgvector(getattr(paper, "embedding", None))

        # Hard filter: require non-empty title and abstract
        if not (getattr(paper, 'title', None) and str(paper.title).strip() and getattr(paper, 'abstract', None) and str(paper.abstract).strip()):
            return False

        with get_cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO papers
                        (title, abstract, date, date_run, score, rationale, related,
                         cosine_similarity, url, embedding_model, embedding, text)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """
                ),
                (
                    paper.title.strip(),
                    paper.abstract.strip(),
                    paper.date,
                    paper.date_run,
                    float(paper.score) if paper.score is not None else None,
                    paper.rationale,
                    bool(paper.related),
                    float(paper.cosine_similarity) if paper.cosine_similarity is not None else None,
                    paper.url,
                    paper.embedding_model,
                    emb_literal,
                    getattr(paper, "text", None),
                ),
            )
        return True

    @staticmethod
    def bulk_insert(papers: List[Any], *, skip_duplicates: bool = True, progress_callback=None) -> Dict[str, int]:
        """Bulk insert papers in a single transaction for improved performance.
        
        Args:
            papers: List of Paper objects to insert
            skip_duplicates: Whether to skip papers that already exist
            progress_callback: Optional callback function(current, total, message)
            
        Returns:
            Dictionary with statistics: {"total": int, "imported": int, "skipped": int, "errors": int}
        """
        stats = {"total": len(papers), "imported": 0, "skipped": 0, "errors": 0}
        
        if not papers:
            return stats
            
        # Get existing URLs for duplicate checking if needed
        existing_urls = set()
        existing_titles = set()
        if skip_duplicates:
            # Use optimized method to get all URLs and titles at once
            existing_urls, existing_titles = PaperRepository.get_all_urls_and_titles()
        
        # Process papers in batches to avoid memory issues
        batch_size = 1000
        for batch_start in range(0, len(papers), batch_size):
            batch_end = min(batch_start + batch_size, len(papers))
            batch = papers[batch_start:batch_end]
            
            # Prepare batch data
            batch_data = []
            for paper in batch:
                # Skip duplicates if requested
                if skip_duplicates:
                    if (paper.url in existing_urls or paper.title in existing_titles):
                        stats["skipped"] += 1
                        continue
                # Hard filter: require non-empty title and abstract
                if not (getattr(paper, 'title', None) and str(paper.title).strip() and getattr(paper, 'abstract', None) and str(paper.abstract).strip()):
                    stats["skipped"] += 1
                    continue
                
                try:
                    emb_literal = to_pgvector(getattr(paper, "embedding", None))
                    batch_data.append((
                        paper.title.strip(),
                        paper.abstract.strip(),
                        paper.date,
                        paper.date_run,
                        float(paper.score) if paper.score is not None else None,
                        paper.rationale,
                        bool(paper.related),
                        float(paper.cosine_similarity) if paper.cosine_similarity is not None else None,
                        paper.url,
                        paper.embedding_model,
                        emb_literal,
                        getattr(paper, "text", None),
                    ))
                    
                    # Add to existing sets to prevent duplicates within this batch
                    if skip_duplicates:
                        existing_urls.add(paper.url)
                        existing_titles.add(paper.title)
                        
                except Exception as e:
                    print(f"Error preparing paper '{getattr(paper, 'title', 'Unknown')}': {e}")
                    print(f"Paper data: title={getattr(paper, 'title', None)}, url={getattr(paper, 'url', None)}")
                    import traceback
                    print(f"Full traceback: {traceback.format_exc()}")
                    stats["errors"] += 1
            
            # Bulk insert this batch
            if batch_data:
                try:
                    with get_cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO papers
                                (title, abstract, date, date_run, score, rationale, related,
                                 cosine_similarity, url, embedding_model, embedding, text)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            batch_data
                        )
                    stats["imported"] += len(batch_data)
                except Exception as e:
                    print(f"Error during batch insert: {e}")
                    import traceback
                    print(f"Batch insert traceback: {traceback.format_exc()}")
                    print(f"Batch size: {len(batch_data)}")
                    if batch_data:
                        print(f"Sample batch item: {batch_data[0]}")
                    stats["errors"] += len(batch_data)
            
            # Report progress
            if progress_callback:
                progress_callback(batch_end, len(papers), f"Bulk importing papers: {batch_end}/{len(papers)}")
        
        return stats

    # ---------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------

    @staticmethod
    def get_by_id(paper_id: int) -> Dict[str, Any] | None:
        with get_cursor() as cur:
            cur.execute(
                "SELECT * FROM papers WHERE id = %s", (paper_id,)
            )
            return cur.fetchone()

    # ---------------------------------------------------------------------
    # Similarity search
    # ---------------------------------------------------------------------

    @staticmethod
    def find_similar(
        query_embedding: List[float],
        *,
        embedding_model_name: str | None = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        # Use the same pgvector format as the working insert method
        emb_literal = to_pgvector(query_embedding)
        
        with get_cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT *, (1 - (embedding <=> %s::vector))::float AS similarity_score
                    FROM papers
                    WHERE embedding IS NOT NULL
                      AND (%s::text IS NULL OR embedding_model = %s)
                      AND (1 - (embedding <=> %s::vector))::float >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                ),
                (emb_literal, embedding_model_name, embedding_model_name, emb_literal, similarity_threshold, emb_literal, limit),
            )
            rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            row["similarity_distance"] = 1 - row["similarity_score"]
            results.append(row)
        return results

    @staticmethod
    def find_similar_existing(paper_id: int, *, limit: int = 10, similarity_threshold: float = 0.7):
        # Get reference paper embedding
        with get_cursor() as cur:
            cur.execute("SELECT * FROM papers WHERE id = %s AND embedding IS NOT NULL", (paper_id,))
            ref = cur.fetchone()
        if not ref:
            return None

        ref_embedding = ref["embedding"]
        if ref_embedding is None:
            return None

        query_embedding = ref_embedding  # already stored literal string like '[...]'
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT *, 1 - (embedding <=> %s)::float AS similarity_score
                FROM papers
                WHERE id != %s AND embedding_model = %s AND embedding IS NOT NULL AND (1 - (embedding <=> %s)::float) >= %s
                ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (query_embedding, paper_id, ref["embedding_model"], query_embedding, similarity_threshold, query_embedding, limit),
            )
            similars = cur.fetchall()

        return {
            "reference_paper": ref,
            "similar_papers": similars,
            "total_similar": len(similars),
        }

    # ---------------------------------------------------------------------
    # Pagination & filtering
    # ---------------------------------------------------------------------

    @staticmethod
    def paginate(
        *,
        page: int = 1,
        page_size: int = 10,
        min_score: float | None = None,
        max_score: float | None = None,
        sort_field: str = "score",
        sort_direction: str = "desc",
        search: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        profile_ids: List[int] | None = None,
        min_profile_score: float | None = None,
        max_profile_score: float | None = None,
        profile_related_only: bool = False,
    ) -> Dict[str, Any]:
        
        # Determine if we need profile filtering
        has_profile_filters = (
            profile_ids is not None or 
            min_profile_score is not None or 
            max_profile_score is not None or 
            profile_related_only
        )
        
        if has_profile_filters:
            # Use profile-aware query with joins
            if profile_ids:
                # Join with paper_profile_scores for specific profiles
                sql = """
                    SELECT DISTINCT p.*, 
                           pps.score as profile_score,
                           pps.related as profile_related,
                           pps.rationale as profile_rationale,
                           pps.date_scored as profile_date_scored,
                           pps.judge_model as profile_judge_model,
                           pps.profile_id
                    FROM papers p
                    INNER JOIN paper_profile_scores pps ON p.id = pps.paper_id
                """
            else:
                # Just regular papers query - profile filters will be ignored
                sql = "SELECT * FROM papers"
        else:
            # Regular papers query without profile data
            sql = "SELECT * FROM papers"
        
        conditions: List[str] = []
        params: List[Any] = []

        # Regular paper filters
        if min_score is not None:
            conditions.append("p.score >= %s" if has_profile_filters and profile_ids else "score >= %s")
            params.append(min_score)
        if max_score is not None:
            conditions.append("p.score <= %s" if has_profile_filters and profile_ids else "score <= %s")
            params.append(max_score)
        if from_date:
            conditions.append("p.date >= %s" if has_profile_filters and profile_ids else "date >= %s")
            params.append(from_date)
        if to_date:
            conditions.append("p.date <= %s" if has_profile_filters and profile_ids else "date <= %s")
            params.append(to_date)
        if search:
            conditions.append("p.fts @@ plainto_tsquery('english', %s)" if has_profile_filters and profile_ids else "fts @@ plainto_tsquery('english', %s)")
            params.append(search)
        
        # Profile-specific filters
        if has_profile_filters and profile_ids:
            # Filter by specific profiles
            placeholders = ', '.join(['%s'] * len(profile_ids))
            conditions.append(f"pps.profile_id IN ({placeholders})")
            params.extend(profile_ids)
            
            # Profile score filters
            if min_profile_score is not None:
                conditions.append("pps.score >= %s")
                params.append(min_profile_score)
            if max_profile_score is not None:
                conditions.append("pps.score <= %s")
                params.append(max_profile_score)
            if profile_related_only:
                conditions.append("pps.related = TRUE")
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        # Build count query separately to avoid string replacement issues
        if has_profile_filters and profile_ids:
            sql_count = """
                SELECT COUNT(DISTINCT p.id) as count
                FROM papers p
                INNER JOIN paper_profile_scores pps ON p.id = pps.paper_id
            """
        else:
            sql_count = "SELECT COUNT(*) as count FROM papers"
        
        # Add the same WHERE conditions to count query
        if conditions:
            sql_count += " WHERE " + " AND ".join(conditions)

        # Sorting
        if sort_field not in {"score", "date", "id", "profile_score"}:
            sort_field = "score"
        
        # Adjust sort field for profile queries
        if has_profile_filters and profile_ids:
            if sort_field == "score":
                sort_field = "p.score"
            elif sort_field == "date":
                sort_field = "p.date"
            elif sort_field == "id":
                sort_field = "p.id"
            elif sort_field == "profile_score":
                sort_field = "pps.score"
        
        direction = "DESC" if sort_direction.lower() == "desc" else "ASC"
        sql += f" ORDER BY {sort_field} {direction} LIMIT %s OFFSET %s"
        offset = (page - 1) * page_size
        params_count = list(params)  # copy for count query
        params.extend([page_size, offset])

        with get_cursor() as cur:
            cur.execute(sql_count, params_count)
            total_items = cur.fetchone()["count"]
            cur.execute(sql, params)
            items = cur.fetchall()

        total_pages = (total_items + page_size - 1) // page_size if total_items else 0
        return {
            "items": items,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next_page": page < total_pages,
        }

    # ---------------------------------------------------------------------
    # Embedding helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def without_embeddings() -> List[Dict[str, Any]]:
        """Get all papers without embeddings.
        
        WARNING: This method loads ALL papers into memory.
        For large datasets, use without_embeddings_paginated() instead.
        """
        with get_cursor() as cur:
            cur.execute("SELECT * FROM papers WHERE embedding IS NULL ORDER BY id DESC")
            return cur.fetchall()

    @staticmethod
    def without_embeddings_paginated(limit: int = 10000, offset: int = 0) -> List[Dict[str, Any]]:
        """Get papers without embeddings with pagination support.
        
        This method enables streaming/chunked processing for large datasets.
        Only returns essential fields (id, title, abstract) to minimize memory usage.
        
        Args:
            limit: Maximum papers to return in this batch
            offset: Number of papers to skip (for pagination)
            
        Returns:
            List of paper dicts with id, title, abstract
        """
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, title, abstract 
                FROM papers 
                WHERE embedding IS NULL
                  AND LENGTH(TRIM(COALESCE(title, ''))) > 0 
                  AND LENGTH(TRIM(COALESCE(abstract, ''))) > 0
                ORDER BY id DESC 
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            return cur.fetchall()

    @staticmethod
    def count_without_embeddings() -> int:
        """Count papers without embeddings.
        
        More efficient than fetching all papers when you only need the count.
        
        Returns:
            Count of papers without embeddings
        """
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) as count 
                FROM papers 
                WHERE embedding IS NULL
                  AND LENGTH(TRIM(COALESCE(title, ''))) > 0 
                  AND LENGTH(TRIM(COALESCE(abstract, ''))) > 0
                """
            )
            row = cur.fetchone()
            return int(row['count']) if row and 'count' in row else 0

    @staticmethod
    def update_embedding(paper_id: int, embedding: List[float]):
        emb_literal = to_pgvector(embedding)
        with get_cursor() as cur:
            cur.execute(
                "UPDATE papers SET embedding = %s WHERE id = %s",
                (emb_literal, paper_id),
            )

    @staticmethod
    def update_fields(paper_id: int, *, score: Optional[float] = None, related: Optional[bool] = None) -> Dict[str, Any]:
        """Update mutable scalar fields on a paper row.

        Only fields explicitly provided will be updated. Returns the updated row.

        Args:
            paper_id: Target paper id
            score: New score value (0-10) if provided
            related: New related flag if provided
        """
        set_clauses: List[str] = []
        params: List[Any] = []

        if score is not None:
            set_clauses.append("score = %s")
            params.append(float(score))
        if related is not None:
            set_clauses.append("related = %s")
            params.append(bool(related))

        if not set_clauses:
            # Nothing to update; return current row
            with get_cursor() as cur:
                cur.execute("SELECT * FROM papers WHERE id = %s", (paper_id,))
                row = cur.fetchone()
                return row

        query = f"UPDATE papers SET {', '.join(set_clauses)} WHERE id = %s RETURNING *"
        params.append(paper_id)

        with get_cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return row

    @staticmethod
    def bulk_update_embeddings(updates: List[Tuple[int, List[float]]], embedding_model: str = "Alibaba-NLP/gte-large-en-v1.5"):
        """
        Bulk update embeddings for multiple papers.
        
        Args:
            updates: List of tuples (paper_id, embedding)
            embedding_model: Name of the embedding model used
        """
        if not updates:
            return
        
        # Process in batches to avoid memory allocation errors
        # With 768-dim embeddings, 5000 papers = ~30MB of data per batch
        batch_size = 5000
        total_updated = 0
        
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            
            with get_cursor() as cur:
                # Build arrays for bulk update
                paper_ids = []
                embedding_strs = []
                
                for paper_id, embedding in batch:
                    paper_ids.append(paper_id)
                    # Convert embedding to PostgreSQL vector format
                    emb_literal = to_pgvector(embedding)
                    embedding_strs.append(emb_literal)
                
                # Use UNNEST to perform bulk update - now also updating embedding_model
                query = """
                    UPDATE papers 
                    SET embedding = data.embedding::vector,
                        embedding_model = %s
                    FROM (
                        SELECT unnest(%s::int[]) as id,
                               unnest(%s::text[]::vector[]) as embedding
                    ) as data
                    WHERE papers.id = data.id
                """
                
                cur.execute(query, (embedding_model, paper_ids, embedding_strs))
                total_updated += len(batch)
                
                # Print progress for large updates
                if len(updates) > batch_size and (i + batch_size) % (batch_size * 5) == 0:
                    print(f"  Updated {total_updated}/{len(updates)} papers...")

    # ---------------------------------------------------------------------
    # Convenience wrappers for router compatibility
    # ---------------------------------------------------------------------

    @staticmethod
    def semantic_search(query_text: str, embedding_model, *, embedding_model_name: str | None = None, limit: int = 10, similarity_threshold: float = 0.7):
        query_embedding = embedding_model.invoke(query_text)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        return PaperRepository.find_similar(query_embedding, embedding_model_name=embedding_model_name, limit=limit, similarity_threshold=similarity_threshold)

    @staticmethod
    def hybrid_search(
        query_text: str,
        embedding_model,
        *,
        embedding_model_name: str | None = None,
        profile_ids: List[int] | None = None,
        min_profile_score: float | None = None,
        max_profile_score: float | None = None,
        profile_related: bool | None = None,
        page: int = 1,
        page_size: int = 10,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        min_score: float | None = None,
        max_score: float | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        similarity_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        import os
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("Invalid search pagination")
        if min(semantic_weight, keyword_weight) < 0 or abs(semantic_weight + keyword_weight - 1) > .01:
            raise ValueError("Search weights must be nonnegative and sum to one")
        # Fixed candidate windows keep page counts stable across pagination.
        candidate_limit = max(100, min(int(os.getenv("SEARCH_CANDIDATES", "500")), 5000))
        query_embedding = embedding_model.invoke(query_text) if semantic_weight else None
        emb_literal = to_pgvector(query_embedding)
        filters = []
        params = {"embedding": emb_literal, "query": query_text, "limit": candidate_limit,
                  "semantic_weight": semantic_weight, "keyword_weight": keyword_weight,
                  "threshold": similarity_threshold, "offset": (page-1)*page_size, "page_size": page_size}
        for column, op, value, key in [("score", ">=", min_score, "min_score"),
                                      ("score", "<=", max_score, "max_score"),
                                      ("date", ">=", from_date, "from_date"),
                                      ("date", "<=", to_date, "to_date")]:
            if value is not None:
                filters.append(f"{column} {op} %({key})s")
                params[key] = value
        profile_projection = "NULL::float AS profile_score"
        if profile_ids:
            params['profiles'] = profile_ids
            profile_conditions = ['pps.paper_id=papers.id', 'pps.profile_id=ANY(%(profiles)s)']
            for column, op, value, key in [('score', '>=', min_profile_score, 'profile_min'),
                                          ('score', '<=', max_profile_score, 'profile_max'),
                                          ('related', '=', profile_related, 'profile_related')]:
                if value is not None:
                    profile_conditions.append(f'pps.{column} {op} %({key})s')
                    params[key] = value
            filters.append('EXISTS (SELECT 1 FROM paper_profile_scores pps WHERE ' + ' AND '.join(profile_conditions) + ')')
            profile_projection = '(SELECT AVG(pps.score)::float FROM paper_profile_scores pps WHERE pps.paper_id=p.id AND pps.profile_id=ANY(%(profiles)s)) AS profile_score'
        where = " AND ".join(filters) or "TRUE"
        model_filter = ""
        if embedding_model_name:
            model_filter = " AND embedding_model = %(model)s"
            params['model'] = embedding_model_name
        # Independent lexical retrieval includes exact matches without embeddings.
        # ORDER BY distance ASC allows the vector index to serve candidates.
        query = f"""
            WITH semantic_candidates AS MATERIALIZED (
                SELECT id, embedding <=> %(embedding)s::vector AS distance
                FROM papers WHERE {where} AND embedding IS NOT NULL {model_filter}
                  AND %(semantic_weight)s > 0
                ORDER BY embedding <=> %(embedding)s::vector LIMIT %(limit)s
            ), semantic AS (
                SELECT id, 1-distance AS semantic_score,
                       row_number() OVER (ORDER BY distance, id) AS rank
                FROM semantic_candidates WHERE 1-distance >= %(threshold)s
            ), lexical_candidates AS MATERIALIZED (
                SELECT id, ts_rank(fts, websearch_to_tsquery('english', %(query)s)) AS keyword_score
                FROM papers WHERE {where} AND %(keyword_weight)s > 0
                  AND fts @@ websearch_to_tsquery('english', %(query)s)
                ORDER BY keyword_score DESC, id LIMIT %(limit)s
            ), lexical AS (
                SELECT *, row_number() OVER (ORDER BY keyword_score DESC, id) AS rank
                FROM lexical_candidates
            ), fused AS MATERIALIZED (
                SELECT COALESCE(s.id,l.id) AS id, COALESCE(s.semantic_score,0) AS semantic_score,
                       COALESCE(l.keyword_score,0) AS keyword_score,
                       COALESCE(%(semantic_weight)s/(60.0+s.rank),0) +
                       COALESCE(%(keyword_weight)s/(60.0+l.rank),0) AS hybrid_score
                FROM semantic s FULL OUTER JOIN lexical l ON s.id=l.id
            ), page AS (
                SELECT p.*, f.semantic_score, f.keyword_score, f.hybrid_score, {profile_projection}
                FROM fused f JOIN papers p ON p.id=f.id
                ORDER BY f.hybrid_score DESC, p.id LIMIT %(page_size)s OFFSET %(offset)s
            )
            SELECT (SELECT count(*) FROM fused) AS total_items,
                   COALESCE((SELECT jsonb_agg(to_jsonb(page) - 'embedding' - 'fts' - 'text') FROM page), '[]'::jsonb) AS items
        """
        with get_cursor() as cur:
            cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(min(candidate_limit, 1000)),))
            cur.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
            cur.execute(query, params)
            row = cur.fetchone()
        total = row['total_items']
        return {"items": row['items'], "total_items": total,
                "total_pages": (total + page_size - 1)//page_size, "current_page": page,
                "candidate_limit": candidate_limit, "count_scope": "retrieved_candidates"}

    @staticmethod
    def search_seed(query: str, limit: int = 10):
        """Search papers by title/abstract for mind-map seed selection."""
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, title, abstract, date, score, url
                FROM papers
                WHERE fts @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(fts, plainto_tsquery('english', %s)) DESC
                LIMIT %s
                """,
                (query, query, limit),
            )
            return cur.fetchall()

    # ---------------------------------------------------------------------
    # Alias methods for compatibility with legacy code
    # ---------------------------------------------------------------------

    @staticmethod
    def insert_paper(paper: Any, *, skip_duplicates: bool = True) -> bool:
        """Alias for insert method."""
        return PaperRepository.insert(paper, skip_duplicates=skip_duplicates)

    @staticmethod
    def get_by_url(url: str) -> Dict[str, Any] | None:
        """Get paper by URL."""
        with get_cursor() as cur:
            cur.execute("SELECT * FROM papers WHERE url = %s", (url,))
            return cur.fetchone()

    @staticmethod
    def get_all() -> List[Dict[str, Any]]:
        """Get all papers."""
        with get_cursor() as cur:
            cur.execute("SELECT * FROM papers ORDER BY id DESC")
            return cur.fetchall()

    @staticmethod
    def update_keywords(paper_id: int, keywords: List[str]) -> None:
        """Update keywords for a paper."""
        keywords_json = json.dumps(keywords) if keywords else None
        with get_cursor() as cur:
            cur.execute(
                "UPDATE papers SET keywords_json = %s WHERE id = %s",
                (keywords_json, paper_id)
            )

    @staticmethod
    def bulk_update_keywords(updates: List[Tuple[int, List[str]]]) -> int:
        """
        Update keywords for multiple papers in a single batch.
        
        Args:
            updates: List of tuples containing (paper_id, keywords_list)
            
        Returns:
            Number of papers updated
        """
        if not updates:
            return 0
            
        with get_cursor() as cur:
            # Prepare data for executemany
            values = []
            for paper_id, keywords in updates:
                keywords_json = json.dumps(keywords) if keywords else None
                values.append((keywords_json, paper_id))
            
            cur.executemany(
                "UPDATE papers SET keywords_json = %s WHERE id = %s",
                values
            )
            return cur.rowcount

    # For backward compatibility
    paper_exists_by_url = exists_by_url 

    # ---------------------------------------------------------------------
    # Mindmap-specific methods
    # ---------------------------------------------------------------------

    @staticmethod
    def find_similar_mindmap(
        seed_paper_id: int, *, k: int = 15, similarity_threshold: float = 0.3,
        profile_ids: Optional[List[int]] = None, min_profile_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find similar papers for mindmap generation with optional profile filtering."""
        # Get seed paper embedding
        with get_cursor() as cur:
            cur.execute(
                "SELECT embedding FROM papers WHERE id = %s AND embedding IS NOT NULL", 
                (seed_paper_id,)
            )
            seed_row = cur.fetchone()
            
        if not seed_row or not seed_row["embedding"]:
            return []
            
        seed_embedding = seed_row["embedding"]
        
        # Base query for similarity search
        if profile_ids:
            # Profile-filtered query
            query = """
                SELECT DISTINCT p.*, 1 - (p.embedding <=> %s)::float AS similarity_score,
                       pps.score as profile_score, pps.related as profile_related
                FROM papers p
                INNER JOIN paper_profile_scores pps ON p.id = pps.paper_id
                WHERE p.id != %s 
                  AND p.embedding IS NOT NULL 
                  AND (1 - (p.embedding <=> %s)::float) >= %s
                  AND pps.profile_id = ANY(%s)
            """
            params = [seed_embedding, seed_paper_id, seed_embedding, similarity_threshold, profile_ids]
            
            if min_profile_score is not None:
                query += " AND pps.score >= %s"
                params.append(min_profile_score)
                
            query += " ORDER BY similarity_score DESC LIMIT %s"
            params.append(k)
        else:
            # Regular similarity search without profile filtering
            query = """
                SELECT *, 1 - (embedding <=> %s)::float AS similarity_score
                FROM papers 
                WHERE id != %s 
                  AND embedding IS NOT NULL 
                  AND (1 - (embedding <=> %s)::float) >= %s
                ORDER BY similarity_score DESC 
                LIMIT %s
            """
            params = [seed_embedding, seed_paper_id, seed_embedding, similarity_threshold, k]
        
        with get_cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    @staticmethod
    def get_keywords(paper_id: int) -> List[str] | None:
        """Get keywords for a paper."""
        with get_cursor() as cur:
            cur.execute("SELECT keywords_json FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            if row and row["keywords_json"]:
                try:
                    return json.loads(row["keywords_json"])
                except (json.JSONDecodeError, TypeError):
                    return []
            return []

    @staticmethod
    def get_summary(paper_id: int) -> str | None:
        """Get paper summary/text."""
        with get_cursor() as cur:
            cur.execute("SELECT summary FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            return row["summary"] if row else None

    @staticmethod
    def update_summary(paper_id: int, summary: str) -> None:
        """Update paper summary/text."""
        with get_cursor() as cur:
            cur.execute(
                "UPDATE papers SET summary = %s WHERE id = %s",
                (summary, paper_id)
            )

    @staticmethod
    def get_text(paper_id: int) -> str | None:
        """Get paper full text."""
        with get_cursor() as cur:
            cur.execute("SELECT text FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            return row["text"] if row else None

    @staticmethod
    def update_text(paper_id: int, text: str) -> None:
        """Update paper full text."""
        with get_cursor() as cur:
            cur.execute(
                "UPDATE papers SET text = %s WHERE id = %s",
                (text, paper_id)
            )

    # ---------------------------------------------------------------------
    # Legacy compatibility aliases for mindmap conversion
    # ---------------------------------------------------------------------

    @staticmethod
    def get(paper_id: int) -> Dict[str, Any] | None:
        """Alias for get_by_id() - legacy compatibility."""
        return PaperRepository.get_by_id(paper_id)

    @staticmethod
    def get_paper_by_id(paper_id: int) -> Dict[str, Any] | None:
        """Legacy method name - use get_by_id() instead."""
        return PaperRepository.get_by_id(paper_id)

    @staticmethod
    def update_paper_embedding(paper_id: int, embedding: List[float]) -> None:
        """Legacy method name - use update_embedding() instead."""
        return PaperRepository.update_embedding(paper_id, embedding)

    @staticmethod
    def get_paper_keywords(paper_id: int) -> List[str] | None:
        """Legacy method name - use get_keywords() instead."""
        return PaperRepository.get_keywords(paper_id)

    @staticmethod
    def update_paper_keywords(paper_id: int, keywords: List[str]) -> None:
        """Legacy method name - use update_keywords() instead."""
        return PaperRepository.update_keywords(paper_id, keywords)

    @staticmethod
    def get_paper_summary(paper_id: int) -> str | None:
        """Legacy method name - use get_summary() instead."""
        return PaperRepository.get_summary(paper_id)

    @staticmethod
    def update_paper_summary(paper_id: int, summary: str) -> None:
        """Legacy method name - use update_summary() instead."""
        return PaperRepository.update_summary(paper_id, summary)

    @staticmethod
    def get_paper_text(paper_id: int) -> str | None:
        """Legacy method name - use get_text() instead."""
        return PaperRepository.get_text(paper_id)

    @staticmethod
    def update_paper_text(paper_id: int, text: str) -> None:
        """Legacy method name - use update_text() instead."""
        return PaperRepository.update_text(paper_id, text)

    @staticmethod
    def find_similar_papers_mindmap(
        seed_paper_id: int, k: int = 15, similarity_threshold: float = 0.3,
        profile_ids: Optional[List[int]] = None, min_profile_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Legacy method name - use find_similar_mindmap() instead."""
        return PaperRepository.find_similar_mindmap(
            seed_paper_id, k=k, similarity_threshold=similarity_threshold,
            profile_ids=profile_ids, min_profile_score=min_profile_score
        )

    # ---------------------------------------------------------------------
    # Utility script support methods
    # ---------------------------------------------------------------------

    @staticmethod
    def get_papers_without_embeddings() -> List[Dict[str, Any]]:
        """Get papers that don't have embeddings yet - used by backfill utility."""
        return PaperRepository.without_embeddings()

    @staticmethod
    def get_papers_without_keywords() -> List[Dict[str, Any]]:
        """Get papers that don't have keywords yet - used by backfill utility."""
        with get_cursor() as cur:
            cur.execute(
                "SELECT id, title, abstract FROM papers WHERE keywords_json IS NULL OR keywords_json = '' ORDER BY id DESC"
            )
            return cur.fetchall()

    @staticmethod
    def get_papers_with_embeddings(
        limit: Optional[int] = 10000,
        offset: int = 0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get papers that have embeddings for trend analysis.

        Args:
            limit: Maximum number of papers to return (None for all)
            offset: Number of papers to skip
            start_date: Optional start date filter
            end_date: Optional end date filter
        """
        with get_cursor() as cur:
            # Build query with optional date filters
            conditions = ["embedding IS NOT NULL"]
            params: List[Any] = []

            if start_date:
                conditions.append("date >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("date <= %s")
                params.append(end_date)

            where_clause = " AND ".join(conditions)

            # If limit is None or very large (100000+), remove it to get all papers
            if limit is None or limit >= 100000:
                query = f"SELECT * FROM papers WHERE {where_clause} ORDER BY date DESC OFFSET %s"
                params.append(offset)
            else:
                query = f"SELECT * FROM papers WHERE {where_clause} ORDER BY date DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])

            cur.execute(query, params)
            return cur.fetchall()

    @staticmethod
    def get_papers_in_date_range(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get papers within a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format (inclusive)
            end_date: End date in YYYY-MM-DD format (inclusive)

        Returns:
            List of paper dictionaries
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"PaperRepository.get_papers_in_date_range called with start_date={start_date}, end_date={end_date}")

        with get_cursor() as cur:
            query = """
                SELECT id, title, abstract, date, url, score, related, rationale, 
                       cosine_similarity, embedding_model, date_run
                FROM papers 
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
                
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
                
            query += " ORDER BY date DESC"
            
            cur.execute(query, params)
            rows = cur.fetchall()
            logger.info(f"Query executed with params: {params}")
            logger.info(f"Query returned {len(rows)} rows")

            return [
                {
                    'id': row['id'],
                    'title': row['title'],
                    'abstract': row['abstract'],
                    'date': row['date'],
                    'url': row['url'],
                    'score': row['score'],
                    'related': row['related'],
                    'rationale': row['rationale'],
                    'cosine_similarity': row['cosine_similarity'],
                    'embedding_model': row['embedding_model'],
                    'date_run': row['date_run']
                }
                for row in rows
            ]
    
    @staticmethod
    def get_paper_ids_in_date_range(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[int]:
        """
        Get only paper IDs within a date range for efficient filtering.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (inclusive)
            end_date: End date in YYYY-MM-DD format (inclusive)
            
        Returns:
            List of paper IDs
        """
        with get_cursor() as cur:
            query = "SELECT id FROM papers WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
                
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
                
            cur.execute(query, params)
            return [row['id'] for row in cur.fetchall()] 

    @staticmethod
    def count_embeddings_status_in_date_range(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, int]:
        """Return counts of total papers and count with non-null embeddings in the date range."""
        conditions: List[str] = []
        params: List[Any] = []
        if start_date:
            conditions.append("date >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("date <= %s")
            params.append(end_date)

        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        with get_cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) AS total, COUNT(embedding) AS embedded FROM papers{where_sql}",
                params,
            )
            row = cur.fetchone()
            total = int(row['total']) if row and 'total' in row else 0
            embedded = int(row['embedded']) if row and 'embedded' in row else 0
            return {"total": total, "embedded": embedded}

    @staticmethod
    def get_papers_missing_embeddings_in_date_range(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return papers within a date range that are missing embeddings.

        Only returns fields needed for embedding preflight to minimize payload.
        Filters to non-empty title and abstract to avoid embedding useless rows.
        
        WARNING: This method loads ALL matching papers into memory.
        For large datasets, use get_papers_missing_embeddings_in_date_range_paginated() instead.
        """
        with get_cursor() as cur:
            query = (
                "SELECT id, title, abstract FROM papers WHERE "
                "(embedding IS NULL OR embedding_model IS NULL OR embedding_model IN ('pending', ''))"
            )
            params: List[Any] = []
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            # Require non-empty title and abstract
            query += " AND LENGTH(TRIM(COALESCE(title, ''))) > 0 AND LENGTH(TRIM(COALESCE(abstract, ''))) > 0"
            query += " ORDER BY date DESC"

            cur.execute(query, params)
            rows = cur.fetchall()
            return [
                {"id": row["id"], "title": row["title"], "abstract": row["abstract"]}
                for row in rows
            ]

    @staticmethod
    def get_papers_missing_embeddings_in_date_range_paginated(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Return papers missing embeddings in date range with pagination.
        
        This method enables streaming/chunked processing for large datasets by
        fetching papers in batches. Memory usage is constant regardless of total
        paper count.
        
        Args:
            start_date: Start date (YYYY-MM-DD format, inclusive)
            end_date: End date (YYYY-MM-DD format, inclusive)
            limit: Maximum papers to return in this batch
            offset: Number of papers to skip (for pagination)
            
        Returns:
            List of paper dicts with id, title, abstract
            
        Example:
            # Process in chunks of 10K
            offset = 0
            while True:
                chunk = PaperRepository.get_papers_missing_embeddings_in_date_range_paginated(
                    "2024-01-01", "2024-12-31", limit=10000, offset=offset
                )
                if not chunk:
                    break
                # Process chunk...
                offset += len(chunk)
        """
        with get_cursor() as cur:
            query = (
                "SELECT id, title, abstract FROM papers WHERE "
                "(embedding IS NULL OR embedding_model IS NULL OR embedding_model IN ('pending', ''))"
            )
            params: List[Any] = []
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            
            # Require non-empty title and abstract
            query += " AND LENGTH(TRIM(COALESCE(title, ''))) > 0 AND LENGTH(TRIM(COALESCE(abstract, ''))) > 0"
            query += " ORDER BY date DESC, id ASC"  # Consistent ordering for pagination
            query += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            rows = cur.fetchall()
            return [
                {"id": row["id"], "title": row["title"], "abstract": row["abstract"]}
                for row in rows
            ]

    @staticmethod
    def count_papers_missing_embeddings_in_date_range(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Return count of papers missing embeddings in date range.
        
        This is more efficient than fetching all papers when you only need the count.
        Useful for progress tracking and determining if processing is needed.
        
        Args:
            start_date: Start date (YYYY-MM-DD format, inclusive)
            end_date: End date (YYYY-MM-DD format, inclusive)
            
        Returns:
            Count of papers missing embeddings
        """
        with get_cursor() as cur:
            query = (
                "SELECT COUNT(*) as count FROM papers WHERE "
                "(embedding IS NULL OR embedding_model IS NULL OR embedding_model IN ('pending', ''))"
            )
            params: List[Any] = []
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            
            # Require non-empty title and abstract
            query += " AND LENGTH(TRIM(COALESCE(title, ''))) > 0 AND LENGTH(TRIM(COALESCE(abstract, ''))) > 0"
            
            cur.execute(query, params)
            row = cur.fetchone()
            return int(row['count']) if row and 'count' in row else 0

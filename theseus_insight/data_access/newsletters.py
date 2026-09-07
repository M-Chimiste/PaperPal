from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID, uuid4

from ..db import get_cursor


class NewsletterRepository:
    """CRUD for `newsletters` table."""

    @staticmethod
    def recent_paper_keys(profile_ids=None):
        with get_cursor() as cur:
            cur.execute("""SELECT paper_keys FROM newsletter_editions
                WHERE created_at >= now() - interval '30 days'
                  AND ((%s::int[] = '{}' AND profile_ids = '{}') OR profile_ids && %s::int[])
                ORDER BY created_at DESC LIMIT 30""", (profile_ids or [], profile_ids or []))
            return {key for row in cur.fetchall() for key in row['paper_keys']}

    @staticmethod
    def save_edition(task_id, content, start_date, end_date, profile_ids, artifact):
        """Atomic, idempotent issue persistence with its evidence and coverage record."""
        import json
        with get_cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", ('edition:' + task_id,))
            cur.execute('SELECT newsletter_id FROM newsletter_editions WHERE task_id=%s', (task_id,))
            existing = cur.fetchone()
            if existing:
                return existing['newsletter_id']
            cur.execute('INSERT INTO newsletters(content,start_date,end_date,date_sent) VALUES(%s,%s,%s,CURRENT_DATE) RETURNING id',
                        (content, start_date, end_date))
            ident = cur.fetchone()['id']
            cur.execute('INSERT INTO newsletter_editions(task_id,newsletter_id,profile_ids,paper_keys,artifact) VALUES(%s,%s,%s,%s,%s)',
                        (task_id, ident, profile_ids or [], [p['key'] for p in artifact.get('papers', [])], json.dumps(artifact)))
            return ident

    @staticmethod
    def insert(newsletter: Any) -> int:
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO newsletters (content, start_date, end_date, date_sent)
                VALUES (%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    newsletter.content,
                    newsletter.start_date,
                    newsletter.end_date,
                    newsletter.date_sent,
                ),
            )
            row = cur.fetchone()
            return row["id"] if row else 0

    @staticmethod
    def all() -> List[Dict[str, Any]]:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM newsletters ORDER BY id DESC")
            return cur.fetchall()


class NewsletterJob:
    """Represents a newsletter generation job with multi-server scoring."""

    def __init__(
        self,
        id: UUID,
        profile_ids: List[int],
        status: str = 'pending',
        use_multi_server: bool = False,
        server_ids: Optional[List[int]] = None,
        scoring_mode: str = 'single',
        papers_to_score: int = 0,
        papers_scored: int = 0,
        research_interests: Optional[str] = None,
        date_range_start: Optional[datetime] = None,
        date_range_end: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.profile_ids = profile_ids
        self.status = status
        self.use_multi_server = use_multi_server
        self.server_ids = server_ids or []
        self.scoring_mode = scoring_mode
        self.papers_to_score = papers_to_score
        self.papers_scored = papers_scored
        self.research_interests = research_interests
        self.date_range_start = date_range_start
        self.date_range_end = date_range_end
        self.created_at = created_at
        self.updated_at = updated_at
        self.completed_at = completed_at
        self.error_message = error_message
        self.result_data = result_data or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsletterJob':
        """Create a NewsletterJob instance from a dictionary."""
        return cls(
            id=data['id'],
            profile_ids=data.get('profile_ids', []),
            status=data.get('status', 'pending'),
            use_multi_server=data.get('use_multi_server', False),
            server_ids=data.get('server_ids', []),
            scoring_mode=data.get('scoring_mode', 'single'),
            papers_to_score=data.get('papers_to_score', 0),
            papers_scored=data.get('papers_scored', 0),
            research_interests=data.get('research_interests'),
            date_range_start=data.get('date_range_start'),
            date_range_end=data.get('date_range_end'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            completed_at=data.get('completed_at'),
            error_message=data.get('error_message'),
            result_data=data.get('result_data', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': str(self.id),
            'profile_ids': self.profile_ids,
            'status': self.status,
            'use_multi_server': self.use_multi_server,
            'server_ids': self.server_ids,
            'scoring_mode': self.scoring_mode,
            'papers_to_score': self.papers_to_score,
            'papers_scored': self.papers_scored,
            'research_interests': self.research_interests,
            'date_range_start': self.date_range_start.isoformat() if self.date_range_start else None,
            'date_range_end': self.date_range_end.isoformat() if self.date_range_end else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'result_data': self.result_data
        }


class NewsletterJobRepository:
    """Repository for managing newsletter job operations."""

    @staticmethod
    def create_job(
        profile_ids: List[int],
        use_multi_server: bool = False,
        server_ids: Optional[List[int]] = None,
        research_interests: Optional[str] = None,
        date_range_start: Optional[datetime] = None,
        date_range_end: Optional[datetime] = None
    ) -> UUID:
        """Create a new newsletter job and return its ID."""
        job_id = uuid4()
        scoring_mode = 'multi-server' if use_multi_server else 'single'

        with get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO newsletter_jobs (
                    id, profile_ids, use_multi_server, server_ids, scoring_mode,
                    research_interests, date_range_start, date_range_end, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                str(job_id),
                profile_ids,
                use_multi_server,
                server_ids or [],
                scoring_mode,
                research_interests,
                date_range_start,
                date_range_end,
                'pending'
            ))

            row = cursor.fetchone()
            # row['id'] is already a UUID object from psycopg, don't wrap it again
            return row['id'] if row else job_id

    @staticmethod
    def get_job(job_id: UUID) -> Optional[NewsletterJob]:
        """Get a newsletter job by ID."""
        with get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM newsletter_jobs WHERE id = %s
            """, (str(job_id),))

            row = cursor.fetchone()
            return NewsletterJob.from_dict(dict(row)) if row else None

    @staticmethod
    def update_job_status(
        job_id: UUID,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of a newsletter job."""
        with get_cursor() as cursor:
            if status in ('completed', 'failed', 'canceled'):
                cursor.execute("""
                    UPDATE newsletter_jobs
                    SET status = %s,
                        error_message = %s,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, error_message, str(job_id)))
            else:
                cursor.execute("""
                    UPDATE newsletter_jobs
                    SET status = %s,
                        error_message = %s
                    WHERE id = %s
                """, (status, error_message, str(job_id)))

    @staticmethod
    def update_job_progress(
        job_id: UUID,
        papers_to_score: Optional[int] = None,
        papers_scored: Optional[int] = None
    ) -> None:
        """Update the scoring progress of a newsletter job."""
        with get_cursor() as cursor:
            if papers_to_score is not None and papers_scored is not None:
                cursor.execute("""
                    UPDATE newsletter_jobs
                    SET papers_to_score = %s,
                        papers_scored = %s
                    WHERE id = %s
                """, (papers_to_score, papers_scored, str(job_id)))
            elif papers_scored is not None:
                cursor.execute("""
                    UPDATE newsletter_jobs
                    SET papers_scored = %s
                    WHERE id = %s
                """, (papers_scored, str(job_id)))
            elif papers_to_score is not None:
                cursor.execute("""
                    UPDATE newsletter_jobs
                    SET papers_to_score = %s
                    WHERE id = %s
                """, (papers_to_score, str(job_id)))

    @staticmethod
    def increment_papers_scored(job_id: UUID, increment: int = 1) -> None:
        """Increment the papers_scored counter for a job."""
        with get_cursor() as cursor:
            cursor.execute("""
                UPDATE newsletter_jobs
                SET papers_scored = papers_scored + %s
                WHERE id = %s
            """, (increment, str(job_id)))

    @staticmethod
    def complete_job(
        job_id: UUID,
        result_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a newsletter job as completed with optional result data."""
        with get_cursor() as cursor:
            cursor.execute("""
                UPDATE newsletter_jobs
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    result_data = %s
                WHERE id = %s
            """, (result_data or {}, str(job_id)))

    @staticmethod
    def fail_job(job_id: UUID, error_message: str) -> None:
        """Mark a newsletter job as failed with error message and clean up tasks."""
        with get_cursor() as cursor:
            # Mark job as failed
            cursor.execute("""
                UPDATE newsletter_jobs
                SET status = 'failed',
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = %s
                WHERE id = %s
            """, (error_message, str(job_id)))
            
            # Clean up all tasks for this failed job
            cursor.execute("""
                DELETE FROM judge_task_queue
                WHERE job_id = %s AND job_type = 'newsletter'
            """, (str(job_id),))

    @staticmethod
    def cleanup_failed_job_tasks() -> int:
        """
        Clean up tasks for all failed/canceled newsletter jobs.
        Returns the number of tasks deleted.
        """
        with get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM judge_task_queue
                WHERE job_type = 'newsletter'
                  AND job_id IN (
                    SELECT id FROM newsletter_jobs
                    WHERE status IN ('failed', 'canceled')
                  )
            """)
            return cursor.rowcount

    @staticmethod
    def get_job_progress(job_id: UUID) -> Dict[str, Any]:
        """Get progress statistics for a newsletter job."""
        with get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM newsletter_scoring_progress
                WHERE newsletter_job_id = %s
            """, (str(job_id),))

            row = cursor.fetchone()
            return dict(row) if row else {}

    @staticmethod
    def get_job_server_stats(job_id: UUID) -> List[Dict[str, Any]]:
        """Get per-server statistics for a newsletter job."""
        with get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM newsletter_server_stats
                WHERE newsletter_job_id = %s
                ORDER BY assigned_server_url
            """, (str(job_id),))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_active_jobs() -> List[NewsletterJob]:
        """Get all active (non-completed/failed/canceled) newsletter jobs."""
        with get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM newsletter_jobs
                WHERE status IN ('pending', 'scoring', 'generating')
                ORDER BY created_at DESC
            """)

            return [NewsletterJob.from_dict(dict(row)) for row in cursor.fetchall()]

    @staticmethod
    def get_recent_jobs(limit: int = 20) -> List[NewsletterJob]:
        """Get recent newsletter jobs."""
        with get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM newsletter_jobs
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            return [NewsletterJob.from_dict(dict(row)) for row in cursor.fetchall()]

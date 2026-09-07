"""Startup never kills processes or resets another instance's jobs.

Durable task recovery is handled by task_dispatch leases plus advisory locks.
Bulk judge workers own their Popen handles and job queue leases; unrelated
processes and legacy unowned jobs are left alone for explicit review.
"""
import logging


async def cleanup_stuck_jobs_and_processes():
    logging.getLogger(__name__).info("Recovery delegated to owned task leases; no global process cleanup")

from ..workers.ownership import worker_environment
"""
Newsletter Scorer - Multi-server orchestration for newsletter LLM judge scoring.

This module provides the NewsletterScorer class which orchestrates distributed
paper scoring across multiple inference servers for newsletter generation.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from uuid import UUID

from ..data_access.newsletters import NewsletterJobRepository
from ..data_access.judge_task_queue import JudgeTaskQueueRepository
from ..data_access.inference_servers import InferenceServersRepository
from ..data_access.settings import SettingsRepository

logger = logging.getLogger(__name__)


class NewsletterScorer:
    """
    Orchestrates multi-server scoring for newsletter generation.
    Similar to BulkJudgeRunner but tailored for newsletter workflow.
    """

    def __init__(self, orchestration_config: Dict[str, Any]):
        """
        Initialize the newsletter scorer.

        Args:
            orchestration_config: Configuration dict from settings containing judge_model config
        """
        self.config = orchestration_config
        self.judge_model_config = orchestration_config.get('judge_model', {})

    async def score_papers_multi_server(
        self,
        job_id: UUID,
        papers: List[Dict[str, Any]],
        profile_ids: List[int],
        server_ids: List[int],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        request_timeout_sec: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Score papers using multi-server worker pool.

        Creates scoring tasks for each (paper, profile) combination.
        Scores are stored in paper_profile_scores and aggregated across profiles.

        Args:
            job_id: Newsletter job UUID
            papers: List of paper dicts with id, title, abstract
            profile_ids: List of profile IDs to score against
            server_ids: List of inference server IDs to use
            progress_callback: Optional callback for progress updates (status, progress)
            request_timeout_sec: Optional timeout for LLM requests
            max_retries: Optional max retry count for failed tasks

        Returns:
            Sorted list of papers with aggregated scores across profiles
        """
        try:
            # 1. Validate servers are enabled and healthy
            logger.info(f"Newsletter job {job_id}: Loading {len(server_ids)} inference servers")
            servers = InferenceServersRepository.get_by_ids(server_ids)

            if not servers:
                raise ValueError(f"No valid servers found for IDs: {server_ids}")

            enabled_servers = [s for s in servers if s.enabled]
            if not enabled_servers:
                raise ValueError("No enabled servers found in selection")

            logger.info(f"Using {len(enabled_servers)} enabled servers for newsletter scoring")

            # 2. Update job status
            NewsletterJobRepository.update_job_status(job_id, 'scoring')
            total_tasks = len(papers) * len(profile_ids)
            NewsletterJobRepository.update_job_progress(
                job_id,
                papers_to_score=total_tasks,
                papers_scored=0
            )

            if progress_callback:
                progress_callback('scoring', 0.0)

            # 3. Enqueue scoring tasks (creates tasks for each paper-profile combination)
            logger.info(f"Enqueueing {len(papers)} papers × {len(profile_ids)} profiles = {total_tasks} scoring tasks")

            # Don't pre-assign tasks to servers - use dynamic queue-based distribution
            task_count = JudgeTaskQueueRepository.enqueue_newsletter_tasks(
                job_id=job_id,
                papers=papers,
                profile_ids=profile_ids,
                server_urls=None  # Use dynamic pooling
            )

            logger.info(f"Enqueued {task_count} scoring tasks")

            # 4. Launch worker processes (one per server)
            await self._launch_workers(
                job_id=job_id,
                servers=enabled_servers,
                timeout=request_timeout_sec,
                max_retries=max_retries
            )

            # 5. Monitor progress and broadcast updates
            await self._monitor_scoring_progress(
                job_id=job_id,
                total_tasks=task_count,
                progress_callback=progress_callback,
                servers=enabled_servers,
                paper_count=len(papers),
                profile_count=len(profile_ids)
            )

            # 6. Retrieve results
            logger.info(f"Retrieving scoring results for job {job_id}")
            results = JudgeTaskQueueRepository.get_newsletter_results(job_id)

            logger.info(f"Retrieved {len(results)} scored papers")

            # 7. Update job status to complete scoring phase
            NewsletterJobRepository.update_job_status(job_id, 'generating')

            return results

        except Exception as e:
            logger.error(f"Newsletter scoring failed for job {job_id}: {e}")
            NewsletterJobRepository.fail_job(job_id, str(e))
            raise

    def score_papers_single_server(
        self,
        papers: List[Dict[str, Any]],
        research_interests: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Score papers using single judge model (sequential processing).
        This is a wrapper that delegates to the existing TheseusInsight implementation.

        Args:
            papers: List of paper dicts
            research_interests: Research interests to score against
            progress_callback: Optional callback for progress updates

        Returns:
            List of papers with scores

        Note: This method is kept for backward compatibility and should delegate
        to TheseusInsight._rank_papers_by_research_interests() in the actual implementation.
        """
        # This will be implemented by calling the existing single-server scoring
        # from TheseusInsight class
        raise NotImplementedError(
            "Single-server scoring should use TheseusInsight._rank_papers_by_research_interests()"
        )

    async def _launch_workers(
        self,
        job_id: UUID,
        servers: List[Any],
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        """Launch judge worker processes for each server."""
        # Get default timeout and max_retries if not provided
        if timeout is None:
            timeout = SettingsRepository.get_int("ollama_request_timeout_sec", 60)
        if max_retries is None:
            max_retries = SettingsRepository.get_int("ollama_max_retries", 3)

        logger.info(f"Launching {len(servers)} worker processes for newsletter job {job_id}")
        logger.info(f"Worker config: timeout={timeout}s, max_retries={max_retries}")

        successful_launches = 0
        failed_launches = []

        for i, server in enumerate(servers):
            try:
                logger.info(f"[{i+1}/{len(servers)}] Launching worker for server: {server.name} ({server.url})")

                # Build worker command
                cmd = [
                    sys.executable, "-m", "theseus_insight.workers.judge_worker",
                    "--job-id", str(job_id),
                    "--server-url", server.url,
                    "--timeout", str(timeout),
                    "--max-retries", str(max_retries)
                ]

                # Add provider type
                if hasattr(server, 'provider') and server.provider:
                    cmd.extend(["--provider", server.provider])

                # Add per-server model overrides if specified
                if hasattr(server, 'model_name') and server.model_name:
                    cmd.extend(["--server-model-name", server.model_name])
                    logger.info(f"Using per-server model: {server.model_name}")

                if hasattr(server, 'model_config') and server.model_config:
                    import json
                    cmd.extend(["--server-model-config", json.dumps(server.model_config)])
                    logger.debug(f"Using per-server config: {server.model_config}")

                # Launch worker process in background
                # Send stdout/stderr to DEVNULL to avoid blocking on pipe buffers
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=os.getcwd(),
                    env=worker_environment()
                )

                logger.info(f"Launched worker process PID {process.pid} for {server.name}")

                # Give process time to start
                time.sleep(1.0)

                if process.poll() is None:
                    logger.info(f"Worker {process.pid} running for {server.name}")
                    successful_launches += 1
                else:
                    logger.error(f"Worker {process.pid} failed to start for {server.name} (exit code: {process.returncode})")
                    failed_launches.append(server.name)

            except Exception as e:
                logger.error(f"Failed to launch worker for {server.name}: {e}")
                failed_launches.append(server.name)

        logger.info(f"Worker launch summary: {successful_launches}/{len(servers)} successful")

        if failed_launches:
            logger.warning(f"Failed to launch workers for: {', '.join(failed_launches)}")

        if successful_launches == 0:
            raise RuntimeError("Failed to launch any worker processes")

    async def _monitor_scoring_progress(
        self,
        job_id: UUID,
        total_tasks: int,
        progress_callback: Optional[Callable[[str, float, str, Optional[Dict]], None]] = None,
        poll_interval_sec: int = 5,
        servers: Optional[List[Any]] = None,
        paper_count: Optional[int] = None,
        profile_count: Optional[int] = None
    ):
        """
        Monitor scoring job progress and broadcast updates.

        Args:
            job_id: Newsletter job UUID
            total_tasks: Total number of scoring tasks
            progress_callback: Optional callback for progress updates
            poll_interval_sec: How often to poll for progress (default 5 seconds)
        """
        logger.info(f"Monitoring newsletter scoring progress for job {job_id}")

        max_wait_minutes = 60  # Maximum time to wait for completion
        max_iterations = (max_wait_minutes * 60) // poll_interval_sec
        iteration = 0

        # Build lookup for server metadata (names/ids) by URL for richer UI display
        server_lookup: Dict[str, Any] = {}
        if servers:
            for server in servers:
                url = getattr(server, 'url', None)
                if url:
                    server_lookup[url] = server

        while iteration < max_iterations:
            try:
                # Get current progress
                progress = JudgeTaskQueueRepository.get_job_progress(job_id)

                completed = progress.get('completed_tasks', 0)
                failed = progress.get('failed_tasks', 0)
                pending = progress.get('pending_tasks', 0)
                in_progress = progress.get('in_progress_tasks', 0)

                # Calculate progress percentage
                finished = completed + failed
                progress_pct = (finished / total_tasks) * 100 if total_tasks > 0 else 0

                logger.debug(
                    f"Job {job_id} progress: {finished}/{total_tasks} "
                    f"({progress_pct:.1f}%) - "
                    f"completed={completed}, failed={failed}, "
                    f"pending={pending}, in_progress={in_progress}"
                )

                # Update newsletter job progress
                NewsletterJobRepository.update_job_progress(
                    job_id,
                    papers_scored=completed
                )

                # Get per-server statistics from judge_task_queue (has better timestamps for throughput calc)
                server_stats = JudgeTaskQueueRepository.get_job_server_stats(job_id)

                stats_by_url = {}
                for stat in (server_stats or []):
                    url = stat.get('assigned_server_url')
                    if url:
                        # Normalize URL for lookup (strip trailing slash)
                        stats_by_url[url.rstrip('/')] = stat
                        # Also keep original for exact matches
                        stats_by_url[url] = stat

                formatted_server_stats: List[Dict[str, Any]] = []
                
                def _calculate_stats(stat: Dict[str, Any]) -> Dict[str, Any]:
                    completed_tasks = stat.get('completed_tasks', 0) or 0
                    
                    # Calculate throughput
                    avg_latency = None
                    throughput = 0.0
                    
                    # Method 1 (preferred): Calculate from timestamps - accounts for parallel processing
                    first_task = stat.get('first_task_at')
                    # Support both column names from different sources
                    last_update = stat.get('last_completed_at') or stat.get('last_update_at')
                    
                    if completed_tasks > 0 and first_task and last_update:
                        # Ensure timezone awareness compatibility if needed
                        if first_task.tzinfo is None:
                            first_task = first_task.replace(tzinfo=timezone.utc)
                        if last_update.tzinfo is None:
                            last_update = last_update.replace(tzinfo=timezone.utc)
                            
                        duration = (last_update - first_task).total_seconds()
                        if duration > 1.0:  # Avoid division by zero or tiny intervals
                            avg_latency = duration / completed_tasks
                            throughput = (completed_tasks / duration) * 60  # Papers per minute
                    
                    # Method 2 (fallback): Use avg_task_duration_seconds 
                    # Note: This assumes serial processing, so underestimates parallel throughput
                    if throughput == 0.0:
                        avg_duration = stat.get('avg_task_duration_seconds')
                        if avg_duration and float(avg_duration) > 0:
                            avg_latency = float(avg_duration)  # Convert Decimal to float
                            throughput = 60.0 / avg_latency  # Papers per minute (serial estimate)
                    
                    return avg_latency, throughput

                if server_lookup:
                    for url, server in server_lookup.items():
                        stat = stats_by_url.get(url) or stats_by_url.get(url.rstrip('/'), {})
                        
                        total = stat.get('total_tasks', 0) or 0
                        completed_tasks = stat.get('completed_tasks', 0) or 0
                        failed_tasks = stat.get('failed_tasks', 0) or 0
                        active_tasks = stat.get('active_tasks', 0) or 0
                        
                        avg_latency, throughput = _calculate_stats(stat)

                        formatted_server_stats.append({
                            'server_id': str(getattr(server, 'id', url)),
                            'server_name': getattr(server, 'name', url),
                            'server_url': url,
                            'total': total,
                            'completed': completed_tasks,
                            'in_progress': active_tasks,
                            'failed': failed_tasks,
                            'papers_processed': completed_tasks,
                            'papers_failed': failed_tasks,
                            'avg_latency': avg_latency,
                            'throughput': throughput,
                            'last_completed_at': stat.get('last_completed_at').isoformat()
                                if stat.get('last_completed_at') else None,
                            'status': 'busy' if active_tasks > 0 else ('idle' if total > 0 else 'offline'),
                        })
                else:
                    # Deduplicate stats if we're iterating blindly
                    seen_urls = set()
                    for stat in (server_stats or []):
                        server_url = stat.get('assigned_server_url')
                        if not server_url or server_url in seen_urls:
                            continue
                        seen_urls.add(server_url)
                        
                        server_info = server_lookup.get(server_url, {})
                        total = stat.get('total_tasks', 0) or 0
                        completed_tasks = stat.get('completed_tasks', 0) or 0
                        failed_tasks = stat.get('failed_tasks', 0) or 0
                        active_tasks = stat.get('active_tasks', 0) or 0

                        avg_latency, throughput = _calculate_stats(stat)

                        formatted_server_stats.append({
                            'server_id': str(getattr(server_info, 'id', server_url)),
                            'server_name': getattr(server_info, 'name', server_url),
                            'server_url': server_url,
                            'total': total,
                            'completed': completed_tasks,
                            'in_progress': active_tasks,
                            'failed': failed_tasks,
                            'papers_processed': completed_tasks,
                            'papers_failed': failed_tasks,
                            'avg_latency': avg_latency,
                            'throughput': throughput,
                            'last_completed_at': stat.get('last_completed_at').isoformat()
                                if stat.get('last_completed_at') else None,
                            'status': 'busy' if active_tasks > 0 else ('idle' if total > 0 else 'offline'),
                        })

                # NEW: Build structured metadata
                summary = {
                    'completed': completed,
                    'failed': failed,
                    'pending': pending,
                    'in_progress': in_progress,
                    'total': total_tasks,
                    'pending_plus_in_progress': pending + in_progress
                }

                metadata = {
                    'newsletter_job_id': str(job_id),
                    'papers_discovered': paper_count,
                    'papers_to_score': total_tasks,
                    'papers_scored': completed,
                    'papers_failed': failed,
                    'papers_pending': pending,
                    'papers_in_progress': in_progress,
                    'profile_count': profile_count,
                    'server_stats': formatted_server_stats,
                    'avg_task_duration': progress.get('avg_task_duration_seconds'),
                    'estimated_time_remaining': progress.get('estimated_time_remaining_seconds')
                }
                metadata['scoring_summary'] = summary

                # Call progress callback with metadata
                if progress_callback:
                    message = f"Scoring papers: {completed}/{total_tasks} completed"
                    if formatted_server_stats:
                        active_servers = sum(1 for s in formatted_server_stats if s.get('in_progress', 0) > 0)
                        message += f" ({active_servers} servers active)"
                    
                    # Pass message and metadata
                    try:
                        progress_callback('scoring', progress_pct, message, metadata)
                    except TypeError:
                        # Fallback for old callback signature
                        progress_callback('scoring', progress_pct)

                # Check if all tasks are finished
                if finished >= total_tasks:
                    logger.info(f"All tasks finished for job {job_id}: {completed} completed, {failed} failed")
                    break

                # Check for stalled job (no progress for too long)
                if iteration > 20 and pending > 0 and in_progress == 0:
                    logger.warning(
                        f"Job {job_id} appears stalled: {pending} tasks pending but no workers active"
                    )

            except Exception as e:
                logger.error(f"Error monitoring progress for job {job_id}: {e}")

            # Wait before next poll
            await asyncio.sleep(poll_interval_sec)
            iteration += 1

        if iteration >= max_iterations:
            logger.error(f"Job {job_id} monitoring timed out after {max_wait_minutes} minutes")
            raise TimeoutError(f"Newsletter scoring job timed out after {max_wait_minutes} minutes")

        logger.info(f"Finished monitoring job {job_id}")

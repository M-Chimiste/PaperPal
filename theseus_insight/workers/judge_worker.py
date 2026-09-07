#!/usr/bin/env python3
"""
Theseus Judge Worker - Simple synchronous worker with single persistent connection.

This worker processes judge tasks synchronously using one database connection
for the entire worker lifetime to avoid connection pool issues.

Usage:
    python -m theseus_insight.workers.judge_worker --server-url http://localhost:11434 --job-id <uuid>
    python -m theseus_insight.workers.judge_worker --all-enabled  # Process all enabled servers
"""

import argparse
import json
import logging
import signal
import sys
import os
import time
import psycopg
import threading
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timedelta


class PaperProcessingError(Exception):
    """
    Exception for paper-specific errors that should NOT trigger the circuit breaker.
    
    Use this for errors related to the paper content (bad abstract, unparseable response, etc.)
    rather than server-level errors (connection refused, timeout, server unavailable).
    
    Paper errors mark the task as failed but don't count toward consecutive_failures,
    allowing the worker to continue processing other papers.
    """
    pass

# Configure logging
# Set up both console and file logging for worker diagnostics
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'logs')
os.makedirs(log_dir, exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler - separate log file per worker (will be set in __init__)
# This will be added per-worker instance
logger.addHandler(console_handler)

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://theseus:theseus@localhost:5432/theseusdb")


class JudgeWorker:
    """Worker with single persistent database connection."""

    def __init__(
        self,
        server_url: str,
        provider: str = "ollama",
        config_json: Optional[dict] = None,
        job_id: Optional[UUID] = None,
        worker_id: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        judge_model_config: Optional[dict] = None,
        server_model_name: Optional[str] = None,
        server_model_config: Optional[dict] = None
    ):
        self.server_url = server_url
        self.provider = provider
        self.config_json = config_json or {}
        self.job_id = job_id
        self.worker_id = worker_id or f"worker_{provider}_{server_url.replace('://', '_').replace('/', '_')}"
        self.max_retries = max_retries
        self.timeout = timeout
        self.running = False
        self.tasks_processed = 0
        self.consecutive_failures = 0
        self.consecutive_empty_checks = 0  # Track empty queue checks
        self.last_health_check = 0
        self.server_healthy = True
        
        # Create single persistent connection with autocommit to avoid transaction state conflicts
        # When using cursor context managers with a persistent connection, autocommit=True
        # prevents psycopg3 internal deadlocks on commit()
        self.conn = psycopg.connect(DATABASE_URL, autocommit=True)
        # Enforce sane timeouts to prevent indefinite waits on DB operations
        try:
            with self.conn.cursor() as _cur:
                _cur.execute("SET statement_timeout = '10s'")
                _cur.execute("SET lock_timeout = '3s'")
                _cur.execute("SET idle_in_transaction_session_timeout = '10s'")
        except Exception as _e:
            logger.warning(f"Failed to apply DB session timeouts: {_e}")
        
        # Cache for profile research interests to avoid repeated DB queries
        self.interests_cache = {}
        self.profile_cache = {}
        
        # Load judge model config from database settings
        self.judge_model_config = self._load_judge_config()
        
        # Apply per-server overrides if provided
        model_name = server_model_name or self.judge_model_config.get('model_name', 'phi4-mini:3.8b-q8_0')
        max_new_tokens = self.judge_model_config.get('max_new_tokens', 512)
        temperature = self.judge_model_config.get('temperature', 0.1)
        num_ctx = self.judge_model_config.get('num_ctx', 4096)
        
        # Apply server_model_config overrides if provided
        if server_model_config:
            max_new_tokens = server_model_config.get('max_new_tokens', max_new_tokens)
            temperature = server_model_config.get('temperature', temperature)
            num_ctx = server_model_config.get('num_ctx', num_ctx)
        
        # Store model config for potential client reinitialization (LM Studio auto-unload recovery)
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._num_ctx = num_ctx
        
        logger.info(f"Loaded judge model config: {model_name}")
        
        # Initialize inference client based on provider using LLMFactory pattern
        if provider == "ollama":
            from LLMFactory.providers import OllamaInference
            self.inference_client = OllamaInference(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_ctx=num_ctx,
                url=server_url,
                request_timeout=timeout
            )
        elif provider == "lmstudio":
            from theseus_insight.utils.lmstudio_client import get_lmstudio_client
            # Extract host from server_url (e.g., http://localhost:1234 -> localhost:1234)
            host = server_url.replace('http://', '').replace('https://', '')
            self._lmstudio_host = host  # Store for reinitialization
            self.inference_client = get_lmstudio_client(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                host=host,
                context_length=num_ctx
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Using judge model: {model_name} (provider: {provider})")
        if server_model_name or server_model_config:
            logger.info(f"Per-server model override active: model_name={server_model_name}, custom_params={bool(server_model_config)}")

        # Add file handler for this specific worker
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'logs')
        log_file = os.path.join(log_dir, f'judge_worker_{self.worker_id}.log')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        self.log_file = log_file

        # Watchdog to dump stack traces if we stall
        self.last_progress_ts = time.time()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

        logger.info(f"Initialized {provider} worker {self.worker_id} for server {server_url}")
        logger.info(f"Logging to {log_file}")

    def _load_judge_config(self) -> dict:
        """Load judge model configuration from database settings."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT value FROM settings WHERE key = 'orchestration'")
                result = cur.fetchone()
                if result and result[0]:
                    orchestration_config = json.loads(result[0])
                    judge_config = orchestration_config.get('judge_model', {})
                    if judge_config:
                        logger.info(f"Loaded judge config from database: {judge_config.get('model_name', 'unknown')}")
                        return judge_config
                    else:
                        logger.warning("No judge_model found in orchestration config, using defaults")
                else:
                    logger.warning("No orchestration config found in database, using defaults")
        except Exception as e:
            logger.error(f"Failed to load judge config from database: {e}")
        
        # Return default config if loading fails
        return {
            'model_name': 'phi4-mini:3.8b-q8_0',
            'max_new_tokens': 512,
            'temperature': 0.1,
            'num_ctx': 4096
        }

    def _reinitialize_lmstudio_client(self):
        """
        Verify LM Studio connection and trigger model reload if needed.
        
        LM Studio may auto-unload models after extended periods of operation.
        The LMStudio SDK uses a singleton pattern that can't be reset within
        a process, so we verify the connection and reuse the existing client.
        """
        if self.provider != "lmstudio":
            return
        
        logger.info(f"Verifying LM Studio connection for {self._model_name}...")
        
        try:
            # Verify that LM Studio has models loaded
            from theseus_insight.utils.lmstudio_client import verify_model_loaded, get_lmstudio_client
            
            if not verify_model_loaded(host=self._lmstudio_host, model_name=self._model_name):
                logger.warning(f"LM Studio model {self._model_name} may not be loaded, waiting 5s...")
                time.sleep(5)
                # Check again
                if not verify_model_loaded(host=self._lmstudio_host, model_name=self._model_name):
                    logger.warning(f"Model still not detected - user may need to load it in LM Studio")
            
            # The existing client should still work - just reuse it from cache
            # Don't clear cache or try to recreate (SDK singleton limitation)
            cached_client = get_lmstudio_client(
                model_name=self._model_name,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                host=self._lmstudio_host,
                context_length=self._num_ctx
            )
            
            if cached_client:
                self.inference_client = cached_client
                logger.info(f"LM Studio client verified and ready")
            else:
                logger.warning(f"Could not get LM Studio client from cache")
                
        except Exception as e:
            logger.error(f"Failed to verify LM Studio client: {e}")
            # Don't raise - let the caller handle retry logic

    def start(self):
        """Start the worker process."""
        logger.info(f"Starting worker {self.worker_id} for job {self.job_id}")
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        from .ownership import watch_parent
        watch_parent(self)
        
        last_heartbeat = time.time()
        last_lease_cleanup = time.time()
        heartbeat_interval = 30  # seconds
        lease_cleanup_interval = 60  # seconds
        
        try:
            while self.running:
                try:
                    # Mark progress for watchdog
                    self.last_progress_ts = time.time()
                    logger.debug("Loop tick start")
                    # Send heartbeat periodically
                    now = time.time()
                    if (now - last_heartbeat) >= heartbeat_interval:
                        logger.debug("Sending heartbeat...")
                        self._send_heartbeat()
                        logger.debug("Heartbeat sent")
                        last_heartbeat = now
                    
                    # Clean up expired leases periodically
                    if (now - last_lease_cleanup) >= lease_cleanup_interval:
                        logger.debug("Running lease cleanup...")
                        expired_count = self._requeue_expired_leases()
                        logger.debug(f"Lease cleanup completed ({expired_count} requeued)")
                        if expired_count > 0:
                            logger.info(f"Requeued {expired_count} expired leases")
                        last_lease_cleanup = now
                    
                    # Check Ollama server health every 60 seconds (informational only)
                    if now - self.last_health_check > 60:
                        logger.debug("Checking Ollama health...")
                        self._check_ollama_health()
                        logger.debug("Health check completed")
                        self.last_health_check = now
                    
                    # Note: We don't skip processing based on health check alone
                    # Let actual task failures determine if server is down
                    
                    # Try to lease a task
                    logger.debug(f"Attempting to lease task for {self.server_url}")
                    task = self._lease_next_task()
                    
                    if task:
                        logger.info(f"Leased task {task['id']} (paper {task['paper_id']}, profile {task['profile_id']})")
                        self._process_task(task)
                        
                        if self.tasks_processed % 10 == 0:
                            logger.info(f"Worker {self.worker_id}: {self.tasks_processed} tasks completed")
                        
                        # Log recovery if we had consecutive failures
                        if self.consecutive_failures >= 3:
                            logger.info(f"Worker recovered after {self.consecutive_failures} consecutive failures")
                        
                        # Reset consecutive failures and empty checks on success
                        self.consecutive_failures = 0
                        self.consecutive_empty_checks = 0
                    else:
                        # No tasks available - increment counter
                        self.consecutive_empty_checks += 1
                        logger.debug(f"No tasks available for {self.server_url} (check {self.consecutive_empty_checks}/6)")
                        
                        # Exit gracefully if queue has been empty for too long
                        if self.consecutive_empty_checks >= 6:
                            logger.info(f"Worker {self.worker_id}: No tasks found after {self.consecutive_empty_checks} checks. All work appears done - exiting gracefully.")
                            break
                        
                        time.sleep(5)
                        
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    
                    # Determine if this is a server-level error that should trigger circuit breaker
                    error_str = str(e).lower()
                    
                    # LM Studio "model not loaded" errors are transient - model may still be loaded
                    # but LM Studio's API is temporarily reporting it as unavailable
                    is_model_loading_error = any(term in error_str for term in [
                        'no model found', 'nomodelmatchingquery', 'totalloadedmodels',
                        'model not loaded', 'model unavailable', 'no model matching'
                    ])
                    
                    # True server errors indicate the server itself is unreachable
                    is_server_error = any(term in error_str for term in [
                        'connection refused', 'timeout', 'unreachable',
                        'network', 'socket', 'eof', 'reset', 'broken pipe', 'no route',
                        'ssl', 'certificate', 'handshake', 'dns'
                    ])
                    
                    if is_model_loading_error:
                        # LM Studio model loading issue - retry but don't count toward circuit breaker
                        # The model may still be loaded, just temporarily unavailable via API
                        logger.warning(f"LM Studio model loading error (not counting toward circuit breaker): {e}")
                    elif is_server_error:
                        self.consecutive_failures += 1
                        logger.warning(f"Server error in main loop, consecutive failures: {self.consecutive_failures}")
                    else:
                        # Other errors - log but don't count toward circuit breaker
                        logger.info(f"Non-server error in main loop, not incrementing consecutive failures")
                    
                    # If too many server failures, exit
                    if self.consecutive_failures >= 10:
                        logger.error("Too many consecutive server failures, shutting down")
                        break
                    
                    time.sleep(10)
        finally:
            self._cleanup()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.running = False
    
    def _watchdog_loop(self):
        """Watch for stalls and log a brief warning (full dumps written to file only)."""
        import traceback, sys
        stall_threshold = 45  # seconds without progress
        stall_reported = False  # Only report once per stall
        while True:
            try:
                time.sleep(15)
                if not self.running:
                    return
                idle = time.time() - getattr(self, 'last_progress_ts', time.time())
                if idle > stall_threshold:
                    if not stall_reported:
                        # Log a brief warning (not the full stack traces)
                        logger.warning(f"Watchdog: worker stalled for {int(idle)}s waiting for LLM response. Will recover after timeout.")
                        stall_reported = True
                        
                        # Write detailed stack traces to file only (for debugging if needed)
                        try:
                            with open(self.log_file, 'a') as f:
                                f.write(f"\n==== Watchdog dump at {time.strftime('%Y-%m-%d %H:%M:%S')} (idle {int(idle)}s) ====\n")
                                for thread_id, frame in sys._current_frames().items():
                                    f.write(f"\n--- Thread {thread_id} ---\n")
                                    f.write(''.join(traceback.format_stack(frame)))
                        except Exception:
                            pass
                else:
                    # Worker is making progress - reset the stall flag
                    stall_reported = False
            except Exception:
                # Watchdog should never crash the worker
                pass
    
    def _check_ollama_health(self):
        """Check if the Ollama server is responding."""
        import requests
        try:
            # Try to get the version endpoint with a short timeout
            # Use a session to avoid connection pool exhaustion
            with requests.Session() as session:
                response = session.get(f"{self.server_url}/api/version", timeout=5)
                if response.status_code == 200:
                    if not self.server_healthy:
                        logger.info(f"Ollama server {self.server_url} is now healthy")
                    self.server_healthy = True
                else:
                    if self.server_healthy:
                        logger.warning(f"Ollama server {self.server_url} returned status {response.status_code}")
                    self.server_healthy = False
        except Exception as e:
            # Log but don't immediately mark as unhealthy - could be transient
            logger.warning(f"Ollama server health check failed (will retry): {e}")
            # Don't set server_healthy = False here - let actual task failures determine that

    def _lease_next_task(self):
        """Lease the next available task."""
        task = None
        try:
            leased_until = datetime.now() + timedelta(minutes=5)
            
            # Use cursor context manager to execute queries
            with self.conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                # First check if there are ANY pending tasks for this server
                cur.execute("""
                    SELECT COUNT(*) as count FROM judge_task_queue
                    WHERE status = 'pending'
                    AND (assigned_server_url IS NULL OR assigned_server_url = %s)
                """, (self.server_url,))
                count_result = cur.fetchone()
                pending_count = count_result['count'] if count_result else 0
                
                if pending_count == 0:
                    logger.debug(f"No pending tasks for {self.server_url}")
                    # Don't commit/rollback here - do it outside the with block
                    return None
                
                logger.debug(f"Found {pending_count} pending tasks for {self.server_url}, attempting lease")
                
                # Now try to lease one
                cur.execute("""
                    UPDATE judge_task_queue
                    SET status = 'leased',
                        assigned_server_url = %s,
                        leased_until = %s,
                        leased_by_worker = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = (
                        SELECT id FROM judge_task_queue
                        WHERE status = 'pending'
                        AND (assigned_server_url IS NULL OR assigned_server_url = %s)
                        ORDER BY created_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING id, job_id, paper_id, profile_id, attempts
                """, (self.server_url, leased_until, self.worker_id, self.server_url))
                
                row = cur.fetchone()
                task = dict(row) if row else None
                
                # Log while still inside cursor context to avoid accessing closed cursor data
                if task:
                    logger.debug(f"Successfully leased task {task['id']}")
                else:
                    logger.warning(f"Failed to lease task (likely locked by another worker)")
            
            # Return task (row data is safe to use after cursor closes in psycopg3)
            return task
            
        except Exception as e:
            logger.error(f"Error leasing task: {e}", exc_info=True)
            return None

    def _requeue_expired_leases(self):
        """Requeue tasks with expired leases or stuck in_progress."""
        try:
            # Execute queries inside cursor context
            with self.conn.cursor() as cur:
                # Requeue expired leases
                cur.execute("""
                    UPDATE judge_task_queue
                    SET status = 'pending',
                        assigned_server_url = NULL,
                        leased_until = NULL,
                        leased_by_worker = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'leased'
                      AND leased_until < CURRENT_TIMESTAMP
                """)
                count = cur.rowcount
                
                # Also requeue tasks stuck in_progress for more than 2 minutes
                # (normal processing should complete within timeout + buffer)
                cur.execute("""
                    UPDATE judge_task_queue
                    SET status = 'pending',
                        assigned_server_url = NULL,
                        leased_until = NULL,
                        leased_by_worker = NULL,
                        last_error = 'Timed out - stuck in progress',
                        attempts = attempts + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'in_progress'
                      AND updated_at < CURRENT_TIMESTAMP - INTERVAL '2 minutes'
                """)
                stuck_count = cur.rowcount
            
            # With autocommit mode, transaction commits automatically
            if stuck_count > 0:
                logger.warning(f"Requeued {stuck_count} tasks stuck in_progress")
            
            return count + stuck_count
        except Exception as e:
            logger.error(f"Error requeuing expired leases: {e}")
            return 0

    def _process_task(self, task):
        """Process a single task (supports both bulk_judge and newsletter job types)."""
        cur = None
        try:
            # Create a fresh cursor for this task
            cur = self.conn.cursor(row_factory=psycopg.rows.dict_row)

            # Mark task as in progress
            cur.execute("""
                UPDATE judge_task_queue
                SET status = 'in_progress',
                    leased_by_worker = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'leased'
            """, (self.worker_id, task['id']))

            if cur.rowcount == 0:
                logger.warning(f"Failed to mark task {task['id']} as in progress")
                cur.close()
                return

            # Detect job type
            job_type = task.get('job_type', 'bulk_judge')

            if job_type == 'newsletter':
                # Newsletter mode: use cached data from task
                paper = {
                    'id': task['paper_id'],
                    'title': task.get('paper_title', ''),
                    'abstract': task.get('paper_abstract', '')
                }
                research_interests = task.get('research_interests', '')
                profile_id = task.get('profile_id')
                profile_name = f"newsletter/profile_{profile_id}" if profile_id is not None else 'newsletter'

                logger.info(
                    f"Processing newsletter task for paper {task['paper_id']} "
                    f"(profile_id={profile_id}, server={self.server_url})"
                )
            else:
                # Bulk judge mode: load paper and profile from database
                # Get paper
                cur.execute("SELECT * FROM papers WHERE id = %s", (task['paper_id'],))
                paper_row = cur.fetchone()
                paper = dict(paper_row) if paper_row else None
                if not paper:
                    raise ValueError(f"Paper {task['paper_id']} not found")

                # Get profile (with caching)
                profile_id = task['profile_id']
                if profile_id in self.profile_cache:
                    profile = self.profile_cache[profile_id]
                else:
                    cur.execute("SELECT * FROM research_profiles WHERE id = %s", (profile_id,))
                    profile_row = cur.fetchone()
                    profile = dict(profile_row) if profile_row else None
                    if not profile:
                        raise ValueError(f"Profile {profile_id} not found")
                    self.profile_cache[profile_id] = profile

                profile_name = profile['name']

                # Get research interests (with caching)
                if profile_id in self.interests_cache:
                    research_interests = self.interests_cache[profile_id]
                else:
                    cur.execute(
                        "SELECT interest_text FROM profile_research_interests WHERE profile_id = %s",
                        (profile_id,)
                    )
                    interests_rows = cur.fetchall()
                    if not interests_rows:
                        raise ValueError(f"No interests for profile {profile_id}")
                    research_interests = " ".join([row['interest_text'] for row in interests_rows])
                    self.interests_cache[profile_id] = research_interests

            # Close cursor to free resources
            # With autocommit mode, transaction already committed
            cur.close()
            cur = None

            # Perform LLM scoring (this happens outside any transaction)
            logger.info(f"Scoring paper for {profile_name}")
            try:
                # Check for empty/invalid abstract before sending to LLM
                abstract = paper.get('abstract', '')
                if not abstract or len(abstract.strip()) < 20:
                    raise PaperProcessingError(f"Paper has empty or too-short abstract (length: {len(abstract.strip()) if abstract else 0})")
                
                score_data = self._score_paper(paper, research_interests)
            except PaperProcessingError:
                # Re-raise paper processing errors as-is
                raise
            except (TimeoutError, ConnectionError, OSError) as e:
                # Server-level errors - re-raise to trigger circuit breaker logic
                logger.error(f"Server error during LLM scoring: {e}")
                raise
            except Exception as e:
                # Check if error indicates a paper content issue vs server issue
                error_str = str(e).lower()
                is_paper_issue = any(term in error_str for term in [
                    'parse', 'json', 'decode', 'invalid', 'malformed', 'format',
                    'content', 'empty', 'response', 'schema', 'validation'
                ])
                is_server_issue = any(term in error_str for term in [
                    'connection', 'timeout', 'refused', 'unreachable', 'unavailable',
                    'network', 'socket', 'model not', 'no model'
                ])
                
                if is_paper_issue and not is_server_issue:
                    # Paper content caused the error
                    raise PaperProcessingError(f"Paper content caused LLM error: {e}")
                else:
                    # Assume server error, propagate normally
                    logger.error(f"LLM scoring failed: {e}")
                    raise

            # Create a new cursor for saving results
            cur = self.conn.cursor(row_factory=psycopg.rows.dict_row)

            # Save score to paper_profile_scores (used for both bulk judge and newsletter)
            # Both job types now use profile-specific scoring for persistence and aggregation
            cur.execute("""
                INSERT INTO paper_profile_scores (paper_id, profile_id, score, related, rationale, judge_model)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (paper_id, profile_id)
                DO UPDATE SET
                    score = EXCLUDED.score,
                    related = EXCLUDED.related,
                    rationale = EXCLUDED.rationale,
                    judge_model = EXCLUDED.judge_model,
                    date_scored = CURRENT_TIMESTAMP
            """, (
                task['paper_id'],
                profile_id,
                score_data['score'],
                score_data['related'],
                score_data['rationale'],
                self.inference_client.model_name
            ))

            # Mark task as completed
            cur.execute("""
                UPDATE judge_task_queue
                SET status = 'completed',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (task['id'],))

            # Close cursor to free resources
            # With autocommit mode, transaction already committed
            cur.close()
            cur = None

            self.tasks_processed += 1
            logger.info(f"Completed {job_type} task with score {score_data['score']} (total: {self.tasks_processed})")

        except PaperProcessingError as e:
            # Paper-level error (bad abstract, unparseable content, etc.)
            # Do NOT re-raise - this allows the main loop to reset consecutive_failures
            # But DO allow retries since LLM responses can be flaky
            logger.warning(f"Task {task['id']} failed due to paper content issue: {e}")
            
            try:
                if cur:
                    cur.close()
                with self.conn.cursor() as err_cur:
                    # Get current attempts count
                    err_cur.execute("SELECT attempts FROM judge_task_queue WHERE id = %s", (task['id'],))
                    result = err_cur.fetchone()
                    current_attempts = result[0] if result else 0
                    new_attempts = current_attempts + 1
                    
                    # Use max_retries from config (default 3) - allow retries for LLM flakiness
                    max_retries = getattr(self, 'max_retries', 3)
                    
                    if new_attempts < max_retries:
                        # Requeue for retry - might work on another attempt (LLM flakiness)
                        err_cur.execute("""
                            UPDATE judge_task_queue
                            SET status = 'pending',
                                last_error = %s,
                                attempts = %s,
                                assigned_server_url = NULL,
                                leased_until = NULL,
                                leased_by_worker = NULL,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (f"Paper content issue (retrying): {e}", new_attempts, task['id']))
                        logger.info(f"Task {task['id']} requeued for retry (attempt {new_attempts}/{max_retries}) - paper issue, may be LLM flakiness")
                    else:
                        # Max retries exceeded - mark as permanently failed
                        err_cur.execute("""
                            UPDATE judge_task_queue
                            SET status = 'failed',
                                last_error = %s,
                                attempts = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (f"Paper content error after {new_attempts} attempts: {e}", new_attempts, task['id']))
                        logger.warning(f"Task {task['id']} permanently failed after {new_attempts} attempts (paper content issue)")
            except Exception as retry_err:
                logger.error(f"Failed to update task status: {retry_err}")
            # Return normally - main loop will reset consecutive_failures (server is healthy)
                
        except Exception as e:
            # All other exceptions - mark task as failed and RE-RAISE to main loop
            # This allows the main loop's circuit breaker to track failures properly
            logger.error(f"Task {task['id']} failed: {e}")

            # Mark task as failed with retry logic before re-raising
            try:
                if cur:
                    cur.close()
                with self.conn.cursor() as err_cur:
                    # Get current attempts count
                    err_cur.execute("SELECT attempts FROM judge_task_queue WHERE id = %s", (task['id'],))
                    result = err_cur.fetchone()
                    current_attempts = result[0] if result else 0
                    new_attempts = current_attempts + 1
                    
                    # Use max_retries from config (default 3)
                    max_retries = getattr(self, 'max_retries', 3)
                    
                    if new_attempts < max_retries:
                        # Requeue for retry
                        err_cur.execute("""
                            UPDATE judge_task_queue
                            SET status = 'pending',
                                last_error = %s,
                                attempts = %s,
                                assigned_server_url = NULL,
                                leased_until = NULL,
                                leased_by_worker = NULL,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (str(e), new_attempts, task['id']))
                        logger.info(f"Task {task['id']} requeued for retry (attempt {new_attempts}/{max_retries})")
                    else:
                        # Max retries exceeded - mark as permanently failed
                        err_cur.execute("""
                            UPDATE judge_task_queue
                            SET status = 'failed',
                                last_error = %s,
                                attempts = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (str(e), new_attempts, task['id']))
                        logger.warning(f"Task {task['id']} permanently failed after {new_attempts} attempts")
            except Exception as retry_err:
                logger.error(f"Failed to update task status: {retry_err}")
            
            # Re-raise so main loop's circuit breaker can track this failure
            raise
        finally:
            # Ensure cursor is closed
            if cur:
                try:
                    cur.close()
                except:
                    pass

    def _score_paper(self, paper: dict, research_interests: str) -> dict:
        """Score a paper using the LLM with timeout enforcement."""
        from ..prompt import RESEARCH_INTERESTS_SYSTEM_PROMPT, research_prompt
        from ..prompt.data_models import ResearchInterestsPromptData
        
        messages = [
            {"role": "user", "content": research_prompt(research_interests, paper['abstract'])}
        ]
        
        # Retry configuration for transient LM Studio errors (model unloaded, etc.)
        max_llm_retries = 3
        retry_delay = 5  # seconds
        
        for llm_attempt in range(max_llm_retries):
            # Use a timeout wrapper to prevent hanging
            result = [None]
            exception = [None]
            
            def _invoke_with_timeout():
                try:
                    result[0] = self.inference_client.invoke(
                        messages=messages,
                        system_prompt=RESEARCH_INTERESTS_SYSTEM_PROMPT,
                        schema=ResearchInterestsPromptData
                    )
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=_invoke_with_timeout, daemon=True)
            thread.start()
            thread.join(timeout=self.timeout + 5)  # Timeout + 5 second buffer
            
            if thread.is_alive():
                # Thread is still running - LLM call hung
                raise TimeoutError(f"LLM inference exceeded {self.timeout + 5}s timeout")
            
            if exception[0]:
                error_str = str(exception[0]).lower()
                # Check if this is a retryable LM Studio error (model unloaded)
                is_model_unloaded = any(term in error_str for term in [
                    'no model found', 'nomodelmatchingquery', 'totalloadedmodels', 
                    'model not loaded', 'model unavailable'
                ])
                
                if is_model_unloaded and llm_attempt < max_llm_retries - 1:
                    logger.warning(
                        f"LM Studio model appears unloaded (attempt {llm_attempt + 1}/{max_llm_retries}). "
                        f"Waiting {retry_delay}s for potential auto-reload..."
                    )
                    # Try to reinitialize the client to trigger model reload
                    self._reinitialize_lmstudio_client()
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise exception[0]
            
            if result[0] is None:
                raise RuntimeError("LLM inference failed without exception")
            
            # Success - break out of retry loop
            response = result[0]
            break
        else:
            # All retries exhausted
            raise RuntimeError(f"LLM inference failed after {max_llm_retries} attempts")
        
        # Parse response - handle various formats robustly
        response_json = self._parse_llm_response(response)
        
        # Defensive check - ensure we have a dict
        if not isinstance(response_json, dict):
            logger.warning(f"_parse_llm_response returned non-dict: {type(response_json)}, using defaults")
            response_json = {'score': 5, 'related': False, 'rationale': str(response_json)[:300]}
        
        score = response_json.get('score', 5)
        if not isinstance(score, (int, float)):
            # Try to extract numeric value from string
            try:
                score = int(str(score).strip())
            except (ValueError, TypeError):
                score = 5
        score = max(1, min(10, int(score)))
        
        return {
            'score': score,
            'related': bool(response_json.get('related', False)),
            'rationale': str(response_json.get('rationale', ''))
        }

    def _parse_llm_response(self, response) -> dict:
        """Parse LLM response into a dictionary, handling various formats robustly."""
        import re
        import json_repair
        
        # If response is already a dict, return it
        if isinstance(response, dict):
            return response
        
        # If response is not a string, convert it
        if not isinstance(response, str):
            try:
                # Could be a Pydantic model or similar
                if hasattr(response, 'model_dump'):
                    return response.model_dump()
                elif hasattr(response, 'dict'):
                    return response.dict()
                elif hasattr(response, '__dict__'):
                    return response.__dict__
            except Exception:
                pass
            response = str(response)
        
        response = response.strip()
        
        # Try json_repair first
        try:
            parsed = json_repair.loads(response)
            if isinstance(parsed, dict):
                return parsed
            # If json_repair returned a string, continue to fallback methods
        except Exception:
            pass
        
        # Try standard JSON parsing
        try:
            import json
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        json_match = re.search(json_block_pattern, response)
        if json_match:
            try:
                parsed = json_repair.loads(json_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        
        # Try to find a JSON object in the response
        json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_obj_pattern, response)
        for match in json_matches:
            try:
                parsed = json_repair.loads(match)
                if isinstance(parsed, dict) and ('score' in parsed or 'related' in parsed):
                    return parsed
            except Exception:
                continue
        
        # Fallback: try to extract values using regex
        extracted = {}
        
        # Extract score (look for patterns like "score": 7, score: 7, Score: 7, etc.)
        score_patterns = [
            r'"?score"?\s*[:=]\s*(\d+)',
            r'(?:Score|SCORE|rating|Rating)\s*[:=]?\s*(\d+)',
            r'(\d+)\s*(?:/\s*10|out of 10)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    extracted['score'] = int(match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract related (look for true/false patterns)
        related_patterns = [
            r'"?related"?\s*[:=]\s*(true|false)',
            r'(?:is\s+)?related\s*[:=]?\s*(yes|no|true|false)',
            r'(?:relevant|relevance)\s*[:=]?\s*(yes|no|true|false|high|low)',
        ]
        for pattern in related_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                extracted['related'] = value in ('true', 'yes', 'high')
                break
        
        # Extract rationale (take a reasonable chunk of text)
        rationale_patterns = [
            r'"?rationale"?\s*[:=]\s*"([^"]+)"',
            r'"?rationale"?\s*[:=]\s*\'([^\']+)\'',
            r'(?:rationale|reason|explanation)\s*[:=]?\s*(.+?)(?:\n|$)',
        ]
        for pattern in rationale_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted['rationale'] = match.group(1).strip()[:500]  # Limit length
                break
        
        # If we couldn't extract a rationale, use part of the response
        if 'rationale' not in extracted and response:
            # Clean up and truncate
            clean_response = re.sub(r'\s+', ' ', response).strip()
            extracted['rationale'] = clean_response[:300] if clean_response else 'No rationale provided'
        
        # Set defaults for missing values
        if 'score' not in extracted:
            extracted['score'] = 5  # Default middle score
        if 'related' not in extracted:
            # Infer from score
            extracted['related'] = extracted.get('score', 5) >= 6
        if 'rationale' not in extracted:
            extracted['rationale'] = 'Unable to parse structured response'
        
        logger.debug(f"Fallback parsing extracted: {extracted}")
        return extracted

    def _send_heartbeat(self):
        """Send worker heartbeat."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO worker_heartbeats (worker_id, server_url, job_id, status, tasks_processed, last_heartbeat)
                    VALUES (%s, %s, %s, 'active', %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (worker_id, server_url, job_id)
                    DO UPDATE SET
                        status = 'active',
                        tasks_processed = EXCLUDED.tasks_processed,
                        last_heartbeat = CURRENT_TIMESTAMP
                """, (
                    self.worker_id,
                    self.server_url,
                    str(self.job_id) if self.job_id else None,
                    self.tasks_processed
                ))
            # With autocommit mode, transaction commits automatically
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")

    def _cleanup(self):
        """Clean up on shutdown."""
        logger.info(f"Worker {self.worker_id} shutting down (processed {self.tasks_processed} tasks)")
        
        # Mark worker as inactive
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE worker_heartbeats
                    SET status = 'inactive',
                        last_heartbeat = CURRENT_TIMESTAMP
                    WHERE worker_id = %s
                      AND server_url = %s
                      AND (job_id = %s OR (%s IS NULL AND job_id IS NULL))
                """, (
                    self.worker_id,
                    self.server_url,
                    str(self.job_id) if self.job_id else None,
                    str(self.job_id) if self.job_id else None
                ))
            # With autocommit mode, transaction commits automatically
        except Exception as e:
            logger.warning(f"Failed to mark worker as inactive: {e}")
        
        # Close inference client connections
        try:
            if hasattr(self.inference_client, 'close'):
                self.inference_client.close()
        except:
            pass
        
        # Close database connection
        try:
            self.conn.close()
        except:
            pass


def run_worker(
    server_url: str,
    job_id: Optional[UUID],
    max_retries: int,
    timeout: int,
    provider: str = "ollama",
    config_json: Optional[dict] = None,
    judge_model_config: Optional[dict] = None,
    server_model_name: Optional[str] = None,
    server_model_config: Optional[dict] = None
):
    """Run a single worker for a server."""
    worker = JudgeWorker(
        server_url=server_url,
        provider=provider,
        config_json=config_json,
        job_id=job_id,
        max_retries=max_retries,
        timeout=timeout,
        judge_model_config=judge_model_config,
        server_model_name=server_model_name,
        server_model_config=server_model_config
    )
    
    try:
        worker.start()
    except Exception as e:
        logger.error(f"Worker for {server_url} failed: {e}", exc_info=True)


def main():
    """Main entry point for the worker launcher."""
    parser = argparse.ArgumentParser(description='Theseus Judge Worker')
    parser.add_argument('--server-url', help='Ollama server URL to process')
    parser.add_argument('--job-id', help='Specific job ID to process')
    parser.add_argument('--provider', default='ollama', help='LLM provider type (ollama, lmstudio, openai, etc.)')
    parser.add_argument('--all-enabled', action='store_true', help='Process all enabled servers')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries per task')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--server-model-name', help='Override model name for this server')
    parser.add_argument('--server-model-config', help='Override model config (JSON string) for this server')

    args = parser.parse_args()

    # Parse server model config if provided
    server_model_config = None
    if args.server_model_config:
        try:
            server_model_config = json.loads(args.server_model_config)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for --server-model-config: {e}")
            sys.exit(1)

    # For --all-enabled, we need to get the list of servers
    # Import here to avoid early module loading
    if args.all_enabled:
        from ..data_access.inference_servers import InferenceServersRepository
        servers = InferenceServersRepository.get_enabled()
        # Include per-server model configuration from database
        servers_to_process = [
            (server.url, server.provider, server.config_json, args.job_id, server.model_name, server.model_config)
            for server in servers
        ]
        logger.info(f"Processing all {len(servers_to_process)} enabled servers")
    elif args.server_url:
        # Single server mode - use CLI arguments for model overrides
        servers_to_process = [
            (args.server_url, args.provider, {}, args.job_id, args.server_model_name, server_model_config)
        ]
        logger.info(f"Processing single server: {args.server_url} (provider: {args.provider})")
    else:
        logger.error("Must specify either --server-url or --all-enabled")
        sys.exit(1)

    # For multiple servers, fork processes
    if len(servers_to_process) > 1:
        import multiprocessing
        processes = []

        for server_url, provider, config_json, job_id, model_name, model_config in servers_to_process:
            job_uuid = UUID(job_id) if job_id else None
            p = multiprocessing.Process(
                target=run_worker,
                args=(
                    server_url,
                    job_uuid,
                    args.max_retries,
                    args.timeout,
                    provider,
                    config_json,
                    None,  # judge_model_config - load from settings
                    model_name,  # server_model_name - from database or CLI
                    model_config  # server_model_config - from database or CLI
                )
            )
            p.start()
            processes.append(p)
            if model_name:
                logger.info(f"Started {provider} worker process for {server_url} with custom model: {model_name}")
            else:
                logger.info(f"Started {provider} worker process for {server_url}")
        
        # Wait for all processes
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, terminating workers...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join(timeout=5)
    else:
        # Single server - run directly
        server_url, provider, config_json, job_id, model_name, model_config = servers_to_process[0]
        job_uuid = UUID(job_id) if job_id else None
        run_worker(
            server_url,
            job_uuid,
            args.max_retries,
            args.timeout,
            provider,
            config_json,
            None,  # judge_model_config - load from settings
            model_name,  # server_model_name - from database or CLI
            model_config  # server_model_config - from database or CLI
        )


if __name__ == "__main__":
    main()

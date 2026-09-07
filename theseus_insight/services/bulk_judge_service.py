from ..workers.ownership import worker_environment
"""Bulk-judge job orchestration: submission, multi-server worker launch,
monitoring, signals, and conflict checks (extracted from the
bulk_operations router in refactor B7).

_start_bulk_judge_operation keeps its FastAPI-facing signature
(BackgroundTasks, HTTPException) because papers.py reuses it as the
shared implementation of two endpoints.
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, HTTPException

from ..data_processing.checkpoint_manager import CheckpointManager
from ..data_processing.queue_producer import JudgeQueueProducer
from ..data_access.bulk_judge import BulkJudgeRunner
from ..data_access.inference_servers import InferenceServersRepository
from ..db import get_connection_pool
from ..api.models import BulkJudgeRequest, JobStartResponse
from .harvest_service import (
    _download_arxiv_for_range, _ensure_embeddings_for_range,
)
from .scheduling import (
    _restore_scheduled_tasks, _restore_scheduled_tasks_on_cancel,
    _suspend_scheduled_tasks,
)

logger = logging.getLogger(__name__)

async def run_bulk_judge_task(
    job_id: UUID,
    request: BulkJudgeRequest,
    checkpoint_manager: CheckpointManager
):
    """Run bulk judge operation in background."""
    print(f"🎯 Background task started: run_bulk_judge_task for job {job_id}")
    try:
        # Get orchestration config for judge model
        from ..data_access import SettingsRepository
        import json
        
        orch_json = SettingsRepository.get("orchestration")
        if not orch_json:
            await checkpoint_manager.fail_job(job_id, "Orchestration configuration not found")
            raise ValueError("Orchestration configuration not found")
        
        orch_config = json.loads(orch_json)
        judge_config = orch_config.get('judge_model', {})
        
        if not judge_config:
            await checkpoint_manager.fail_job(job_id, "Judge model configuration not found in orchestration config")
            raise ValueError("Judge model configuration not found")
        
        # Initialize runner with proper judge model config
        runner = BulkJudgeRunner(
            judge_model_config=judge_config,
            verbose=True,
            use_optimized_scorer=True,
            checkpoint_manager=checkpoint_manager
        )
        
        await runner.run_bulk_judge(
            profile_ids=request.profile_ids,
            all_profiles=request.all_profiles,
            limit=request.limit,
            start_date=request.start_date,
            end_date=request.end_date,
            batch_size=request.batch_size
        )
    except Exception as e:
        await checkpoint_manager.fail_job(job_id, str(e))
        raise

async def run_multi_server_bulk_judge_task(
    job_id: UUID,
    request: BulkJudgeRequest,
    checkpoint_manager: CheckpointManager,
    selected_servers: List
):
    """Run multi-server bulk judge operation using queue-based processing."""
    print(f"🎯 Background task started: run_multi_server_bulk_judge_task for job {job_id}")
    logger.info(f"🎯 Background task started: run_multi_server_bulk_judge_task for job {job_id}")
    
    # Force flush to ensure logs are written
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        pool = await get_connection_pool()

        # Create queue producer and enqueue tasks
        queue_producer = JudgeQueueProducer()
        
        # Convert profile_ids from strings to integers
        profile_ids_int = [int(pid) for pid in request.profile_ids] if request.profile_ids else None
        
        result = queue_producer.enqueue_bulk_judge_job(
            job_id=job_id,  # Pass as UUID, not string
            profile_ids=profile_ids_int,
            date_from=request.start_date,
            date_to=request.end_date,
            overwrite_existing=request.overwrite_existing
        )

        print(f"📊 Queue producer result: {result}")

        # Launch worker processes for each selected server
        await _launch_worker_processes(job_id, selected_servers, request.request_timeout_sec, request.max_retries)

        # Monitor job progress until completion
        await _monitor_multi_server_job(job_id, checkpoint_manager)

    except Exception as e:
        await checkpoint_manager.fail_job(job_id, str(e))
        # Restore suspended tasks on failure
        try:
            await _restore_scheduled_tasks(request.suspend_scheduled_tasks, checkpoint_manager, job_id)
        except Exception as restore_error:
            print(f"Warning: Failed to restore scheduled tasks: {restore_error}")
        raise
    finally:
        # Restore suspended tasks on completion
        try:
            await _restore_scheduled_tasks(request.suspend_scheduled_tasks, checkpoint_manager, job_id)
        except Exception as restore_error:
            print(f"Warning: Failed to restore scheduled tasks: {restore_error}")

async def _launch_single_worker(
    job_id: UUID,
    server_url: str,
    request_timeout_sec: int,
    max_retries: int,
    model_name: Optional[str] = None,
    model_config: Optional[dict] = None,
    provider: str = "ollama"
):
    """Launch a single worker process for a specific server."""
    import subprocess
    import sys
    import os
    import json

    print(f"🚀 Launching single worker for job {job_id} on {server_url} (provider: {provider})")

    # Get the current conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')

    try:
        # Build the command to run the worker
        cmd = [
            sys.executable, '-m', 'theseus_insight.workers.judge_worker',
            '--job-id', str(job_id),
            '--server-url', server_url,
            '--provider', provider,  # Use the server's configured provider (ollama/lmstudio)
            '--timeout', str(request_timeout_sec),
            '--max-retries', str(max_retries)
        ]

        # Add per-server model overrides if specified
        if model_name:
            cmd.extend(['--server-model-name', model_name])
            print(f"🔧 Using per-server model name: {model_name}")

        if model_config:
            cmd.extend(['--server-model-config', json.dumps(model_config)])
            print(f"🔧 Using per-server model config: {model_config}")
        
        # Launch the process in the background; send output to DEVNULL to avoid pipe backpressure
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
            env=worker_environment(),
            start_new_session=True  # Detach from parent process
        )
        
        print(f"✅ Single worker launched for {server_url} (PID: {process.pid})")
        return process.pid
        
    except Exception as e:
        print(f"❌ Failed to launch single worker for {server_url}: {e}")
        raise

async def _launch_worker_processes(job_id: UUID, selected_servers, request_timeout_sec: Optional[int], max_retries: Optional[int]):
    """Launch worker processes for each selected server."""
    import subprocess
    import os
    import sys

    print(f"🚀 Launching worker processes for job {job_id} with {len(selected_servers)} servers")

    try:
        # Get global defaults if not provided
        if request_timeout_sec is None or max_retries is None:
            from ..data_access.settings import SettingsRepository
            if request_timeout_sec is None:
                request_timeout_sec = SettingsRepository.get_int("ollama_request_timeout_sec", 30)
            if max_retries is None:
                max_retries = SettingsRepository.get_int("ollama_max_retries", 3)

        print(f"⚙️ Worker configuration: timeout={request_timeout_sec}, max_retries={max_retries}")
        print(f"🚀 About to launch {len(selected_servers)} worker processes")

        # Launch a worker process for each server
        successful_launches = 0
        failed_launches = []
        
        for i, server in enumerate(selected_servers):
            try:
                print(f"\n{'='*60}")
                print(f"🔧 [{i+1}/{len(selected_servers)}] Preparing to launch worker for server: {server.name} ({server.url}) - ID: {server.id}")

                # Launch worker in background
                cmd = [
                    sys.executable, "-m", "theseus_insight.workers.judge_worker",
                    "--job-id", str(job_id),
                    "--server-url", server.url,
                    "--provider", server.provider,  # Use the server's configured provider (ollama/lmstudio)
                    "--timeout", str(request_timeout_sec),
                    "--max-retries", str(max_retries)
                ]

                # Add per-server model overrides if specified
                if server.model_name:
                    cmd.extend(["--server-model-name", server.model_name])
                    print(f"🔧 Using per-server model name: {server.model_name}")

                if server.model_config:
                    import json
                    cmd.extend(["--server-model-config", json.dumps(server.model_config)])
                    print(f"🔧 Using per-server model config: {server.model_config}")

                print(f"📋 Worker command: {' '.join(cmd)}")

                # Launch as background process
                print(f"🚀 Launching worker process for server {server.name}")
                print(f"📋 Command: {' '.join(cmd)}")

                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        cwd=os.getcwd(),
                        env=worker_environment()
                    )

                    print(f"✅ Launched worker process for server {server.name} (PID: {process.pid})")

                    # Wait a moment for the process to start and check for immediate errors
                    import time
                    time.sleep(1.0)  # Give it more time to start

                    if process.poll() is None:
                        print(f"🟢 Worker process {process.pid} is running successfully for {server.name}")
                        successful_launches += 1
                        # Let the process continue running in background
                        # Don't wait for it or communicate with it
                    else:
                        # Process failed to start
                        stdout, stderr = process.communicate(timeout=5)
                        print(f"❌ Worker process {process.pid} failed to start for {server.name} (exit code: {process.returncode})")
                        print(f"STDOUT: {stdout}")
                        print(f"STDERR: {stderr}")
                        failed_launches.append(server.name)

                except subprocess.TimeoutExpired:
                    print(f"⚠️ Worker process {process.pid} timed out during startup")
                except Exception as e:
                    print(f"❌ Failed to create worker process for server {server.name}: {e}")
                    import traceback
                    traceback.print_exc()

            except Exception as e:
                print(f"❌ Failed to launch worker for server {server.name}: {e}")
                failed_launches.append(server.name)
                import traceback
                traceback.print_exc()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Worker Launch Summary:")
        print(f"  ✅ Successful launches: {successful_launches}/{len(selected_servers)}")
        if failed_launches:
            print(f"  ❌ Failed launches: {', '.join(failed_launches)}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"❌ Error launching worker processes: {e}")
        import traceback
        traceback.print_exc()

async def _monitor_multi_server_job(job_id: UUID, checkpoint_manager: CheckpointManager):
    """Monitor multi-server job progress until completion."""
    import asyncio
    from ..data_access.judge_task_queue import JudgeTaskQueueRepository
    from ..data_access.worker_heartbeats import WorkerHeartbeatsRepository

    logger.info(f"🎯 Monitor started for job {job_id}")

    try:
        queue_repo = JudgeTaskQueueRepository()
        heartbeat_repo = WorkerHeartbeatsRepository()

        iteration = 0
        while True:
            iteration += 1
            # Get job progress
            progress = queue_repo.get_job_progress(job_id)
            
            logger.debug(f"Monitor iteration {iteration} for job {job_id}: progress={progress}")

            if progress:
                # Update job progress
                await checkpoint_manager.update_progress(
                    job_id,
                    progress['completed_tasks'],
                    progress['total_tasks']
                )

                # Check if job is complete
                completed = progress.get('completed_tasks', 0)
                failed = progress.get('failed_tasks', 0)
                total = progress.get('total_tasks', 0)
                pending = progress.get('pending_tasks', 0)
                in_prog = progress.get('in_progress_tasks', 0)
                leased = progress.get('leased_tasks', 0)
                
                logger.info(f"📊 Job {job_id} progress: {completed}/{total} completed, {failed} failed, {pending} pending, {in_prog} in-progress")
                
                # Job is complete when all tasks are in terminal states (completed, failed, or canceled)
                # OR when completed tasks equal total (for backwards compatibility)
                all_processed = (completed + failed >= total) and pending == 0 and in_prog == 0 and leased == 0
                
                if (completed >= total or all_processed) and total > 0:
                    # Mark job as complete and terminate workers
                    logger.info(f"✅ Job {job_id} COMPLETE: {completed}/{total} tasks finished")
                    if progress['failed_tasks'] > 0:
                        # Mark job as completed with failures
                        await checkpoint_manager.complete_job(
                            job_id,
                            f"Completed with {progress['failed_tasks']} failed tasks out of {progress['total_tasks']} total tasks"
                        )
                    else:
                        # Mark job as fully successful
                        await checkpoint_manager.complete_job(
                            job_id,
                            f"Successfully completed {progress['total_tasks']} tasks"
                        )
                    
                    # Terminate all worker processes for this job
                    logger.info(f"🛑 Terminating worker processes for completed job {job_id}")
                    await _signal_workers_cancel(job_id)
                    logger.info(f"🎉 Job {job_id} monitoring complete - exiting monitor loop")
                    break

            # Check for active workers - fallback completion detection
            active_workers = heartbeat_repo.get_active_workers(job_id)
            logger.debug(f"Active workers for job {job_id}: {len(active_workers) if active_workers else 0}")
            
            if not active_workers and progress and progress['pending_tasks'] == 0 and progress['in_progress_tasks'] == 0:
                # No active workers and no pending/in-progress tasks - mark as complete
                logger.info(f"✅ Job {job_id} detected complete via worker shutdown (no active workers, no pending tasks)")
                if progress['failed_tasks'] > 0:
                    await checkpoint_manager.complete_job(
                        job_id,
                        f"Completed with {progress['failed_tasks']} failed tasks out of {progress['total_tasks']} total tasks"
                    )
                else:
                    await checkpoint_manager.complete_job(
                        job_id,
                        f"Successfully completed {progress['completed_tasks']} tasks"
                    )
                
                # Ensure all worker processes are terminated
                logger.info(f"🛑 Terminating any remaining worker processes for job {job_id}")
                await _signal_workers_cancel(job_id)
                logger.info(f"🎉 Job {job_id} monitoring complete - exiting monitor loop")
                break

            # Wait before next check (5 seconds for faster completion detection)
            await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"❌ Monitor for job {job_id} crashed: {e}", exc_info=True)
        try:
            await checkpoint_manager.fail_job(job_id, f"Monitoring failed: {str(e)}")
        except Exception as fail_error:
            logger.error(f"Failed to mark job as failed: {fail_error}")
        raise

async def _start_bulk_judge_operation(
    request: BulkJudgeRequest,
    background_tasks: BackgroundTasks
) -> JobStartResponse:
    """Core bulk judge operation logic that can be called from multiple endpoints."""
    if not request.all_profiles and not request.profile_ids:
        raise HTTPException(
            status_code=400,
            detail="Either all_profiles must be true or profile_ids must be provided"
        )

    pool = await get_connection_pool()
    checkpoint_manager = CheckpointManager(pool)

    # Check for job conflicts before starting
    conflict_data = await _check_job_conflicts()
    if conflict_data["has_conflicts"]:
        # Smart conflict detection:
        # - Allow multi-server bulk judge jobs to run concurrently (different Ollama servers)
        # - Block if single-server bulk judge is running (same resources)
        # - Block if other resource-intensive jobs are running (newsletters, mindmaps, podcasts)
        
        # Check for non-bulk-judge resource-intensive jobs
        non_bulk_judge_conflicts = [
            job for job in conflict_data["conflicts"]
            if job["job_type"] in ['harvest_judge', 'newsletter_generation', 'mindmap_generation', 'podcast_generation']
        ]
        
        # Check for single-server bulk judge jobs
        single_server_bulk_judge = [
            job for job in conflict_data["conflicts"]
            if job["job_type"] == 'bulk_judge' and not job.get("multi_server", False)
        ]
        
        # Block if there are any non-bulk-judge resource-intensive jobs
        if non_bulk_judge_conflicts:
            conflict_details = "; ".join([f"{job['job_type']} ({job['description']})" for job in non_bulk_judge_conflicts])
            raise HTTPException(
                status_code=409,
                detail=f"Resource conflict with running jobs: {conflict_details}. " +
                       f"Recommendation: {conflict_data['recommendations'][0] if conflict_data['recommendations'] else 'Wait for jobs to complete or cancel them.'}"
            )
        
        # Block if there's a single-server bulk judge running and we're NOT using multi-server mode
        if single_server_bulk_judge and not request.use_multi_server:
            conflict_details = "; ".join([f"{job['job_type']} ({job['description']})" for job in single_server_bulk_judge])
            raise HTTPException(
                status_code=409,
                detail=f"A single-server bulk judge job is running: {conflict_details}. " +
                       f"Either wait for it to complete, or enable multi-server mode to process in parallel."
            )
        
        # Check for multi-server bulk judge jobs with overlapping servers
        if request.use_multi_server:
            multi_server_bulk_judge = [
                job for job in conflict_data["conflicts"]
                if job["job_type"] == 'bulk_judge' and job.get("multi_server", False)
            ]
            
            if multi_server_bulk_judge:
                # Check for server overlap
                requested_servers = set(request.server_ids) if request.server_ids else None
                
                for running_job in multi_server_bulk_judge:
                    running_servers = running_job.get("servers", [])
                    if not running_servers:
                        # If running job has no specific servers, it's using all available servers
                        # This creates a potential conflict
                        raise HTTPException(
                            status_code=409,
                            detail=f"A multi-server bulk judge job is already using all available Ollama servers. "
                                   f"Job: {running_job['job_type']} ({running_job['description']}). "
                                   f"Recommendation: Wait for it to complete, or ensure it has specific servers assigned."
                        )
                    
                    running_server_ids = set(running_servers) if isinstance(running_servers, list) else set()
                    
                    # If we didn't specify servers, we want to use all available servers
                    if requested_servers is None:
                        # Check if there's any multi-server job running at all
                        raise HTTPException(
                            status_code=409,
                            detail=f"Cannot start multi-server job using all servers while another multi-server job is running. "
                                   f"Running job: {running_job['job_type']} ({running_job['description']}). "
                                   f"Recommendation: Specify specific server_ids that don't overlap, or wait for the job to complete."
                        )
                    
                    # Check for overlap between requested and running servers
                    overlap = requested_servers & running_server_ids
                    if overlap:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Server conflict: The following Ollama servers are already in use by another bulk judge job: {sorted(overlap)}. "
                                   f"Running job: {running_job['job_type']} ({running_job['description']}). "
                                   f"Recommendation: Use different server_ids that don't overlap, or wait for the job to complete."
                        )

    # Validate multi-server configuration if enabled
    selected_servers = []
    if request.use_multi_server:
        print(f"🔍 Multi-server mode enabled, requested server_ids: {request.server_ids}")
        server_repo = InferenceServersRepository()
        available_servers = server_repo.get_enabled()
        print(f"📋 Found {len(available_servers)} available servers: {[(s.id, s.name, s.url) for s in available_servers]}")

        if not available_servers:
            raise HTTPException(
                status_code=400,
                detail="Multi-server mode requested but no enabled Ollama servers found. Please configure and enable servers in Settings."
            )

        if request.server_ids:
            # Validate specific server IDs
            server_id_set = set(request.server_ids)
            available_ids = {server.id for server in available_servers}
            invalid_ids = server_id_set - available_ids

            if invalid_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid server IDs: {list(invalid_ids)}. Available server IDs: {list(available_ids)}"
                )

            selected_servers = [server for server in available_servers if server.id in request.server_ids]
            print(f"🎯 Selected {len(selected_servers)} servers by ID filter: {[(s.id, s.name, s.url) for s in selected_servers]}")
        else:
            # Use all enabled servers
            selected_servers = available_servers
            print(f"🎯 Using all {len(selected_servers)} enabled servers: {[(s.id, s.name, s.url) for s in selected_servers]}")

        if not selected_servers:
            raise HTTPException(
                status_code=400,
                detail="No valid servers available for multi-server processing"
            )
        
        # Validate that judge model is Ollama-compatible (multi-server supports Ollama and LMStudio)
        from ..data_access import SettingsRepository
        import json
        
        orch_json = SettingsRepository.get("orchestration")
        if orch_json:
            orch_config = json.loads(orch_json)
            judge_config = orch_config.get('judge_model', {})
            model_type = judge_config.get('model_type', '').lower()
            
            # LMStudio uses the same OpenAI-compatible API as Ollama
            # Also allow empty model_type if servers are already configured (backwards compatibility)
            valid_model_types = {'ollama', 'lmstudio', ''}
            if model_type not in valid_model_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Multi-server mode only supports Ollama/LMStudio models. Current judge model type: {model_type}. "
                           f"Please switch to an Ollama or LMStudio model in Settings → Orchestration Config or disable multi-server mode."
                )
            
            effective_type = model_type if model_type else 'ollama-compatible'
            print(f"✅ Validated judge model is {effective_type}: {judge_config.get('model_name', 'unknown')}")

    # Suspend scheduled tasks if requested
    suspended_tasks_snapshot = []
    if request.suspend_scheduled_tasks:
        try:
            suspended_tasks_snapshot = await _suspend_scheduled_tasks()
        except Exception as e:
            # Log but don't fail - scheduler suspension is not critical
            print(f"Warning: Failed to suspend scheduled tasks: {e}")

    # Create job record with enhanced configuration
    configuration = {
        "profile_ids": request.profile_ids,
        "all_profiles": request.all_profiles,
        "limit": request.limit,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "batch_size": request.batch_size,
        "use_multi_server": request.use_multi_server,
        "server_ids": request.server_ids,
        "request_timeout_sec": request.request_timeout_sec,
        "max_retries": request.max_retries,
        "suspend_scheduled_tasks": request.suspend_scheduled_tasks,
        "overwrite_existing": request.overwrite_existing,
        "suspended_tasks_snapshot": suspended_tasks_snapshot,
        "selected_servers": [{"id": s.id, "name": s.name, "url": s.url} for s in selected_servers] if selected_servers else []
    }

    # Create the job and get the assigned job ID
    created_job_id = await checkpoint_manager.create_job(
        job_type="bulk_judge",
        configuration=configuration
    )

    # Start appropriate background task
    if request.use_multi_server:
        # Convert profile IDs to integers for use throughout
        profile_ids_int = [int(pid) for pid in request.profile_ids] if request.profile_ids else None
        
        # Run arXiv download + embedding + judging in background task (async, non-blocking)
        # This allows the response to return immediately and UI to stay responsive
        async def run_download_embedding_and_judging():
            """Background task: download papers, embed them, then launch judge workers"""
            # Step 1: Download papers from arXiv
            try:
                dl_stats = _download_arxiv_for_range(
                    request.start_date, 
                    request.end_date,
                    overwrite_existing=request.overwrite_existing,
                    profile_ids=profile_ids_int,
                    use_profile_arxiv_filters=request.use_profile_arxiv_filters
                )
                logger.info(f"✅ ArXiv download complete: {dl_stats}")
            except Exception as dl_e:
                logger.error(f"❌ ArXiv download failed: {dl_e}")
                await checkpoint_manager.fail_job(created_job_id, f"ArXiv download failed: {str(dl_e)}")
                return

            # Step 2: Ensure embeddings for all papers in range
            try:
                pre_embedded = await _ensure_embeddings_for_range(request.start_date, request.end_date)
                logger.info(f"✅ Embeddings ensured for {pre_embedded} papers")
            except Exception as pre_e:
                logger.error(f"❌ Embedding stage failed: {pre_e}")
                await checkpoint_manager.fail_job(created_job_id, f"Embedding failed: {str(pre_e)}")
                return

            try:
                # Enqueue judge tasks
                queue_producer = JudgeQueueProducer()
                result = queue_producer.enqueue_bulk_judge_job(
                    job_id=created_job_id,
                    profile_ids=profile_ids_int,
                    date_from=request.start_date,
                    date_to=request.end_date,
                    overwrite_existing=request.overwrite_existing,
                    create_processing_job=False,
                    server_urls=None  # dynamic pooling
                )
                
                logger.info(f"📊 Queue producer result: {result}")
                
                # Launch worker processes
                await _launch_worker_processes(created_job_id, selected_servers, request.request_timeout_sec, request.max_retries)
                
                # Monitor the job
                await _monitor_multi_server_job(created_job_id, checkpoint_manager)
                
            except Exception as e:
                logger.error(f"Error in judging phase: {e}")
                await checkpoint_manager.fail_job(created_job_id, f"Judging failed: {str(e)}")
        
        # Add the entire workflow (download → embed → judge) as a background task
        background_tasks.add_task(run_download_embedding_and_judging)
    else:
        # Use traditional single-server processing
        background_tasks.add_task(
            run_bulk_judge_task,
            created_job_id,
            request,
            checkpoint_manager
        )

    # Generate descriptive message
    profile_desc = "all profiles" if request.all_profiles else f"{len(request.profile_ids)} profiles"
    server_desc = f" using {len(selected_servers)} servers" if request.use_multi_server else " (single server)"

    return JobStartResponse(
        job_id=created_job_id,
        job_type="bulk_judge",
        status="running",
        message=f"Bulk judge operation started for {profile_desc}{server_desc}"
    )

async def _is_multi_server_job(job_id: UUID, pool) -> bool:
    """Check if a job is using multi-server processing."""
    async with pool.acquire() as conn:
        configuration = await conn.fetchval(
            "SELECT configuration FROM processing_jobs WHERE id = $1",
            job_id
        )

        if configuration:
            # Handle both dict and JSON string configurations
            if isinstance(configuration, str):
                try:
                    import json
                    configuration = json.loads(configuration)
                except (json.JSONDecodeError, TypeError):
                    return False
            
            if isinstance(configuration, dict):
                return configuration.get('use_multi_server', False)

    return False

async def _signal_workers_pause(job_id: UUID):
    """Signal workers to pause processing."""
    try:
        from ..data_access.worker_heartbeats import WorkerHeartbeatsRepository
        heartbeat_repo = WorkerHeartbeatsRepository()

        # Update worker status to paused
        active_workers = heartbeat_repo.get_active_workers(job_id)
        for worker in active_workers:
            # This would signal the worker process to pause
            # Implementation depends on inter-process communication method
            pass

    except Exception as e:
        print(f"Error signaling workers to pause: {e}")

async def _signal_workers_resume(job_id: UUID):
    """Signal workers to resume processing."""
    try:
        from ..data_access.worker_heartbeats import WorkerHeartbeatsRepository
        heartbeat_repo = WorkerHeartbeatsRepository()

        # Update worker status to active
        active_workers = heartbeat_repo.get_active_workers(job_id)
        for worker in active_workers:
            # This would signal the worker process to resume
            # Implementation depends on inter-process communication method
            pass

    except Exception as e:
        print(f"Error signaling workers to resume: {e}")

async def _signal_workers_cancel(job_id: UUID):
    """Signal workers to cancel processing and terminate processes."""
    import subprocess
    import signal
    
    try:
        from ..data_access.worker_heartbeats import WorkerHeartbeatsRepository
        heartbeat_repo = WorkerHeartbeatsRepository()

        # Update worker status to canceled in database
        active_workers = heartbeat_repo.get_active_workers(job_id)
        for worker in active_workers:
            heartbeat_repo.update_worker_status(worker.worker_id, "canceled")

        # Find and terminate worker processes
        try:
            # Get all judge_worker processes for this job
            result = subprocess.run([
                "ps", "aux"
            ], capture_output=True, text=True, check=True)
            
            processes_to_kill = []
            for line in result.stdout.split('\n'):
                if 'theseus_insight.workers.judge_worker' in line and str(job_id) in line:
                    # Extract PID (second column)
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            processes_to_kill.append(pid)
                        except ValueError:
                            continue
            
            # Terminate processes gracefully first, then force kill if needed
            for pid in processes_to_kill:
                try:
                    print(f"🛑 Terminating worker process {pid} for job {job_id}")
                    subprocess.run(["kill", "-TERM", str(pid)], check=False)
                except Exception as e:
                    print(f"Warning: Could not terminate process {pid}: {e}")
            
            # Wait a moment, then force kill any remaining processes
            import asyncio
            await asyncio.sleep(2)
            
            for pid in processes_to_kill:
                try:
                    # Check if process still exists
                    subprocess.run(["kill", "-0", str(pid)], check=True, capture_output=True)
                    # If we get here, process still exists, force kill it
                    print(f"🔥 Force killing worker process {pid} for job {job_id}")
                    subprocess.run(["kill", "-KILL", str(pid)], check=False)
                except subprocess.CalledProcessError:
                    # Process already terminated, which is what we want
                    pass
                except Exception as e:
                    print(f"Warning: Could not force kill process {pid}: {e}")
                    
        except Exception as e:
            print(f"Error terminating worker processes: {e}")

        # Clean up database resources
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            # Mark all tasks for this job as canceled
            await conn.execute(
                """
                UPDATE judge_task_queue 
                SET status = 'canceled', updated_at = NOW()
                WHERE job_id = $1 AND status IN ('pending', 'leased', 'in_progress')
                """,
                job_id
            )
            
            # Mark worker heartbeats as inactive instead of deleting (preserve failure history)
            await conn.execute(
                """
                UPDATE worker_heartbeats 
                SET status = 'inactive', 
                    last_heartbeat = NOW()
                WHERE job_id = $1 AND status != 'failed'
                """,
                job_id
            )
            
        print(f"✅ Cleaned up resources for job {job_id}")

    except Exception as e:
        print(f"Error signaling workers to cancel: {e}")
        import traceback
        traceback.print_exc()

async def _cleanup_failed_job(job_id: UUID):
    """Clean up resources for a failed job."""
    try:
        print(f"🧹 Cleaning up failed job {job_id}")
        
        # Use the same cleanup logic as cancellation
        await _signal_workers_cancel(job_id)
        
        # Update job status to failed if not already
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE processing_jobs 
                SET status = 'failed', completed_at = NOW()
                WHERE id = $1 AND status IN ('running', 'pending', 'paused')
                """,
                job_id
            )
        
        # Restore scheduled tasks
        await _restore_scheduled_tasks_on_cancel(job_id, pool)
        
        print(f"✅ Failed job {job_id} cleaned up")
        
    except Exception as e:
        print(f"Error cleaning up failed job {job_id}: {e}")
        import traceback
        traceback.print_exc()

def _get_job_description(job_type: str, configuration: Optional[Dict[str, Any]]) -> str:
    """Generate a human-readable description of a job."""
    if job_type == 'bulk_judge':
        if configuration:
            if configuration.get('use_multi_server'):
                servers = configuration.get('selected_servers', [])
                server_desc = f"{len(servers)} servers" if servers else "multiple servers"
                return f"Bulk judge using {server_desc}"
            else:
                return "Bulk judge (single server)"
        return "Bulk judge operation"
    elif job_type == 'harvest_judge':
        return "Harvest and judge operation"
    elif job_type == 'newsletter_generation':
        return "Newsletter generation"
    elif job_type == 'mindmap_generation':
        return "Mind-map generation"
    elif job_type == 'podcast_generation':
        return "Podcast generation"
    else:
        return f"{job_type.replace('_', ' ').title()}"

def _get_conflict_recommendations(conflicts: List[Dict[str, Any]], bulk_judge_running: bool) -> List[str]:
    """Generate recommendations for handling job conflicts."""
    recommendations = []

    if bulk_judge_running:
        recommendations.append("A bulk judge job is currently running. Consider pausing it before starting other jobs.")
        recommendations.append("Bulk judge jobs can take several hours and may impact system performance.")

    if len(conflicts) > 1:
        recommendations.append(f"There are {len(conflicts)} active jobs running. Consider waiting for completion or canceling non-essential jobs.")

    # Check for resource-intensive jobs
    resource_intensive = ['harvest_judge', 'bulk_judge', 'newsletter_generation']
    intensive_running = [c for c in conflicts if c['job_type'] in resource_intensive]

    if len(intensive_running) > 1:
        recommendations.append("Multiple resource-intensive jobs are running. This may impact system performance.")

    return recommendations

async def _check_job_conflicts():
    """Check for potential conflicts with running jobs."""
    pool = await get_connection_pool()

    try:
        async with pool.acquire() as conn:
            # Check for running bulk judge jobs
            running_jobs = await conn.fetch(
                """
                SELECT id, job_type, started_at, configuration
                FROM processing_jobs
                WHERE status IN ('running', 'pending')
                AND job_type IN ('bulk_judge', 'harvest_judge', 'newsletter_generation', 'mindmap_generation', 'podcast_generation')
                """
            )

            conflicts = []
            bulk_judge_running = False

            for job in running_jobs:
                job_info = {
                    "job_id": str(job['id']),
                    "job_type": job['job_type'],
                    "started_at": job['started_at'].isoformat() if job['started_at'] else None,
                    "description": _get_job_description(job['job_type'], job['configuration'])
                }

                if job['job_type'] == 'bulk_judge':
                    bulk_judge_running = True
                    # Check if it's a multi-server job
                    configuration = job['configuration']
                    if configuration and isinstance(configuration, dict) and configuration.get('use_multi_server', False):
                        job_info["multi_server"] = True
                        job_info["servers"] = configuration.get('selected_servers', [])

                conflicts.append(job_info)

            return {
                "has_conflicts": len(conflicts) > 0,
                "bulk_judge_running": bulk_judge_running,
                "conflicts": conflicts,
                "recommendations": _get_conflict_recommendations(conflicts, bulk_judge_running)
            }

    except Exception as e:
        # Return empty result on error to avoid blocking job submission
        return {
            "has_conflicts": False,
            "bulk_judge_running": False,
            "conflicts": [],
            "recommendations": []
        }

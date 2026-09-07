from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import pathlib
import json
import uuid
from datetime import datetime, timedelta, date

from .api.routers import all_routers, websocket_manager
from .api.dependencies import CREDENTIAL_KEYS
from .api.tasks import task_manager
from .scheduler import scheduler
from .api.models import ModelConfig, TTSModelConfig, OrchestrationConfig
from .data_access import SettingsRepository

from pydantic import BaseModel, Field, ValidationError
from typing import List

from .api.models import (
    Model, ModelCreate, Paper, Run, PaginatedResponse,
    NewsletterConfig, RunStatus, OrchestrationConfig,
    VisualizerSettings, ArxivCategoriesConfig, ModelProvider,
    EmailRecipients, ResearchInterests, ModelConfig, TTSModelConfig,
    PodcastGenerationParams, PodcastListItemResponse, PodcastDetailResponse,
    PaperApiResponse, PaginatedPapersResponse, NewsletterRunParams,
    SimilaritySearchRequest, SimilaritySearchResponse, SimilarPapersRequest, SimilarPapersResponse,
    HybridSearchRequest, HybridSearchResponse, ResearchAgentRunRequest, ResearchAgentRunResponse,
    LiteratureReviewResult, ResearchAgentModelConfigApi
)
from .api.tasks import task_manager, TaskStatus
from .theseus_insight import TheseusInsight
from . import theseus_insight as ti_module
from .utils.path_resolver import get_config_path, config_file_exists

# Determine base path for static files
IS_RUNNING_IN_DOCKER = os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if IS_RUNNING_IN_DOCKER:
    STATIC_FILES_BASE_DIR = pathlib.Path("static_frontend")
else:
    # Development mode - look for theseus-ui/dist relative to project root
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    STATIC_FILES_BASE_DIR = PROJECT_ROOT / "theseus-ui" / "dist"

print(f"INFO:     Static files directory: {STATIC_FILES_BASE_DIR}")
print(f"INFO:     Static files exist: {STATIC_FILES_BASE_DIR.exists() if STATIC_FILES_BASE_DIR else False}")
if STATIC_FILES_BASE_DIR and STATIC_FILES_BASE_DIR.exists():
    print(f"INFO:     Frontend index.html exists: {(STATIC_FILES_BASE_DIR / 'index.html').exists()}")
    print(f"INFO:     Frontend assets directory exists: {(STATIC_FILES_BASE_DIR / 'assets').exists()}")

STATIC_ASSETS_DIR = STATIC_FILES_BASE_DIR / "assets"
STATIC_INDEX_HTML = STATIC_FILES_BASE_DIR / "index.html"

# Database and other shared objects are now in dependencies module

# Lifespan context manager
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    app_instance.state.ready = False
    # Startup logic
    print("INFO:     Starting up Theseus Insight API...")
    try:
        # Check and apply database migrations
        from .db.migrations import check_and_apply_migrations
        print("INFO:     Checking database migrations...")
        await check_and_apply_migrations()
        
        # Directory structure is now created during database initialization

        # Load credentials from DB (encrypted) and apply to environment
        for key in CREDENTIAL_KEYS:
            if key == "OLLAMA_URL":
                val = SettingsRepository.get(key)
            else:
                val = SettingsRepository.get_secret_setting(key)  # Use encrypted secrets from repository
            if val:
                os.environ[key] = val
                if hasattr(ti_module, key):
                    setattr(ti_module, key, val)
        
        # Handle OLLAMA_URL passthrough for Docker networking
        ollama_passthrough = os.getenv("OLLAMA_PASSTHROUGH", "true").lower() == "true"
        is_running_in_docker = os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true"
        ollama_url = os.getenv("OLLAMA_URL", "")
        
        if ollama_passthrough and is_running_in_docker and ollama_url:
            # Replace localhost, 127.0.0.1, or ::1 with host.docker.internal for Docker containers
            import re
            # Match localhost, 127.0.0.1, or ::1 with optional port
            localhost_pattern = r'(https?://)(?:localhost|127\.0\.0\.1|::1)(:(\d+))?'
            if re.search(localhost_pattern, ollama_url):
                # Replace with host.docker.internal
                new_ollama_url = re.sub(localhost_pattern, r'\1host.docker.internal\2', ollama_url)
                os.environ["OLLAMA_URL"] = new_ollama_url
                if hasattr(ti_module, "OLLAMA_URL"):
                    setattr(ti_module, "OLLAMA_URL", new_ollama_url)
                print(f"INFO:     OLLAMA_PASSTHROUGH enabled: Modified OLLAMA_URL from {ollama_url} to {new_ollama_url}")
            else:
                print(f"INFO:     OLLAMA_PASSTHROUGH enabled but OLLAMA_URL doesn't use localhost: {ollama_url}")
        elif not ollama_passthrough and is_running_in_docker:
            print("INFO:     OLLAMA_PASSTHROUGH disabled: Using container networking for Ollama")
        
        required_env_vars = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GMAIL_SENDER_ADDRESS",
            "CLIENT_SECRET", "PROJECT_ID", "GMAIL_APP_PASSWORD"
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")

        # Initialize model providers
        from .data_access import ModelProviderRepository
        try:
            ModelProviderRepository.ensure_providers_exist()
        except Exception as e:
            print(f"ERROR:    Failed to initialize model providers: {e}")

        if SettingsRepository.get("orchestration") is None:
            print("INFO:     Orchestration settings not found in DB. Populating from JSON file...")
            orchestration_json_path = get_config_path('orchestration.json')
            if config_file_exists('orchestration.json'):
                try:
                    with open(orchestration_json_path, 'r') as f:
                        default_orchestration_data = json.load(f)
                    default_embedding_model = ModelConfig(model_name='Alibaba-NLP/gte-modernbert-base', model_type='sentence-transformers', trust_remote_code=True)
                    default_judge_model = ModelConfig(model_name='phi4-mini:3.8b-q8_0', model_type='ollama', max_new_tokens=512, temperature=0.1, num_ctx=4096)
                    default_content_extraction_model = ModelConfig(model_name='gemma3:27b-it-qat', model_type='ollama', max_new_tokens=4096, temperature=0.1, num_ctx=131072)
                    default_newsletter_sections_model = ModelConfig(model_name='gemma3:27b-it-qat', model_type='ollama', max_new_tokens=4096, temperature=0.1, num_ctx=131072)
                    default_newsletter_intro_model = ModelConfig(model_name='gemini-2.0-flash', model_type='gemini', max_new_tokens=4096, temperature=0.1, num_ctx=131072)
                    default_podcast_model = ModelConfig(model_name='gemini-2.0-flash', model_type='gemini', max_new_tokens=8192, temperature=0.1, num_ctx=131072)
                    default_tts_model = TTSModelConfig(tts_provider='openai', tts_model_name='tts-1', speaker_1_voice='sage', speaker_1_speed=1.0, speaker_2_voice='ash', speaker_2_speed=1.0)
                    orchestration_config = OrchestrationConfig(
                        embedding_model=ModelConfig(**default_orchestration_data.get('embedding_model', default_embedding_model.dict())),
                        judge_model=ModelConfig(**default_orchestration_data.get('judge_model', default_judge_model.dict())),
                        content_extraction_model=ModelConfig(**default_orchestration_data.get('content_extraction_model', default_content_extraction_model.dict())),
                        newsletter_sections_model=ModelConfig(**default_orchestration_data.get('newsletter_sections_model', default_newsletter_sections_model.dict())),
                        newsletter_intro_model=ModelConfig(**default_orchestration_data.get('newsletter_intro_model', default_newsletter_intro_model.dict())),
                        podcast_model=ModelConfig(**default_orchestration_data.get('podcast_model', default_podcast_model.dict())) if default_orchestration_data.get('podcast_model') else default_podcast_model,
                        tts_model=TTSModelConfig(**default_orchestration_data.get('tts_model', default_tts_model.dict())) if default_orchestration_data.get('tts_model') else default_tts_model
                    )
                    SettingsRepository.set("orchestration", orchestration_config.json())
                    print("INFO:     Successfully populated orchestration settings into DB.")
                except Exception as e:
                    print(f"ERROR: Failed to load or parse orchestration.json for DB pre-population: {e}")
            else:
                print(f"Warning: orchestration.json not found at {orchestration_json_path}.")
        else:
            print("INFO:     Orchestration settings found in DB. Skipping pre-population.")
        if SettingsRepository.get("research_interests") is None:
            print("INFO:     Research interests not found in DB. Populating from TXT file...")
            research_txt_path = get_config_path('research_interests.txt')
            if config_file_exists('research_interests.txt'):
                try:
                    with open(research_txt_path, 'r') as f:
                        default_interests = f.read().strip()
                    SettingsRepository.set("research_interests", default_interests)
                    print("INFO:     Successfully populated research interests into DB.")
                except Exception as e:
                    print(f"ERROR: Failed to load research_interests.txt for DB pre-population: {e}")
            else:
                print(f"Warning: research_interests.txt not found at {research_txt_path}.")
        else:
            print("INFO:     Research interests settings found in DB. Skipping pre-population.")
        
        # Start task manager workers
        print("INFO:     Starting task manager workers...")
        try:
            await task_manager.start_worker()
            print("INFO:     Task manager workers started successfully.")
        except Exception as e:
            raise RuntimeError("Task workers failed to start") from e
        
        # Run media file cleanup
        print("INFO:     Running media file cleanup...")
        try:
            # Allow configuring cleanup age via environment variable
            cleanup_age_days = int(os.getenv("MEDIA_CLEANUP_AGE_DAYS", "30"))
            cleanup_old_media_files(max_age_days=cleanup_age_days)
        except Exception as e:
            print(f"Warning: Media file cleanup encountered an error: {e}")
            # Continue startup even if cleanup fails
        
        # Clean up any orphaned worker processes from previous runs
        print("INFO:     Running startup cleanup (stuck jobs and orphaned processes)...")
        try:
            from .startup_cleanup import cleanup_stuck_jobs_and_processes
            await cleanup_stuck_jobs_and_processes()
        except Exception as e:
            print(f"Warning: Startup cleanup failed: {e}")
        
        # Start scheduler for nightly jobs
        print("INFO:     Starting scheduler for nightly jobs...")
        try:
            await scheduler.start()
            print("INFO:     Scheduler started successfully.")
        except Exception as e:
            raise RuntimeError("Scheduler failed to start") from e
            
    except Exception as e:
        await task_manager.cleanup()
        await scheduler.stop()
        raise RuntimeError("Theseus startup failed; inspect startup logs") from e
    print("INFO:     Theseus Insight API startup complete.")
    app_instance.state.ready = True
    yield
    app_instance.state.ready = False
    # Shutdown logic
    print("INFO:     Shutting down Theseus Insight API...")
    try:
        # Clean up any running worker processes on shutdown
        print("INFO:     Cleaning up worker processes...")
        try:
            from .startup_cleanup import cleanup_stuck_jobs_and_processes
            await cleanup_stuck_jobs_and_processes()
        except Exception as e:
            print(f"Warning: Worker process cleanup failed: {e}")
        
        # Stop scheduler
        await scheduler.stop()
        print("INFO:     Scheduler stopped.")
        
        # Clean up TaskManager resources
        await task_manager.cleanup()
        print("INFO:     TaskManager cleanup completed.")
        
        # Clean up WebSocket connections
        await websocket_manager.cleanup_all()
        print("INFO:     WebSocket connections closed.")
    except Exception as e:
        print(f"Error during shutdown: {e}")
    print("INFO:     Theseus Insight API shutdown complete.")

# Initialize FastAPI app
app = FastAPI(
    title="Theseus Insight API",
    description="Backend API for Theseus Insight research paper analysis platform",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
if "*" in CORS_ORIGINS:
    raise ValueError("CORS_ORIGINS must contain explicit origins")
from .security import AccessMiddleware
app.add_middleware(AccessMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Include all routers
for router in all_routers:
    app.include_router(router)

# --- All route definitions have been moved to separate router files --- #

# --- Serve React Frontend --- #
# This block MUST come AFTER all other API routes have been defined.
print(f"INFO:     Setting up static file serving...")
print(f"INFO:     Assets directory: {STATIC_ASSETS_DIR} (exists: {STATIC_ASSETS_DIR.exists() if STATIC_ASSETS_DIR else False})")
print(f"INFO:     Base directory: {STATIC_FILES_BASE_DIR} (exists: {STATIC_FILES_BASE_DIR.exists() if STATIC_FILES_BASE_DIR else False})")

if STATIC_ASSETS_DIR and STATIC_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_ASSETS_DIR), name="assets")
    print(f"INFO:     Mounted /assets from {STATIC_ASSETS_DIR}")
else:
    print(f"WARNING: Static assets directory not found at {STATIC_ASSETS_DIR}. Frontend assets will not be served.")

# Mount the entire static frontend directory to serve public assets like logo.png
if STATIC_FILES_BASE_DIR and STATIC_FILES_BASE_DIR.exists():
    # Mount with a specific route pattern to avoid conflicts
    app.mount("/static", StaticFiles(directory=STATIC_FILES_BASE_DIR), name="static")
    print(f"INFO:     Mounted /static from {STATIC_FILES_BASE_DIR}")
    
    # Also serve specific public assets directly at root level
    @app.get("/logo.png")
    async def serve_logo():
        logo_path = STATIC_FILES_BASE_DIR / "logo.png"
        if logo_path.exists():
            return FileResponse(logo_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="Logo not found")
    
    @app.get("/favicon.ico")
    async def serve_favicon():
        favicon_path = STATIC_FILES_BASE_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/x-icon")
        raise HTTPException(status_code=404, detail="Favicon not found")
else:
    print(f"WARNING: Static files base directory not found at {STATIC_FILES_BASE_DIR}. Frontend will not be served.")


# Scheduler diagnostic endpoints
@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get the current status of the scheduler and its jobs."""
    try:
        # Get basic scheduler status
        scheduler_status = scheduler.get_job_status()
        
        # Add additional diagnostic information
        from .data_access import PaperRepository, TopicMetricsRepository
        
        # Check recent papers count
        cutoff_date = (datetime.now() - timedelta(days=30)).date()
        recent_papers_count = len(PaperRepository.get_papers_with_embeddings(limit=10000))
        
        # Check if we have any topic metrics
        latest_period_end = TopicMetricsRepository.get_latest_period_end("week")
        
        # Get recent logs for scheduler-related tasks
        from .data_access import LogsRepository
        recent_logs = LogsRepository.recent(limit=10)
        scheduler_logs = [log for log in recent_logs if 'trend' in log.get('task_id', '').lower()]
        
        return {
            "scheduler_status": scheduler_status,
            "diagnostics": {
                "recent_papers_count": recent_papers_count,
                "latest_metric_period": latest_period_end.isoformat() if latest_period_end else None,
                "recent_scheduler_logs": scheduler_logs,
                "next_scheduled_run": None,  # Will be filled if scheduler is running
                "minimum_papers_required": 100,
                "server_time": datetime.now().isoformat(),
                "timezone": "UTC"
            }
        }
    except Exception as e:
        return {"error": f"Failed to get scheduler status: {str(e)}"}

@app.get("/api/scheduler/logs")
async def get_scheduler_logs():
    """Get recent scheduler-related logs."""
    try:
        from .data_access import LogsRepository
        all_logs = LogsRepository.recent(limit=50)
        
        # Filter for scheduler-related logs
        scheduler_logs = [
            log for log in all_logs 
            if any(keyword in log.get('task_id', '').lower() or keyword in log.get('status', '').lower() 
                   for keyword in ['trend', 'schedule', 'nightly', 'cleanup'])
        ]
        
        return {"logs": scheduler_logs}
    except Exception as e:
        return {"error": f"Failed to get scheduler logs: {str(e)}"}

@app.post("/api/scheduler/test")
async def test_scheduler():
    """Test the scheduler by running a simple test job."""
    try:
        result = scheduler.schedule_test_job()
        return result
    except Exception as e:
        return {"error": f"Failed to schedule test job: {str(e)}"}

@app.get("/api/scheduler/diagnostics")
async def get_scheduler_diagnostics():
    """Get comprehensive diagnostics to understand why scheduler might not be processing."""
    try:
        from .data_access import (
            PaperRepository, TopicMetricsRepository, 
            PaperTopicsRepository, TrendsRepository,
            SettingsRepository
        )
        
        # Check papers data
        cutoff_date_30 = (datetime.now() - timedelta(days=30)).date()
        cutoff_date_7 = (datetime.now() - timedelta(days=7)).date()
        
        # Get papers with embeddings
        all_papers = PaperRepository.get_papers_with_embeddings(limit=10000)
        
        # Count papers by date ranges
        recent_papers_30d = [p for p in all_papers if p.get('date') and 
                            (isinstance(p['date'], date) and p['date'] >= cutoff_date_30 or
                             isinstance(p['date'], str) and datetime.strptime(p['date'], '%Y-%m-%d').date() >= cutoff_date_30)]
        
        recent_papers_7d = [p for p in all_papers if p.get('date') and 
                           (isinstance(p['date'], date) and p['date'] >= cutoff_date_7 or
                            isinstance(p['date'], str) and datetime.strptime(p['date'], '%Y-%m-%d').date() >= cutoff_date_7)]
        
        # Check topic assignments
        papers_needing_topics = PaperTopicsRepository.get_papers_needing_topic_assignment()
        
        # Check latest metrics
        latest_weekly_metrics = TopicMetricsRepository.get_latest_period_end("week")
        latest_monthly_metrics = TopicMetricsRepository.get_latest_period_end("month")
        
        # Check configuration
        orchestration_config = SettingsRepository.get("orchestration")
        research_interests = SettingsRepository.get("research_interests")
        
        # Check for errors in recent logs
        from .data_access import LogsRepository
        recent_logs = LogsRepository.recent(limit=100)
        error_logs = [log for log in recent_logs if 'error' in log.get('status', '').lower() or 'failed' in log.get('status', '').lower()]
        
        # Analyze potential issues
        issues = []
        recommendations = []
        
        if len(recent_papers_30d) < 100:
            issues.append(f"Low paper count: Only {len(recent_papers_30d)} papers in last 30 days (minimum required: 100)")
            recommendations.append("Consider reducing the minimum papers threshold or harvesting more papers")
        
        if len(recent_papers_7d) < 10:
            issues.append(f"Very low recent activity: Only {len(recent_papers_7d)} papers in last 7 days")
            recommendations.append("Check if paper harvesting is working correctly")
        
        if len(papers_needing_topics) > 100:
            issues.append(f"Many papers without topics: {len(papers_needing_topics)} papers need research interest classification")
            recommendations.append("Run research interest classification to assign papers to your interests")
        
        if not orchestration_config:
            issues.append("Missing orchestration configuration")
            recommendations.append("Configure orchestration settings via /api/settings")
        
        if not research_interests:
            issues.append("Missing research interests configuration")
            recommendations.append("Configure research interests via /api/settings")
        
        if latest_weekly_metrics is None:
            issues.append("No weekly metrics found - research interest classification may not have run")
            recommendations.append("Run research interest classification via /api/trends/research-interests/recompute")
        
        if error_logs:
            issues.append(f"Recent errors detected: {len(error_logs)} error logs found")
            recommendations.append("Check error logs for detailed failure information")
        
        # Status assessment
        if len(issues) == 0:
            status = "healthy"
        elif len(issues) <= 2:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "paper_data": {
                "total_papers_with_embeddings": len(all_papers),
                "papers_last_30_days": len(recent_papers_30d),
                "papers_last_7_days": len(recent_papers_7d),
                "papers_needing_topics": len(papers_needing_topics),
                "minimum_required_for_processing": 100
            },
            "metrics_data": {
                "latest_weekly_metrics": latest_weekly_metrics.isoformat() if latest_weekly_metrics else None,
                "latest_monthly_metrics": latest_monthly_metrics.isoformat() if latest_monthly_metrics else None,
                "has_historical_data": latest_weekly_metrics is not None
            },
            "configuration": {
                "has_orchestration_config": orchestration_config is not None,
                "has_research_interests": research_interests is not None,
                "orchestration_config_length": len(orchestration_config) if orchestration_config else 0,
                "research_interests_length": len(research_interests) if research_interests else 0
            },
            "recent_activity": {
                "total_recent_logs": len(recent_logs),
                "error_logs": len(error_logs),
                "recent_errors": error_logs[:5]  # Show first 5 errors
            },
            "issues": issues,
            "recommendations": recommendations,
            "next_steps": [
                "Check /api/scheduler/status for job scheduling details",
                "Review /api/scheduler/logs for scheduler-specific logs",
                "Test scheduler with /api/scheduler/test"
            ]
        }
    except Exception as e:
        return {"error": f"Failed to get scheduler diagnostics: {str(e)}"}

@app.get("/health/live", include_in_schema=False)
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready", include_in_schema=False)
async def readiness():
    from .db.async_bridge import run_repo
    from .db import get_cursor
    from fastapi.responses import JSONResponse
    def ping():
        with get_cursor() as cur:
            cur.execute("SELECT 1")
    workers = (task_manager.general_worker_task, task_manager.visualizer_worker_task)
    if not getattr(app.state, "ready", False) or any(w is None or w.done() for w in workers) or not (scheduler.is_running or getattr(scheduler, "standby", False)):
        return JSONResponse({"status": "not_ready", "dependency": "runtime"}, status_code=503)
    try:
        import asyncio
        await asyncio.wait_for(run_repo(ping), timeout=3)
    except Exception:
        return JSONResponse({"status": "not_ready", "dependency": "database"}, status_code=503)
    return {"status": "ready"}


@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    """Serves the index.html for any path not caught by API routes or specific static files."""

    # Skip API routes and specific files
    if full_path.startswith("api/") or full_path.startswith("ws/") or full_path.startswith("docs") or full_path.startswith("openapi"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    if STATIC_INDEX_HTML and STATIC_INDEX_HTML.exists():
        print(f"INFO:     Serving React app from {STATIC_INDEX_HTML} for path: /{full_path}")
        return FileResponse(STATIC_INDEX_HTML)
    else:
        # Enhanced error message with debugging information
        error_details = {
            "message": "Frontend not available",
            "static_files_dir": str(STATIC_FILES_BASE_DIR) if STATIC_FILES_BASE_DIR else "Not set",
            "static_files_exists": STATIC_FILES_BASE_DIR.exists() if STATIC_FILES_BASE_DIR else False,
            "index_html_path": str(STATIC_INDEX_HTML) if STATIC_INDEX_HTML else "Not set",
            "index_html_exists": STATIC_INDEX_HTML.exists() if STATIC_INDEX_HTML else False,
            "is_docker": IS_RUNNING_IN_DOCKER,
            "current_working_dir": str(pathlib.Path.cwd()),
            "requested_path": full_path
        }

        detail_message = f"Frontend index.html not found at {STATIC_INDEX_HTML}."

        if not IS_RUNNING_IN_DOCKER:
            detail_message += " Ensure the frontend has been built (e.g., `npm run build` in `theseus-ui` directory)."

        print(f"ERROR: Frontend serving failed. Details: {error_details}")
        raise HTTPException(status_code=404, detail=detail_message)

def cleanup_old_media_files(max_age_days: int = 30):
    """
    Clean up old podcast and visualization files that are older than max_age_days.
    This preserves database records but removes actual media files to save disk space.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        total_deleted = 0
        total_size_freed = 0

        # Directories to clean
        cleanup_dirs = [
            "data/podcasts",
            "data/visualizations",
            "data/temp"  # Also clean temp files
        ]

        for base_dir in cleanup_dirs:
            if not os.path.exists(base_dir):
                continue

            print(f"INFO:     Cleaning up old files in {base_dir}...")
            dir_deleted = 0
            dir_size_freed = 0

            # Walk through all subdirectories and files
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Get file modification time
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                        # Check if file is older than cutoff
                        if file_mtime < cutoff_date:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)

                            dir_deleted += 1
                            dir_size_freed += file_size
                            print(f"INFO:     Deleted old file: {file_path} (age: {(datetime.now() - file_mtime).days} days)")

                    except Exception as e:
                        print(f"Warning: Could not delete file {file_path}: {e}")
                        continue

            # Clean up empty directories after file deletion
            try:
                for root, dirs, files in os.walk(base_dir, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            # Only remove if directory is empty and not the base directory
                            if not os.listdir(dir_path) and dir_path != base_dir:
                                os.rmdir(dir_path)
                                print(f"INFO:     Removed empty directory: {dir_path}")
                        except Exception as e:
                            # Directory not empty or other error, skip
                            continue
            except Exception as e:
                print(f"Warning: Error during directory cleanup in {base_dir}: {e}")

            total_deleted += dir_deleted
            total_size_freed += dir_size_freed

            if dir_deleted > 0:
                size_mb = dir_size_freed / (1024 * 1024)
                print(f"INFO:     Cleaned {dir_deleted} files from {base_dir}, freed {size_mb:.2f} MB")

        if total_deleted > 0:
            total_size_mb = total_size_freed / (1024 * 1024)
            print(f"INFO:     Total cleanup: {total_deleted} files deleted, {total_size_mb:.2f} MB freed")
        else:
            print(f"INFO:     No old files found to clean up (older than {max_age_days} days)")

    except Exception as e:
        print(f"ERROR: Failed to run media file cleanup: {e}")
        # Don't raise the error - we don't want cleanup failure to prevent API startup

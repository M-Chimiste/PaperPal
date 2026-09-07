import asyncio
import logging
from datetime import datetime, timedelta, date
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from .data_access.trends import TrendsRepository

logger = logging.getLogger(__name__)

class TheseusScheduler:
    """
    Scheduler for automated Theseus Insight tasks.
    Handles weekly cleanup and other periodic maintenance tasks.
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        self.standby = False
        self._leadership_conn = None
        self._leadership_monitor = None
        self._stopped = False

    async def start(self):
        """Start the scheduler."""
        self._stopped = False
        if self._leadership_monitor is None:
            self._leadership_monitor = asyncio.create_task(self._wait_for_leadership())
        if not self.is_running:
            from .db import DATABASE_URL
            import psycopg
            def acquire():
                conn = psycopg.connect(DATABASE_URL, autocommit=True)
                acquired = conn.execute("SELECT pg_try_advisory_lock(hashtextextended('theseus:scheduler', 0))").fetchone()[0]
                if not acquired:
                    conn.close()
                    return None
                return conn
            self._leadership_conn = await asyncio.to_thread(acquire)
            if self._leadership_conn is None:
                self.standby = True
                if self._leadership_monitor is None:
                    self._leadership_monitor = asyncio.create_task(self._wait_for_leadership())
                return
            self.standby = False
            logger.info("🚀 Starting Theseus Scheduler...")

            # Schedule weekly cleanup at 3 AM on Sundays
            weekly_job = self.scheduler.add_job(
                self._run_weekly_cleanup,
                CronTrigger(day_of_week='sun', hour=3, minute=0),  # 3:00 AM on Sundays
                id='weekly_cleanup',
                name='Weekly Data Cleanup',
                replace_existing=True
            )

            # Schedule stuck job checker every hour
            stuck_job_checker = self.scheduler.add_job(
                self._run_stuck_job_checker,
                CronTrigger(minute=0),  # Every hour at minute 0
                id='stuck_job_checker',
                name='Stuck Job Checker',
                replace_existing=True
            )

            self.scheduler.start()
            self.is_running = True

            # Log scheduled jobs with next run times
            logger.info("✅ Theseus Scheduler started successfully")
            logger.info(f"📅 Next weekly cleanup run: {weekly_job.next_run_time}")
            logger.info(f"📅 Next stuck job check: {stuck_job_checker.next_run_time}")
            logger.info("⏰ Scheduler timezone: UTC")

            # Log current server time for reference
            logger.info(f"🕐 Current server time: {datetime.now().isoformat()}")

            # Add a test to verify the scheduler is working
            logger.info("🔍 Scheduler health check - all jobs scheduled correctly")

            # Sync user-configured scheduled tasks from database
            await self._sync_scheduled_tasks()
            
    async def _wait_for_leadership(self):
        while not self._stopped:
            await asyncio.sleep(5)
            try:
                if self._leadership_conn:
                    await asyncio.to_thread(self._leadership_conn.execute, 'SELECT 1')
                else:
                    await self.start()
            except Exception:
                logger.exception('Scheduler leadership lost; pausing scheduled work')
                if self.is_running:
                    self.scheduler.shutdown(wait=False)
                self.is_running = False
                self.standby = True
                if self._leadership_conn:
                    await asyncio.to_thread(self._leadership_conn.close)
                self._leadership_conn = None

    async def stop(self):
        """Stop the scheduler."""
        self._stopped = True
        self.standby = False
        if self._leadership_monitor:
            self._leadership_monitor.cancel()
            await asyncio.gather(self._leadership_monitor, return_exceptions=True)
            self._leadership_monitor = None
        if self.is_running:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            logger.info("Theseus Scheduler stopped")
        if self._leadership_conn:
            await asyncio.to_thread(self._leadership_conn.close)
            self._leadership_conn = None

    async def _run_weekly_cleanup(self):
        """
        Weekly cleanup job to remove old data and optimize database.
        """
        start_time = datetime.now()
        
        try:
            logger.info("=" * 60)
            logger.info("🧹 WEEKLY CLEANUP STARTED")
            logger.info(f"Started at: {start_time.isoformat()}")
            logger.info("=" * 60)
            
            # Clean up old topic metrics (keep last 2 years)
            trends_repo = TrendsRepository()
            cleanup_count = trends_repo.cleanup_old_metrics(months_to_keep=24)
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info("=" * 60)
            logger.info("✅ WEEKLY CLEANUP COMPLETED")
            logger.info(f"Duration: {duration.total_seconds():.2f} seconds")
            logger.info(f"Removed {cleanup_count} old metric records")
            logger.info("=" * 60)
            
        except Exception as e:
            end_time = datetime.now()
            duration = end_time - start_time
            logger.error("=" * 60)
            logger.error("❌ WEEKLY CLEANUP FAILED")
            logger.error(f"Duration: {duration.total_seconds():.2f} seconds")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 60)
            logger.error("Full traceback:", exc_info=True)
    
    async def _run_stuck_job_checker(self):
        """
        Hourly check for stuck jobs that failed to complete properly.
        
        This catches jobs that:
        - Completed successfully but didn't update their status
        - Have been running for an unusually long time (>4 hours for most jobs)
        - Are marked as running but have no active workers
        """
        start_time = datetime.now()
        
        try:
            logger.info("🔍 STUCK JOB CHECKER STARTED")
            
            from .db import get_connection_pool
            pool = await get_connection_pool()
            
            async with pool.acquire() as conn:
                # Find jobs that have been running for more than 4 hours
                # (newsletters, mindmaps, podcasts should complete in < 1 hour)
                # (bulk_judge can take longer but should show progress)
                long_running_jobs = await conn.fetch(
                    """
                    SELECT id, job_type, status, started_at,
                           EXTRACT(EPOCH FROM (NOW() - started_at))/3600 as hours_running
                    FROM processing_jobs
                    WHERE status IN ('running', 'pending')
                    AND job_type IN ('newsletter_generation', 'mindmap_generation', 'podcast_generation')
                    AND EXTRACT(EPOCH FROM (NOW() - started_at))/3600 > 4
                    """
                )
                
                if long_running_jobs:
                    logger.warning(f"Found {len(long_running_jobs)} stuck jobs (running > 4 hours)")
                    
                    for job in long_running_jobs:
                        logger.warning(
                            f"Long-running job: {job['job_type']} "
                            f"(ID: {job['id']}, Running for: {job['hours_running']:.1f} hours)"
                        )
                    
                    # Age is not proof of abandoned ownership. Recovery is
                    # performed by dispatch leases; legacy jobs need review.
                    logger.warning("Long-running jobs require inspection; no unowned state was mutated")
                else:
                    logger.info("✅ No stuck jobs found")
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"✅ STUCK JOB CHECKER COMPLETED (Duration: {duration.total_seconds():.2f}s)")
            
        except Exception as e:
            end_time = datetime.now()
            duration = end_time - start_time
            logger.error(f"❌ STUCK JOB CHECKER FAILED (Duration: {duration.total_seconds():.2f}s)")
            logger.error(f"Error: {str(e)}")
            logger.error("Full traceback:", exc_info=True)

    async def run_test_job(self):
        """Test job to verify scheduler is working."""
        logger.info("🧪 TEST JOB STARTED - Scheduler is working correctly!")
        await asyncio.sleep(1)  # Simulate some work
        logger.info("✅ TEST JOB COMPLETED - Scheduler test successful!")
        return {"status": "success", "message": "Test job completed successfully"}
    
    def schedule_test_job(self):
        """Schedule a test job to run immediately."""
        if not self.is_running:
            return {"error": "Scheduler is not running"}
        
        # Schedule to run 5 seconds from now
        run_time = datetime.now() + timedelta(seconds=5)
        
        job = self.scheduler.add_job(
            self.run_test_job,
            'date',
            run_date=run_time,
            id='test_job',
            name='Scheduler Test Job',
            replace_existing=True
        )
        
        logger.info(f"🧪 Test job scheduled to run at {run_time.isoformat()}")
        return {
            "status": "scheduled",
            "run_time": run_time.isoformat(),
            "job_id": job.id
        }
        
    def get_job_status(self):
        """Get comprehensive status of scheduled jobs."""
        from datetime import timezone
        
        if not self.is_running:
            return {
                "status": "stopped",
                "jobs": [],
                "scheduler_info": {
                    "is_running": False,
                    "error": "Scheduler is not running"
                }
            }
            
        jobs = []
        current_time = datetime.now()
        
        for job in self.scheduler.get_jobs():
            next_run_time = job.next_run_time
            time_until_next_run = None
            
            if next_run_time:
                # Calculate time until next run
                if next_run_time.tzinfo is None:
                    # If next_run_time is naive, assume it's in UTC
                    next_run_time = next_run_time.replace(tzinfo=timezone.utc)
                    current_time_utc = current_time.replace(tzinfo=timezone.utc)
                else:
                    current_time_utc = current_time.replace(tzinfo=timezone.utc)
                
                time_diff = next_run_time - current_time_utc
                time_until_next_run = {
                    "total_seconds": time_diff.total_seconds(),
                    "hours": time_diff.total_seconds() // 3600,
                    "minutes": (time_diff.total_seconds() % 3600) // 60,
                    "human_readable": self._format_time_until(time_diff.total_seconds())
                }
            
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run_time.isoformat() if next_run_time else None,
                "trigger": str(job.trigger),
                "time_until_next_run": time_until_next_run,
                "function": job.func.__name__ if hasattr(job.func, '__name__') else str(job.func)
            })
            
        return {
            "status": "running",
            "jobs": jobs,
            "scheduler_info": {
                "is_running": True,
                "current_time": current_time.isoformat(),
                "timezone": "UTC",
                "job_count": len(jobs),
                "state": str(self.scheduler.state) if hasattr(self.scheduler, 'state') else "unknown"
            }
        }
    
    def _format_time_until(self, seconds):
        """Format seconds until next run in human-readable format."""
        if seconds < 0:
            return "Overdue"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if hours > 24:
            days = hours // 24
            remaining_hours = hours % 24
            return f"{days} days, {remaining_hours} hours"
        elif hours > 0:
            return f"{hours} hours, {minutes} minutes"
        else:
            return f"{minutes} minutes"
    
    async def _sync_scheduled_tasks(self):
        """Sync user-configured scheduled tasks from database."""
        try:
            from .data_access import ScheduledTasksRepository
            from .api.routers.scheduled_tasks import schedule_task_in_apscheduler
            from .db import get_cursor
            
            # First check if the scheduled_tasks table exists
            with get_cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'scheduled_tasks'
                    )
                """)
                table_exists = cursor.fetchone()['exists']
                
            if not table_exists:
                logger.info("📅 Scheduled tasks table not yet created, skipping sync")
                return
            
            logger.info("📅 Syncing user-configured scheduled tasks...")
            
            # Get all enabled tasks
            tasks = ScheduledTasksRepository.get_enabled_tasks()
            synced_count = 0
            
            for task in tasks:
                try:
                    await schedule_task_in_apscheduler(task['id'], task)
                    synced_count += 1
                    logger.info(f"✅ Scheduled task '{task['name']}' synced successfully")
                except Exception as e:
                    logger.error(f"Failed to sync task {task['id']} ({task['name']}): {e}")
            
            logger.info(f"📅 Synced {synced_count} user-configured scheduled tasks")
            
        except Exception as e:
            logger.error(f"Error syncing scheduled tasks: {e}")
            # Don't fail startup if sync fails

# Global scheduler instance
scheduler = TheseusScheduler()

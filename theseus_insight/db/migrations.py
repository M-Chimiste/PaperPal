"""Automatic database migration system for Theseus Insight.

This module checks and applies database migrations automatically on startup.
"""
from __future__ import annotations

import os
import pathlib
from typing import List, Tuple, Optional
import hashlib
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

from . import DATABASE_URL


class MigrationRunner:
    """Handles automatic database migration checking and application."""
    
    def __init__(self, migration_dir: Optional[pathlib.Path] = None):
        """Initialize the migration runner.
        
        Args:
            migration_dir: Directory containing migration SQL files. 
                         Defaults to scripts/ in project root.
        """
        if migration_dir is None:
            # Determine migration directory based on environment
            if os.path.exists("/app/sql"):
                # Docker environment
                self.migration_dir = pathlib.Path("/app/sql")
            else:
                # Local development - scripts directory relative to this file
                project_root = pathlib.Path(__file__).resolve().parent.parent.parent
                self.migration_dir = project_root / "scripts"
        else:
            self.migration_dir = migration_dir
            
        # Define migrations in order with their metadata
        self.migrations: List[Tuple[int, str, str]] = [
            (0, "000_migration_compatibility.sql", "Migration helper functions"),
            (1, "001_init_schema_postgres.sql", "Initial database schema"),
            (2, "002_migrate_to_profiles.sql", "Add research profiles feature"),
            (3, "003_profiles_trends_integration.sql", "Integrate profiles with trends"),
            (4, "004_add_staging_tables.sql", "Add staging tables for bulk operations"),
            (5, "005_optimize_indexes.sql", "Optimize indexes for performance"),
            (6, "006_add_processing_checkpoints.sql", "Add checkpoint system for resumable processing"),
            (7, "007_add_scheduled_tasks.sql", "Add scheduled tasks configuration"),
            (8, "008_add_multi_ollama_support.sql", "Add multi-Ollama server support for bulk judge operations"),
            (9, "009_add_lmstudio_multi_server.sql", "Add LMStudio multi-server support and rename to inference_servers"),
            (10, "010_add_per_server_model_config.sql", "Add per-server model name and config overrides for non-homogeneous deployments"),
            (11, "011_newsletter_multi_server.sql", "Add newsletter multi-server judge support"),
            (12, "012_timeline_indexes.sql", "Add indexes for research timeline queries"),
            (13, "013_profile_interest_metrics.sql", "Add profile-aware interest metrics tables"),
            (14, "014_interest_short_labels.sql", "Add short labels for profile interests"),
            (15, "015_profile_star_map.sql", "Add cached star map points per profile"),
            (16, "016_profile_star_map_3d.sql", "Add Z coordinate for 3D star map"),
            (17, "017_runtime_reliability.sql", "Durable dispatch, delivery receipts, diagnostics and search index"),
            (18, "018_newsletter_quality.sql", "Newsletter evidence artifacts and coverage history"),
        ]
    
    def _get_file_checksum(self, filepath: pathlib.Path) -> str:
        """Calculate MD5 checksum of a file."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _ensure_migration_table(self) -> None:
        """Ensure the schema_migrations table exists."""
        create_migration_sql = self.migration_dir / "create_migration_tracking.sql"
        
        if not create_migration_sql.exists():
            # Fallback: create the table inline if script not found
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT,
                            applied_at TIMESTAMPTZ DEFAULT NOW(),
                            checksum TEXT
                        );
                        CREATE INDEX IF NOT EXISTS idx_schema_migrations_name 
                        ON schema_migrations(name);
                    """)
                conn.commit()
        else:
            # Run the create migration tracking script
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    with open(create_migration_sql, 'r') as f:
                        cur.execute(f.read())
                conn.commit()
    
    def _is_migration_applied(self, migration_name: str) -> bool:
        """Check if a migration has already been applied."""
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) as count FROM schema_migrations WHERE name = %s",
                    (migration_name,)
                )
                result = cur.fetchone()
                return result['count'] > 0 if result else False
    
    def _apply_migration(self, version: int, filename: str, description: str) -> None:
        """Apply a single migration file."""
        migration_path = self.migration_dir / filename
        
        if not migration_path.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_path}")
        
        checksum = self._get_file_checksum(migration_path)
        
        print(f"[MIGRATION] Applying {filename}: {description}")
        
        # Apply the migration in a transaction
        with psycopg.connect(DATABASE_URL) as conn:
            try:
                with conn.cursor() as cur:
                    # Read and execute the migration file
                    with open(migration_path, 'r') as f:
                        migration_sql = f.read()
                    cur.execute(migration_sql)
                    
                    # Record the migration
                    cur.execute("""
                        INSERT INTO schema_migrations (version, name, description, checksum)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (version) DO NOTHING
                    """, (version, filename, description, checksum))
                
                conn.commit()
                print(f"[MIGRATION] ✓ Successfully applied {filename}")
                
            except Exception as e:
                conn.rollback()
                print(f"[MIGRATION] ✗ Failed to apply {filename}: {e}")
                raise
    
    def _verify_critical_tables(self) -> List[str]:
        """Verify that critical tables exist and return any missing tables."""
        critical_tables = [
            "papers",
            "research_profiles",
            "profile_research_interests",
            "paper_profile_scores",
            "topics",
            "settings",
            "model_providers",
            "scheduled_tasks",
            "scheduled_task_runs",
            "inference_servers",  # Renamed from ollama_servers in migration 009
            "judge_task_queue",
            "worker_heartbeats",
            "newsletter_jobs"  # Added in migration 011
        ]
        
        missing_tables = []
        
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                for table in critical_tables:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        ) as exists
                    """, (table,))
                    result = cur.fetchone()
                    if not result or not result['exists']:
                        missing_tables.append(table)
        
        return missing_tables
    
    def run_migrations(self) -> Tuple[int, int, List[str]]:
        """Check and apply all pending migrations.
        
        Returns:
            Tuple of (migrations_applied, migrations_skipped, issues)
        """
        print("[MIGRATION] Starting automatic migration check...")
        
        issues = []
        migrations_applied = 0
        migrations_skipped = 0
        
        try:
            # Ensure migration tracking table exists
            self._ensure_migration_table()
            
            # Check and apply each migration
            for version, filename, description in self.migrations:
                try:
                    if self._is_migration_applied(filename):
                        print(f"[MIGRATION] ✓ Already applied: {filename}")
                        migrations_skipped += 1
                    else:
                        self._apply_migration(version, filename, description)
                        migrations_applied += 1
                except FileNotFoundError as e:
                    issue = f"Migration file not found: {filename}"
                    print(f"[MIGRATION] ⚠ {issue}")
                    issues.append(issue)
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a constraint already exists error
                    if "already exists" in error_str.lower():
                        print(f"[MIGRATION] ⚠ Skipping {filename}: Constraint/index already exists")
                        # This is not a critical error, migration was partially applied before
                        migrations_skipped += 1
                    else:
                        issue = f"Failed to apply {filename}: {error_str}"
                        print(f"[MIGRATION] ✗ {issue}")
                        issues.append(issue)
                        # Stop on first real failure to maintain consistency
                        break
            
            # Verify critical tables exist
            missing_tables = self._verify_critical_tables()
            if missing_tables:
                issue = f"Missing critical tables after migration: {', '.join(missing_tables)}"
                print(f"[MIGRATION] ✗ {issue}")
                issues.append(issue)
            else:
                print("[MIGRATION] ✓ All critical tables verified")
            
            # Summary
            print(f"[MIGRATION] Summary: {migrations_applied} applied, {migrations_skipped} skipped")
            if issues:
                print(f"[MIGRATION] Issues encountered: {len(issues)}")
                for issue in issues:
                    print(f"[MIGRATION]   - {issue}")
            
        except Exception as e:
            issue = f"Migration system error: {str(e)}"
            print(f"[MIGRATION] ✗ {issue}")
            issues.append(issue)
        
        return migrations_applied, migrations_skipped, issues


async def check_and_apply_migrations() -> None:
    """Async wrapper to check and apply migrations during FastAPI startup.
    
    This function is designed to be called from the FastAPI lifespan context.
    """
    runner = MigrationRunner()
    import asyncio
    def migrate_locked():
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("SELECT pg_advisory_lock(hashtextextended('theseus:migrations', 0))")
            try:
                return runner.run_migrations()
            finally:
                conn.execute("SELECT pg_advisory_unlock(hashtextextended('theseus:migrations', 0))")
    applied, skipped, issues = await asyncio.to_thread(migrate_locked)
    
    if issues:
        raise RuntimeError(f"Database migration failed: {len(issues)} issue(s); inspect migration logs")
    
    # Post-migration checks and fixes
    _ensure_paper_profile_scores_constraint()
    
    print(f"[MIGRATION] Database migration check complete")


def _ensure_paper_profile_scores_constraint() -> None:
    """Ensure the unique constraint exists on paper_profile_scores.
    
    This is a defensive check added after discovering some databases
    had the table created without the constraint. This ensures
    ON CONFLICT clauses work correctly in bulk operations.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Check if constraint exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'paper_profile_scores_paper_id_profile_id_key'
                          AND conrelid = 'paper_profile_scores'::regclass
                    ) as exists
                """)
                result = cur.fetchone()
                
                if not result or not result[0]:
                    print("[MIGRATION] Adding missing unique constraint to paper_profile_scores...")
                    
                    # Remove any duplicate rows before adding constraint
                    cur.execute("""
                        DELETE FROM paper_profile_scores a USING (
                            SELECT MIN(id) as id, paper_id, profile_id
                            FROM paper_profile_scores 
                            GROUP BY paper_id, profile_id
                            HAVING COUNT(*) > 1
                        ) b
                        WHERE a.paper_id = b.paper_id 
                          AND a.profile_id = b.profile_id 
                          AND a.id != b.id
                    """)
                    
                    # Add the unique constraint
                    cur.execute("""
                        ALTER TABLE paper_profile_scores 
                        ADD CONSTRAINT paper_profile_scores_paper_id_profile_id_key 
                        UNIQUE (paper_id, profile_id)
                    """)
                    
                    conn.commit()
                    print("[MIGRATION] ✓ Added unique constraint to paper_profile_scores")
                else:
                    print("[MIGRATION] ✓ Unique constraint on paper_profile_scores verified")
                    
    except Exception as e:
        print(f"[MIGRATION] ⚠ Warning: Could not verify/add constraint: {e}")
        # Don't fail startup, just log the warning

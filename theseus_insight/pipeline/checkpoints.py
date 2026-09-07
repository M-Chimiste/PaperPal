"""Checkpoint persistence for the newsletter/profile/embedding pipelines.

FROZEN CONTRACT — extracted byte-compatible from TheseusInsight
(theseus_insight.py:640-751). Resumable runs written by older versions
must keep loading:

- File checkpoints: {checkpoint_dir}/{stage}_checkpoint.pkl, a pickle of
  {'data', 'timestamp', 'stage'} with an ISO timestamp string.
- DB checkpoints: asyncpg CheckpointManager, job_type
  "newsletter_generation", DataFrames encoded as
  {'dataframe': df.to_dict('records')}.
- save_async always dual-writes the file checkpoint as fallback.
"""
import datetime
import os
import pickle
import shutil
import json
import tempfile
from pathlib import Path
from ..observability import fingerprint

import pandas as pd

from ..data_processing.checkpoint_manager import CheckpointManager
from ..db import get_connection_pool


class CheckpointAdapter:
    """Owns file + database checkpointing for one pipeline run."""

    def __init__(self, checkpoint_dir: str, *, use_database_checkpoints: bool = False,
                 verbose: bool = False):
        self.checkpoint_dir = checkpoint_dir
        self.use_database_checkpoints = use_database_checkpoints
        self.verbose = verbose
        self.checkpoint_manager = None
        self.job_id = None

    async def init_db_job(self, config: dict) -> None:
        """Initialize the checkpoint manager and create the DB job."""
        directory = Path(self.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / 'manifest.json'
        from ..observability import provenance
        from ..api.tasks import record_async
        run_provenance = provenance(config)
        identity = fingerprint({'config': config, 'prompt_hash': run_provenance['prompt_hash']})
        if config.get('task_id'):
            await record_async(config['task_id'], 'provenance', 'recorded', details=run_provenance)
        previous = json.loads(manifest.read_text()) if manifest.exists() else None
        if previous and previous['config_hash'] != identity:
            raise ValueError('Checkpoint configuration changed; start a new run to avoid mixing results')
        if previous is None:
            if any(directory.glob('*_checkpoint.pkl')):
                raise ValueError('Unversioned checkpoints need explicit review; use a new checkpoint directory')
            self._atomic_write(manifest, json.dumps({'version': 1, 'config_hash': identity}).encode())
        if self.checkpoint_manager is None and self.use_database_checkpoints:
            pool = await get_connection_pool()
            self.checkpoint_manager = CheckpointManager(pool)
            if previous and previous.get('job_id'):
                from uuid import UUID
                self.job_id = UUID(previous['job_id'])
                async with pool.acquire() as conn:
                    if not await conn.fetchval('SELECT id FROM processing_jobs WHERE id=$1', self.job_id):
                        raise ValueError('Checkpoint job no longer exists; start a new run')
                    await conn.execute("UPDATE processing_jobs SET status='running', completed_at=NULL WHERE id=$1", self.job_id)
            else:
                self.job_id = await self.checkpoint_manager.create_job("newsletter_generation", config)
                self._atomic_write(manifest, json.dumps({'version': 1, 'config_hash': identity, 'job_id': str(self.job_id)}).encode())
            if self.verbose:
                print(f"📝 Created new newsletter job {self.job_id}")

    async def save_async(self, stage: str, data: any) -> None:
        """Save a checkpoint using the database checkpoint manager."""
        if self.checkpoint_manager and self.job_id:
            checkpoint_data = data
            if isinstance(data, pd.DataFrame):
                checkpoint_data = {'dataframe': data.to_dict('records')}

            await self.checkpoint_manager.save_checkpoint(
                self.job_id,
                stage,
                checkpoint_data,
                item_count=len(data) if hasattr(data, '__len__') else 1,
                update_state={"current_stage": stage}
            )
            if self.verbose:
                print(f"✅ Saved database checkpoint for stage: {stage}")

        # Always save file checkpoint as fallback
        self.save(stage, data)

    def save(self, stage: str, data: any) -> None:
        """Save a file-based checkpoint for the given pipeline stage."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{stage}_checkpoint.pkl")
        checkpoint_data = {
            'data': data,
            'timestamp': datetime.datetime.now().isoformat(),
            'stage': stage
        }
        self._atomic_write(Path(checkpoint_path), pickle.dumps(checkpoint_data))
        if self.verbose and not self.use_database_checkpoints:
            print(f"Saved file checkpoint for stage: {stage}")

    @staticmethod
    def _atomic_write(path, data):
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    async def load_async(self, stage: str) -> any:
        """Load a checkpoint using the database checkpoint manager."""
        if self.checkpoint_manager and self.job_id:
            checkpoint = await self.checkpoint_manager.get_latest_checkpoint(self.job_id, stage)
            if checkpoint:
                data = checkpoint['checkpoint_data']
                # Reconstruct DataFrame if needed
                if 'dataframe' in data:
                    data = pd.DataFrame(data['dataframe'])
                if self.verbose:
                    print(f"✅ Loaded database checkpoint for stage: {stage}")
                return data

        # Fall back to file checkpoint
        return self.load(stage)

    def load(self, stage: str) -> any:
        """Load a file-based checkpoint for the given pipeline stage."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{stage}_checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                if self.verbose and not self.use_database_checkpoints:
                    print(f"Loaded file checkpoint for stage: {stage} from {checkpoint['timestamp']}")
                return checkpoint['data']
            except Exception as e:
                if self.verbose:
                    print(f"Error loading file checkpoint for stage {stage}: {str(e)}")
                return None
        if self.verbose and not self.use_database_checkpoints:
            print(f"No file checkpoint found for stage: {stage}")
        return None

    async def cleanup_async(self) -> None:
        """Mark job as completed and clean up file checkpoints."""
        if self.checkpoint_manager and self.job_id:
            await self.checkpoint_manager.complete_job(self.job_id)
            if self.verbose:
                print("✅ Marked job as completed in database")
        self.cleanup()

    def cleanup(self) -> None:
        """Remove all checkpoint files after successful completion."""
        if os.path.exists(self.checkpoint_dir):
            try:
                shutil.rmtree(self.checkpoint_dir)
                if self.verbose and not self.use_database_checkpoints:
                    print("Cleaned up all file checkpoints")
            except Exception as e:
                if self.verbose:
                    print(f"Error cleaning up file checkpoints: {str(e)}")

    async def fail_job(self, error: str) -> None:
        """Mark the DB job failed (no-op when DB checkpoints are off)."""
        if self.checkpoint_manager and self.job_id:
            await self.checkpoint_manager.fail_job(self.job_id, error)

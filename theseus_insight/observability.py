"""Stage timing and reproducible provenance, without logging credentials."""
import hashlib
import json
import logging
import time
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def provenance(config):
    import os
    import subprocess
    root = Path(__file__).resolve().parent
    prompt_files = sorted((root / 'prompt').glob('*.py')) + [root / 'research_agent' / 'prompts.py']
    prompt_hash = fingerprint({str(p.relative_to(root)): p.read_text() for p in prompt_files})
    revision = os.getenv('BUILD_REVISION', 'unknown')
    dirty = None
    if (root.parent / '.git').exists():
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root.parent, text=True, timeout=2).strip()
            dirty = subprocess.run(['git', 'diff', '--quiet'], cwd=root.parent, timeout=2).returncode != 0
        except (OSError, subprocess.SubprocessError):
            pass
    return {'config_hash': fingerprint(config), 'prompt_hash': prompt_hash,
            'code_hash': fingerprint({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob('*.py'))}),
            'code_revision': revision, 'working_tree_modified': dirty}


async def run_stage(name, fn, ti, *args):
    from .api.tasks import record_async
    started = time.monotonic()
    await record_async(ti.task_id, name, 'started')
    try:
        result = await fn(ti, *args)
    except Exception as exc:
        await record_async(ti.task_id, name, 'failed', duration_ms=(time.monotonic()-started)*1000,
                           error_type=type(exc).__name__, details={'retry_stage': name})
        raise
    await record_async(ti.task_id, name, 'completed', duration_ms=(time.monotonic()-started)*1000)
    return result

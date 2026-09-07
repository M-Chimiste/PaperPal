"""Registry of background-task handlers (extracted from TaskManager, B6).

Each handler is `async (task_manager, task_id) -> None`. TaskManager keeps
thin `run_*_task` delegate methods so the bound-method references that
routers pass to enqueue_task keep working; HANDLERS maps task_type
strings to handlers for registry-based dispatch.
"""
from . import (
    bulk_embed, database_io, mindmap, newsletter, podcast,
    profile_ingest, visualizer,
)

from . import recoverable

HANDLERS = {
    "research": recoverable.research,
    "custom_newsletter": recoverable.custom_newsletter,
    "profile_newsletter": recoverable.profile_newsletter,
    "star-map": recoverable.star_map,
    "profile_interest": recoverable.profile_interest,
    "newsletter": newsletter.run,
    "podcast": podcast.run,
    "visualizer": visualizer.run,
    "database_export": database_io.run_export,
    "database_import": database_io.run_import,
    "mindmap_expand": mindmap.run_expand,
    "mindmap_pdf_parse": mindmap.run_pdf_parse,
    "profile_aware_ingest": profile_ingest.run,
    "bulk_embed": bulk_embed.run,
}

__all__ = ["HANDLERS"]

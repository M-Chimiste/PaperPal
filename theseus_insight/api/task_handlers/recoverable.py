"""Serializable entry points for previously closure-based newsletter tasks."""
import asyncio
import os
from ...data_access import TaskRepository


async def run_custom_newsletter_task(task_id):
    from ...services.newsletter_run_service import run_custom_newsletter
    from ..models import NewsletterRunParams
    task = await asyncio.to_thread(TaskRepository.get_task, task_id)
    await run_custom_newsletter(NewsletterRunParams(**task['config_json']), task_id, asyncio.get_running_loop())


async def run_profile_newsletter_task(task_id):
    from ..tasks import task_manager, TaskStatus
    from ._common import progress_callback
    from ...theseus_insight import TheseusInsight
    task = await asyncio.to_thread(TaskRepository.get_task, task_id)
    config = task['config_json']
    callback = progress_callback(task_manager, task_id)
    def execute():
        ti = TheseusInsight(
            research_interests_override=config['research_interests'],
            start_date_override=config['start_date'], end_date_override=config['end_date'],
            receiver_address_override=config['email_recipients'],
            profile_ids_override=[config['profile_id']], orchestration_config=config['orchestration_config'],
            generate_podcast=config['generate_podcast_run'], db_saving=True,
            task_id=task_id, checkpoint_dir=os.path.join('data', 'checkpoints', task_id),
        )
        return ti.run(progress_callback=callback)
    await task_manager.update_task_status(task_id, TaskStatus.PROCESSING, current_step='initializing')
    result = await asyncio.to_thread(execute)
    await task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result, current_step='newsletter_complete')


async def custom_newsletter(task_manager, task_id):
    await run_custom_newsletter_task(task_id)


async def profile_newsletter(task_manager, task_id):
    await run_profile_newsletter_task(task_id)


async def star_map(task_manager, task_id):
    from ...star_map.task import run_profile_star_map_task
    await run_profile_star_map_task(task_id)


async def profile_interest(task_manager, task_id):
    from ..routers.trends import run_profile_interest_task
    await run_profile_interest_task(task_id)


async def run_research_task(task_id):
    from ..routers import research_agent
    from datetime import datetime
    task = await asyncio.to_thread(TaskRepository.get_task, task_id)
    config = task['config_json']
    research_agent.research_tasks.setdefault(task_id, {
        'task_id': task_id, 'research_question': config['research_question'],
        'mode': config['mode'], 'status': 'pending', 'created_at': datetime.utcnow(),
        'progress': {},
    })
    run = research_agent._run_multi_agent_research_task if config['mode'] == 'multi' else research_agent._run_single_agent_research_task
    await run(task_id, config['research_question'], config['config'], config['save_to_library'])


async def research(task_manager, task_id):
    await run_research_task(task_id)

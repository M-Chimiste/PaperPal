from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Any, Dict, Optional
import os

from ..models import PaginatedResponse, Run
from ...data_access import NewsletterRepository, PodcastRepository, TaskRepository
from ..tasks import task_manager, TaskStatus

from ..models import StatusMessageResponse, ActiveTasksResponse, CompletedTasksResponse

router = APIRouter(tags=["runs-and-tasks"])

# Runs endpoints
@router.get("/api/runs", response_model=PaginatedResponse)
async def get_runs(
    page: int = Query(1, gt=0),
    sort_field: Optional[str] = None,
    sort_direction: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """
    Retrieves a paginated list of runs.

    This endpoint fetches a paginated list of runs from the database, allowing for filtering by date range and sorting options.
    It returns a list of runs with their details, including ID, date, pipeline type, status, duration, and artifact path.

    Args:
        page (int): The page number to fetch. Defaults to 1.
        sort_field (Optional[str]): The field to sort the runs by. Defaults to None.
        sort_direction (Optional[str]): The direction to sort the runs. Defaults to None.
        from_date (Optional[str]): The start date for filtering runs. Defaults to None.
        to_date (Optional[str]): The end date for filtering runs. Defaults to None.

    Returns:
        PaginatedResponse: A response object containing the list of runs, total items, total pages, current page, and the next page number if available.

    Raises:
        HTTPException: If an error occurs while fetching the runs.
    """
    try:
        # TODO: Implement proper pagination and filtering
        # For now, return all runs
        runs = []
        newsletters = NewsletterRepository.all()
        podcasts = PodcastRepository.all()
        
        # Convert newsletters to runs
        for n in newsletters:
            runs.append(Run(
                id=n['id'],
                date=n['date_sent'],
                pipeline_type='newsletter',
                status='completed',  # Assuming all stored ones are completed
                duration=0.0,  # TODO: Add duration tracking
                artifact_path=f"newsletters/{n['id']}/content.md"
            ))
            
        # Convert podcasts to runs
        for p in podcasts:
            runs.append(Run(
                id=p['id'],
                date=p['date'],
                pipeline_type='podcast',
                status='completed',
                duration=0.0,
                artifact_path=f"podcasts/{p['id']}/audio.mp3"
            ))
            
        # Sort runs by date descending
        runs.sort(key=lambda x: x.date, reverse=True)
        
        # Basic pagination
        page_size = 10
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_runs = runs[start_idx:end_idx]
        
        return PaginatedResponse(
            items=page_runs,
            nextPage=page + 1 if end_idx < len(runs) else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/runs/{run_id}/artifact", response_model=StatusMessageResponse)
async def delete_run_artifact(run_id: int):
    """
    Deletes the artifact associated with a run.

    This endpoint deletes the artifact associated with a run by its ID.
    It returns a success message if the artifact is deleted successfully.

    Args:
        run_id (int): The ID of the run to delete the artifact for.

    Returns:
        dict: A dictionary containing the status and a success message.

    Raises:
        HTTPException: If the run is not found or the artifact cannot be deleted.
    """
    try:
        # First determine if this is a newsletter or podcast run
        newsletters = NewsletterRepository.all()
        podcasts = PodcastRepository.all()
        
        # Check newsletters
        newsletter = next((n for n in newsletters if n['id'] == run_id), None)
        if newsletter:
            artifact_path = f"data/newsletters/{run_id}/content.md"
            if os.path.exists(artifact_path):
                os.remove(artifact_path)
                # Remove the directory if empty
                try:
                    os.rmdir(os.path.dirname(artifact_path))
                except OSError:
                    pass  # Directory not empty
            return {"status": "success"}
            
        # Check podcasts
        podcast = next((p for p in podcasts if p['id'] == run_id), None)
        if podcast:
            artifact_path = f"data/podcasts/{run_id}/audio.mp3"
            if os.path.exists(artifact_path):
                os.remove(artifact_path)
                # Remove the directory if empty
                try:
                    os.rmdir(os.path.dirname(artifact_path))
                except OSError:
                    pass  # Directory not empty
            return {"status": "success"}
            
        # If we get here, the run wasn't found
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Tasks endpoints
@router.get("/api/tasks/{task_id}/status", response_model=Dict[str, Any])
async def get_task_status(task_id: str):
    """
    Retrieves the current status of a task.

    This endpoint fetches the current status of a task by its ID.
    It returns the status of the task if found, otherwise raises an HTTP exception.

    Args:
        task_id (str): The ID of the task to retrieve the status for.

    Returns:
        dict: A dictionary containing the status of the task.
    
    Raises:
        HTTPException: If the task is not found.
    """
    try:
        status = task_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/tasks/active", response_model=ActiveTasksResponse)
async def get_active_tasks(task_types: Optional[str] = Query(None)):
    """
    Retrieves all active tasks, optionally filtered by type.

    This endpoint fetches all active tasks from the database, optionally filtered by task type.
    It returns a list of active tasks.

    Args:
        task_types (Optional[str]): The types of tasks to filter by. Defaults to None.

    Returns:
        dict: A dictionary containing the list of active tasks.
    """
    try:
        types_filter = task_types.split(',') if task_types else None
        active_tasks = TaskRepository.get_active_tasks(task_types=types_filter)
        return {"active_tasks": active_tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/tasks/recent-completed", response_model=CompletedTasksResponse)
async def get_recent_completed_tasks(task_types: Optional[str] = Query(None)):
    """
    Retrieves recent completed tasks with results available for download.

    This endpoint fetches recent completed tasks from the database, optionally filtered by task type.
    It returns a list of completed tasks.

    Args:
        task_types (Optional[str]): The types of tasks to filter by. Defaults to None.

    Returns:
        dict: A dictionary containing the list of completed tasks.
    
    Raises:
        HTTPException: If an error occurs while fetching the completed tasks.
    """
    try:
        types_filter = task_types.split(',') if task_types else None
        completed_tasks = TaskRepository.get_recent_completed_tasks(task_types=types_filter)
        return {"completed_tasks": completed_tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """
    Retrieves the result of a completed task.

    This endpoint fetches the result of a completed task by its ID.
    It returns the result of the task if found, otherwise raises an HTTP exception.

    Args:
        task_id (str): The ID of the task to retrieve the result for.

    Returns:
        dict: A dictionary containing the result of the task.
    
    Raises:
        HTTPException: If the task is not found or not completed.
    """
    try:
        task = task_manager.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
        if task["status"] != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Task is not completed (current status: {task['status']})"
            )
            
        if not task.get("result"):
            raise HTTPException(status_code=404, detail="No result available for this task")
            
        return task["result"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/tasks/{task_id}/abort", response_model=StatusMessageResponse)
async def abort_task(task_id: str):
    """
    Aborts a running task.

    This endpoint aborts a running task by its ID.
    It returns a success message if the task is aborted successfully.

    Args:
        task_id (str): The ID of the task to abort.

    Returns:
        dict: A dictionary containing the status and a success message.
    
    Raises:
        HTTPException: If the task is not found or cannot be aborted.
    """
    try:
        task = task_manager.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
        if task["status"] not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
            raise HTTPException(
                status_code=400,
                detail=f"Task cannot be aborted (current status: {task['status']})"
            )
            
        # Mark task as failed with abort message
        await task_manager.update_task_status(
            task_id,
            TaskStatus.FAILED,
            message="Task aborted by user",
            error="Task was manually aborted",
            current_step="aborted"
        )
        
        return {"status": "success", "message": f"Task {task_id} has been aborted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/tasks/{task_id}/download/{file_type}")
async def download_task_artifact(task_id: str, file_type: str):
    """
    Downloads a task artifact (newsletter or podcast).

    This endpoint downloads a task artifact by its ID and file type.
    It returns the artifact if found, otherwise raises an HTTP exception.

    Args:
        task_id (str): The ID of the task to download the artifact for.
        file_type (str): The type of file to download.

    Returns:
        FileResponse: A file response containing the artifact.
    
    Raises:
        HTTPException: If the task is not found, not completed, or the artifact is not found.
    """
    try:
        task = task_manager.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
        print(f"DEBUG: Download request for task {task_id}, type: {file_type}")
        print(f"DEBUG: Task status: {task.get('status')}")
        print(f"DEBUG: Task type: {task.get('type')}")
            
        if task["status"] != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Task is not completed (current status: {task['status']})"
            )
            
        result = task.get("result")
        if not result:
            print(f"DEBUG: No result found for task {task_id}")
            raise HTTPException(status_code=404, detail="No result available for this task")
            
        print(f"DEBUG: Task result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
        if task["type"] == "newsletter":
            if file_type != "markdown":
                raise HTTPException(status_code=400, detail="Only markdown format is available for newsletters")
                
            # Create a temporary markdown file
            output_path = f"data/temp/{task_id}/newsletter.md"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(result["newsletter_content"])
                
            return FileResponse(
                output_path,
                media_type="text/markdown",
                filename="newsletter.md"
            )
            
        elif task["type"] == "podcast":
            if file_type not in ["audio", "video"]:
                raise HTTPException(
                    status_code=400,
                    detail="Available formats for podcasts: audio, video"
                )
                
            # Get the appropriate file path
            if file_type == "audio":
                file_path = result.get("output_file")
                if not file_path:
                    print(f"DEBUG: No 'output_file' key in result for audio download")
                    raise HTTPException(
                        status_code=404,
                        detail="No audio file available in task result. Expected 'output_file' key."
                    )
                media_type = "audio/mpeg"
                filename = "podcast.mp3"
            else:  # video
                file_path = result.get("visualizer_file")
                if not file_path:
                    print(f"DEBUG: No 'visualizer_file' key in result for video download")
                    raise HTTPException(
                        status_code=404,
                        detail="No video visualization available for this podcast. Expected 'visualizer_file' key."
                    )
                media_type = "video/mp4"
                filename = "podcast.mp4"
                
            print(f"DEBUG: Attempting to serve file: {file_path}")
            if not os.path.exists(file_path):
                print(f"DEBUG: File does not exist at path: {file_path}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Artifact file not found: {file_path}"
                )
                
            return FileResponse(
                file_path,
                media_type=media_type,
                filename=filename
            )

        elif task["type"] == "visualizer": # Added block for visualizer
            if file_type != "video":
                raise HTTPException(
                    status_code=400,
                    detail="Only video format is available for visualizer tasks"
                )
            
            if not result.get("visualizer_file"):
                raise HTTPException(
                    status_code=404,
                    detail="No video visualization available for this visualizer task"
                )
            file_path = result["visualizer_file"]
            media_type = "video/mp4"
            filename = "visualization.mp4" # Or derive from task/result if needed

            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Artifact file not found: {file_path}"
                )
            
            return FileResponse(
                file_path,
                media_type=media_type,
                filename=filename
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task type: {task['type']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/tasks/{task_id}/diagnostics")
def task_diagnostics(task_id: str):
    from ...db import get_cursor
    with get_cursor() as cur:
        cur.execute("SELECT stage,status,duration_ms,error_type,details,created_at FROM task_stage_events WHERE task_id=%s ORDER BY id DESC LIMIT 200", (task_id,))
        events = cur.fetchall()
        cur.execute("SELECT status,updated_at FROM delivery_receipts WHERE task_id=%s", (task_id,))
        deliveries = cur.fetchall()
        cur.execute("SELECT status,current_step,message,error FROM tasks WHERE task_id=%s", (task_id,))
        task = cur.fetchone()
    return {"events": events, "deliveries": deliveries, "task": task}


@router.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str):
    import asyncio
    from ...data_access.runtime import DispatchRepository
    task = await asyncio.to_thread(TaskRepository.get_task, task_id)
    if not task or task['status'] != 'failed':
        raise HTTPException(409, 'Only failed tasks can be retried')
    try:
        await asyncio.to_thread(DispatchRepository.retry, task_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    await task_manager.start_worker()
    return {"task_id": task_id, "status": "pending"}


@router.post("/api/tasks/{task_id}/delivery-resolution")
def resolve_delivery(task_id: str, resolution: str = Query(..., pattern="^(sent|retry)$")):
    from ...db import get_cursor
    with get_cursor() as cur:
        cur.execute("SELECT status FROM tasks WHERE task_id=%s", (task_id,))
        task = cur.fetchone()
        if not task or task['status'] != 'failed':
            raise HTTPException(409, 'Resolve delivery only after the task has failed')
        cur.execute("UPDATE delivery_receipts SET status=%s,updated_at=now() WHERE task_id=%s AND status IN ('sending','uncertain') RETURNING delivery_key", (resolution, task_id))
        if not cur.fetchone():
            raise HTTPException(409, 'No uncertain delivery to resolve')
    return {"status": resolution}


@router.get("/api/runtime/provenance")
def runtime_provenance():
    from ...observability import provenance
    from ...data_access import SettingsRepository
    config = SettingsRepository.get_orchestration_config()
    return {**provenance(config), "model": config.get("embedding_model", {}).get("model_name", "unknown")}

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Optional, List, Dict, Any
from pydantic import ValidationError
import uuid
import os
import json
import asyncio
from datetime import datetime

from ..models import (
    NewsletterConfig, PodcastGenerationParams, NewsletterRunParams,
    PodcastListItemResponse, PodcastDetailResponse
)
from ...data_access import (
    NewsletterRepository,
    PodcastRepository,
    TaskRepository,
)
from ..tasks import task_manager
from ..helpers.serialization import isoformat_fields
from ...services import newsletter_run_service

router = APIRouter(tags=["newsletters-and-podcasts"])

# Newsletter endpoints
@router.post("/api/newsletter/run")
async def run_newsletter(
    background_tasks: BackgroundTasks,
    config: Optional[str] = Form(None),
    intro_music_file: Optional[UploadFile] = File(None)
):
    """
    Starts the newsletter generation pipeline.

    This endpoint accepts a newsletter configuration and an optional intro music file.
    It creates a new task for generating a newsletter based on the provided configuration.
    """
    try:
        # Parse config
        if not config:
            raise HTTPException(status_code=400, detail="Newsletter configuration is required")
        
        newsletter_config = NewsletterConfig.model_validate_json(config)
        task_id = str(uuid.uuid4())
        
        # Validate topic_id if provided
        if newsletter_config.topic_id is not None:
            from ...data_access import TopicsRepository
            topic_data = TopicsRepository.get(newsletter_config.topic_id)
            if not topic_data:
                raise HTTPException(status_code=404, detail=f"Topic {newsletter_config.topic_id} not found")
        
        # Save intro music file if provided
        if intro_music_file:
            file_path = f"data/temp/{task_id}/intro_music{os.path.splitext(intro_music_file.filename)[1]}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(await intro_music_file.read())
            newsletter_config.dict()["intro_music_path"] = file_path
        
        # Create and start task
        await task_manager.create_task(
            task_id=task_id,
            task_type="newsletter",
            config=newsletter_config.dict()
        )
        await task_manager.enqueue_task(task_manager.run_newsletter_task, task_id)
        
        return {"taskId": task_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/actions/run-newsletter-pipeline", response_model=Dict[str, str])
async def run_newsletter_pipeline_endpoint(
    params: NewsletterRunParams,
    background_tasks: BackgroundTasks
):
    """
    Initiates the newsletter pipeline run process.

    This endpoint accepts parameters for running the newsletter pipeline and initiates a background task
    to execute the pipeline. It returns a task ID for tracking the progress of the pipeline run.

    Args:
        params (NewsletterRunParams): Parameters for running the newsletter pipeline.
        background_tasks (BackgroundTasks): An instance of BackgroundTasks for managing background tasks.

    Returns:
        dict: A dictionary containing the task ID for tracking the pipeline run progress.
    """
    # Validate topic_id if provided
    if params.topic_id is not None:
        from ...data_access import TopicsRepository
        topic_data = TopicsRepository.get(params.topic_id)
        if not topic_data:
            raise HTTPException(status_code=404, detail=f"Topic {params.topic_id} not found")
    
    # Validate profile parameters if provided
    if params.profile_id or params.profile_ids or params.profile_tag or params.profile_tags:
        from ...data_access import ProfileRepository
        
        # Validate specific profile IDs exist
        if params.profile_id:
            profile = ProfileRepository.get_by_id(params.profile_id)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile {params.profile_id} not found")
        
        if params.profile_ids:
            for profile_id in params.profile_ids:
                profile = ProfileRepository.get_by_id(profile_id)
                if not profile:
                    raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        # Validate tags exist (optional - tags that don't exist will simply return no profiles)
        if params.profile_tag or params.profile_tags:
            tag_list = []
            if params.profile_tag:
                tag_list.append(params.profile_tag)
            if params.profile_tags:
                tag_list.extend(params.profile_tags)
            
            # Check if any profiles exist with these tags
            profiles_with_tags = ProfileRepository.get_by_tags(tag_list)
            if not profiles_with_tags:
                raise HTTPException(status_code=404, detail=f"No profiles found with tags: {tag_list}")
    
    task_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()

    # Business logic lives in services/newsletter_run_service.py (B7);
    # this endpoint keeps only validation (above) and enqueueing.
    await task_manager.create_task(task_id, "custom_newsletter", params.model_dump(mode="json"))
    from ..task_handlers.recoverable import run_custom_newsletter_task
    await task_manager.enqueue_task(run_custom_newsletter_task, task_id)
    return {"task_id": task_id, "message": "Newsletter generation process has been initiated."}

# Podcast endpoints
@router.post("/api/podcast/generate")
async def generate_podcast_pipeline(
    background_tasks: BackgroundTasks,
    params_json: str = Form(..., description="JSON string of PodcastGenerationParams"),
    intro_music_file: Optional[UploadFile] = File(None),
    pdf_files: Optional[List[UploadFile]] = File(None, description="List of PDF files if input_type is 'pdfs'")
):
    """
    Initiates the podcast generation pipeline.

    This endpoint accepts parameters for generating a podcast, including an optional intro music file and PDF files.
    It creates a new task for generating the podcast based on the provided parameters, saves any uploaded files to a temporary directory,
    and enqueues the task for processing.

    Args:
        background_tasks (BackgroundTasks): An instance of BackgroundTasks for managing background tasks.
        params_json (str): A JSON string representing the podcast generation parameters.
        intro_music_file (Optional[UploadFile]): The intro music file to be used in the podcast.
        pdf_files (Optional[List[UploadFile]]): A list of PDF files to be used in the podcast, if applicable.

    Returns:
        dict: A dictionary containing the task ID and a success message.
    """
    try:
        generation_params = PodcastGenerationParams.model_validate_json(params_json)
        task_id = str(uuid.uuid4())

        # This will be the main config dictionary passed to the task manager
        # It will be used by the background task to instantiate and run PodcastGenerator
        task_config = {
            "input_type": generation_params.input_type,
            "podcast_model_config": generation_params.podcast_model_config.dict(),
            "tts_model_config": generation_params.tts_model_config.dict(),
            "create_visualization": generation_params.create_visualization,
            "db_saving": True, # Default, can be made configurable if needed
            "data_path": os.getenv("DATABASE_URL", "postgresql://theseus:theseus@localhost:5432/theseusdb"), # Global DB URL
            "verbose": True, # Default, can be made configurable
            "output_dir_base": "data/podcasts", # Base directory for task outputs
            "task_id": task_id # Pass task_id for organizing outputs
        }

        if generation_params.urls:
            task_config["urls"] = generation_params.urls
        
        # Handle uploaded intro music file
        if intro_music_file:
            temp_dir = f"data/temp/{task_id}"
            os.makedirs(temp_dir, exist_ok=True)
            intro_music_path = os.path.join(temp_dir, f"intro_{intro_music_file.filename}")
            with open(intro_music_path, "wb") as f:
                f.write(await intro_music_file.read())
            task_config["intro_music_path"] = intro_music_path

        # Handle uploaded PDF files
        if generation_params.input_type == "pdfs" and pdf_files:
            saved_pdf_paths = []
            pdf_temp_dir = f"data/temp/{task_id}/uploaded_pdfs"
            os.makedirs(pdf_temp_dir, exist_ok=True)
            for i, pdf_file in enumerate(pdf_files):
                # Sanitize filename or use a unique name
                safe_filename = f"doc_{i}_{os.path.basename(pdf_file.filename or f'file{i}.pdf')}"
                pdf_path = os.path.join(pdf_temp_dir, safe_filename)
                with open(pdf_path, "wb") as f:
                    f.write(await pdf_file.read())
                saved_pdf_paths.append(pdf_path)
            task_config["input_pdf_paths"] = saved_pdf_paths
        elif generation_params.input_type == "pdfs" and not pdf_files:
            raise HTTPException(status_code=400, detail="PDF files are required when input_type is 'pdfs'.")


        if generation_params.create_visualization and generation_params.visualizer_params:
            vis_p = generation_params.visualizer_params
            task_config.update({
                "visualizer_settings": vis_p.dict() # Pass the whole dict
            })
        else:
            task_config["visualizer_settings"] = None


        # Create task with the comprehensive config
        await task_manager.create_task(
            task_id=task_id,
            task_type="podcast", # Using existing "podcast" type, assuming run_podcast_task can handle new config
            config=task_config 
        )
        
        # The task_manager.run_podcast_task needs to be able to:
        # 1. Instantiate PodcastGenerator with relevant parts of task_config
        #    (podcast_model_config, tts_model_config, intro_music_path, etc.)
        # 2. Call PodcastGenerator.generate_podcast() with input_pdf_paths or processed URLs,
        #    output directory derived from task_id, and visualizer settings.
        # 3. Handle URL fetching and conversion to PDF if input_type is 'urls' (this is complex).
        #    For now, this implementation primarily supports 'pdfs' directly for PodcastGenerator.
        #    If 'urls' are passed, 'run_podcast_task' needs to manage downloading/converting them to PDF paths.
        await task_manager.enqueue_task(task_manager.run_podcast_task, task_id)
        
        return {"task_id": task_id, "message": "Podcast generation process initiated."}
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for params_json.")
    except ValueError as e: # Handles Pydantic validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException: # Re-raise existing HTTPExceptions
        raise
    except Exception as e:
        print(f"Error in /api/podcast/generate: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error processing podcast request: {str(e)}")

def _convert_podcast_timestamps(podcast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert PostgreSQL datetime objects to ISO format strings for API responses.

    The podcasts.date column is DATE, so isoformat() equals the old
    strftime('%Y-%m-%d') output.
    """
    return isoformat_fields(podcast_data, ('date',))

@router.get("/api/podcasts/history", response_model=List[PodcastListItemResponse])
async def get_podcast_history_list():
    """
    Retrieves a list of podcast history items.

    This endpoint fetches a list of podcast history items from the database.
    It returns a list of podcast items with their ID, title, date, and a description snippet.
    The list is sorted in descending order by date, with the most recent podcasts first.

    Returns:
        List[PodcastListItemResponse]: A list of podcast history items.
    """
    try:
        podcasts_data = PodcastRepository.all()
        
        response_items = []
        for p_data in podcasts_data:
            # Convert PostgreSQL datetime objects to strings
            converted_data = _convert_podcast_timestamps(p_data)
            
            description_snippet = (converted_data['description'][:150] + '...') if len(converted_data['description']) > 150 else converted_data['description']
            response_items.append(
                PodcastListItemResponse(
                    id=converted_data['id'],
                    title=converted_data['title'],
                    date=converted_data['date'],
                    description_snippet=description_snippet
                )
            )
        # To strictly sort by date if IDs don't guarantee it:
        response_items.sort(key=lambda x: datetime.strptime(x.date, '%Y-%m-%d'), reverse=True)
        return response_items
    except Exception as e:
        print(f"Error fetching podcast history list: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while fetching podcast history.")

@router.get("/api/podcasts/history/{podcast_id}", response_model=PodcastDetailResponse)
async def get_podcast_detail(podcast_id: int):
    """
    Retrieves the details of a podcast by its ID.

    This endpoint fetches the details of a podcast from the database by its ID.
    It returns a podcast detail object with ID, title, date, description, and script.

    Args:
        podcast_id (int): The ID of the podcast to retrieve.

    Returns:
        PodcastDetailResponse: A podcast detail object.
    """
    try:
        podcast_data = PodcastRepository.get(podcast_id)
        if not podcast_data:
            raise HTTPException(status_code=404, detail=f"Podcast with ID {podcast_id} not found.")
        
        # Convert PostgreSQL datetime objects to strings
        converted_data = _convert_podcast_timestamps(podcast_data)
        
        # The script from db.fetch_podcast_by_id is already a Python list of dicts
        # Pydantic will validate it against List[PodcastScriptItem]
        return PodcastDetailResponse(
            id=converted_data['id'],
            title=converted_data['title'],
            date=converted_data['date'],
            description=converted_data['description'],
            script=converted_data['script'] # Pydantic validation happens here
        )
    except HTTPException: # Re-raise HTTPException directly
        raise
    except ValidationError as ve: # Catch Pydantic validation errors specifically for the script
        print(f"Validation error for podcast script (ID: {podcast_id}): {ve}")
        raise HTTPException(status_code=500, detail=f"Error validating podcast script data for podcast ID {podcast_id}.")
    except Exception as e:
        print(f"Error fetching podcast detail (ID: {podcast_id}): {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while fetching details for podcast ID {podcast_id}.")

@router.delete("/api/podcasts/history/{podcast_id}")
async def delete_podcast(podcast_id: int):
    
    try:
        # Check if podcast exists first
        podcast_data = PodcastRepository.get(podcast_id)
        if not podcast_data:
            raise HTTPException(status_code=404, detail=f"Podcast with ID {podcast_id} not found.")
        
        # Delete the podcast
        PodcastRepository.delete(podcast_id)
        
        return {"status": "success", "message": f"Podcast with ID {podcast_id} has been deleted successfully."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting podcast (ID: {podcast_id}): {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while deleting podcast ID {podcast_id}.")

@router.put("/api/podcasts/history/{podcast_id}/title")
async def update_podcast_title(podcast_id: int, title_data: dict):
    """Update the title of a podcast by its ID."""
    try:
        # Validate request body
        if "title" not in title_data:
            raise HTTPException(status_code=400, detail="Title field is required.")
        
        new_title = title_data["title"].strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty.")
        
        # Check if podcast exists first
        podcast_data = PodcastRepository.get(podcast_id)
        if not podcast_data:
            raise HTTPException(status_code=404, detail=f"Podcast with ID {podcast_id} not found.")
        
        # Update the podcast title
        PodcastRepository.update_title(podcast_id, new_title)
        
        return {"status": "success", "message": f"Podcast title updated successfully.", "title": new_title}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating podcast title (ID: {podcast_id}): {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while updating podcast title for ID {podcast_id}.")

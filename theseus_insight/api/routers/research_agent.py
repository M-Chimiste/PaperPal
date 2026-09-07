"""
FastAPI Router for Research Agent

Provides REST API endpoints for research agent functionality including
task management, status tracking, result retrieval, and history management.
Enhanced to support both single-agent and multi-agent modes.
"""

import asyncio
import logging
import uuid
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Literal
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ...research_agent.workflow import ResearchAgentWorkflow, create_research_workflow
from ...research_agent.tools import UnifiedSearchTool, LocalSearchTool, ExternalSearchTool, SearchConfig
from ...research_agent.model_router import get_embedding_model
from ...research_agent.orchestrator import MultiAgentOrchestrator, OrchestrationConfig, OrchestrationResult
from ..models import ResearchAgentModelConfigApi
from ...data_access import (
    ResearchRunRepository, ResearchAgentStateRepository, 
    TaskRepository, SettingsRepository
)
from ...data_model.papers import Paper
import os


# Pydantic models for API requests/responses
class ResearchTaskRequest(BaseModel):
    """Request model for starting a research task."""
    research_question: str = Field(..., description="The research question to investigate")
    mode: Optional[Literal["single", "multi"]] = Field(None, description="Research agent mode ('single' or 'multi'). If not specified, uses current system setting.")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration overrides")
    save_to_library: bool = Field(True, description="Whether to save results to research library")
    
    class Config:
        json_schema_extra = {
            "example": {
                "research_question": "What are the latest developments in quantum computing for machine learning?",
                "mode": "multi",
                "config": {
                    "search_config": {
                        "local_limit": 20,
                        "external_limit": 15
                    },
                    "synthesis_config": {
                        "strategy": "weighted_consensus"
                    }
                },
                "save_to_library": True
            }
        }


class ResearchTaskResponse(BaseModel):
    """Response model for research task creation."""
    task_id: str = Field(..., description="Unique identifier for the research task")
    status: str = Field(..., description="Current status of the task")
    created_at: str = Field(..., description="Task creation timestamp")
    research_question: str = Field(..., description="The research question being investigated")
    mode: str = Field(..., description="Research agent mode used ('single' or 'multi')")


class ResearchTaskStatus(BaseModel):
    """Response model for research task status."""
    task_id: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    mode: Optional[str] = None  # "single" or "multi"
    progress: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ResearchTaskResult(BaseModel):
    """Response model for research task results."""
    task_id: str
    status: str
    research_question: str
    mode: str  # "single" or "multi"
    final_answer: Optional[str] = None
    generation_summary: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    sub_queries: List[str] = Field(default_factory=list)
    sources_gathered: List[Dict[str, Any]] = Field(default_factory=list)
    judged_sources: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    compressed_notes: str = ""
    workflow_messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ResearchHistoryItem(BaseModel):
    """Response model for research history items."""
    task_id: str
    research_question: str
    status: str
    mode: Optional[str] = None  # "single" or "multi"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None


class ResearchHistoryResponse(BaseModel):
    """Response model for paginated research history."""
    items: List[ResearchHistoryItem]
    total: int
    limit: int
    offset: int


class ResearchModeRequest(BaseModel):
    """Request model for switching research agent mode."""
    mode: Literal["single", "multi"] = Field(..., description="New research agent mode")


class ResearchModeResponse(BaseModel):
    """Response model for research mode operations."""
    current_mode: str
    validation: Dict[str, Any]
    success: bool
    message: str


class ResearchConfigResponse(BaseModel):
    """Response model for research configuration."""
    current_mode: str
    single_agent_config: Dict[str, Any]
    multi_agent_config: Dict[str, Any]
    validation: Dict[str, Any]
    available_modes: List[str] = Field(default=["single", "multi"])


# Router setup
router = APIRouter(prefix="/api/research-agent", tags=["research-agent"])
logger = logging.getLogger(__name__)

def _convert_datetime_to_string(value):
    """Convert datetime/date objects to ISO format strings."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value

# _serialize_paper_info moved to api/helpers/serialization.py (B7);
# alias keeps the many internal call sites unchanged.
from ..helpers.serialization import serialize_research_object as _serialize_paper_info

# Global dictionaries for tracking research tasks and results
# These are used by the workflow and websockets for real-time progress updates
research_tasks: Dict[str, Dict[str, Any]] = {}
task_results: Dict[str, Dict[str, Any]] = {}




def _mark_orphaned_research_tasks_as_failed():
    """
    Mark all pending/running research tasks as failed on startup.
    This handles cases where the server crashed while tasks were running.
    """
    try:
        current_time = datetime.utcnow().isoformat()
        
        # Get all orphaned research tasks (pending or running)
        orphaned_runs = ResearchRunRepository.get_research_runs_by_status(['pending', 'running'])
        
        if orphaned_runs:
            logger.info(f"Found {len(orphaned_runs)} orphaned research tasks, marking as failed")
            
            for run in orphaned_runs:
                task_id = run['task_id']
                
                # Update research run status
                ResearchRunRepository.update_research_run_status(
                    task_id=task_id,
                    status="failed",
                    completed_at=current_time,
                    error_message="Task was interrupted by server restart"
                )
                
                # Update general task status as well
                TaskRepository.update_task_status(
                    task_id=task_id,
                    status="failed",
                    progress=0.0,
                    current_step="interrupted",
                    message="Task failed due to server restart",
                    error="Task was interrupted by server restart",
                    end_time=current_time
                )
                
                logger.info(f"Marked orphaned research task {task_id} as failed")
    except Exception as e:
        logger.error(f"Error marking orphaned research tasks as failed: {e}")


# Mark orphaned tasks as failed on startup
# Recovery is explicit; importing a router must never mutate active runs.


async def get_research_workflow() -> ResearchAgentWorkflow:
    """
    Dependency to create and configure a single-agent research workflow.
    
    Returns:
        Configured ResearchAgentWorkflow instance
    """
    try:
        # Get embedding model configuration from database
        orchestration_config_json = SettingsRepository.get("orchestration")
        if orchestration_config_json:
            import json
            orchestration_config = json.loads(orchestration_config_json)
        else:
            # Default configuration with embedding model
            orchestration_config = {
                'embedding_model': {
                    'model_type': 'sentence-transformer',
                    'model_name': 'Alibaba-NLP/gte-modernbert-base',
                    'trust_remote_code': True
                }
            }
        
        # Create embedding model using the model router
        embedding_model = get_embedding_model(orchestration_config)
        
        # Create search tools (LocalSearchTool uses repositories now)
        local_search_tool = LocalSearchTool(embedding_model)
        external_search_tool = ExternalSearchTool()
        
        # Create unified search tool without search_config parameter
        unified_search_tool = UnifiedSearchTool(
            local_search_tool=local_search_tool,
            external_search_tool=external_search_tool
        )
        
        # Use research agent model configuration from orchestration config
        research_agent_config = orchestration_config.get("research_agent_model_config", {})
        
        if research_agent_config and research_agent_config.get("boss_model"):
            # Use the research agent model configuration - ONLY use configured models
            boss_model = research_agent_config.get("boss_model", {})
            
            # Node-specific models default to boss_model if not specified
            query_planner_model = research_agent_config.get("query_planner_model") or boss_model
            evidence_selector_model = research_agent_config.get("evidence_selector_model") or boss_model  
            compression_model = research_agent_config.get("compression_model") or boss_model
            answer_generator_model = research_agent_config.get("answer_generator_model") or boss_model
            
            config = {
                "research_agent_model_config": research_agent_config,
                "query_planner_model": query_planner_model,
                "evidence_selector_model": evidence_selector_model, 
                "compression_model": compression_model,
                "answer_generator_model": answer_generator_model,
                "default_model": boss_model,  # Fallback for any other nodes
                
                # Also add individual node configs for the model router
                "query_planner": query_planner_model,
                "evidence_selector": evidence_selector_model,
                "scratchpad_compress": compression_model,
                "answer_generator": answer_generator_model
            }
            logger.info(f"Using research agent configuration with boss model: {boss_model.get('model_name', 'unknown')} ({boss_model.get('model_type', 'unknown')})")
        else:
            # NO HARDCODED FALLBACK - Require proper configuration
            logger.error("No research agent configuration found in database settings. Please configure models in Settings.tsx")
            raise HTTPException(
                status_code=500, 
                detail="Research agent not configured. Please configure research agent models in Settings → Research Agent Configuration"
            )
        
        # Create workflow with configuration
        workflow = create_research_workflow(
            config=config,
            unified_search_tool=unified_search_tool,
            max_research_loops=research_agent_config.get("max_research_loops", 3),
            max_research_context_tokens=research_agent_config.get("max_research_context_tokens", 15000),
            compress_to_ratio=research_agent_config.get("compress_to_ratio", 0.2)
        )
        
        return workflow
        
    except Exception as e:
        logger.error(f"Error creating research workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize research workflow: {str(e)}")


async def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """
    Dependency to create and configure a multi-agent orchestrator.
    
    Returns:
        Configured MultiAgentOrchestrator instance
    """
    try:
        # Ensure dual-mode configuration exists
        SettingsRepository.ensure_dual_mode_config()
        
        # Get orchestration configuration
        orchestration_config = SettingsRepository.get_orchestration_config()
        
        # Create embedding model
        embedding_model = get_embedding_model(orchestration_config)
        
        # Create search tools
        local_search_tool = LocalSearchTool(embedding_model)
        external_search_tool = ExternalSearchTool()
        unified_search_tool = UnifiedSearchTool(
            local_search_tool=local_search_tool,
            external_search_tool=external_search_tool
        )
        
        # Get multi-agent configuration
        multi_agent_config = SettingsRepository.get_multi_agent_config()
        
        if not multi_agent_config or not multi_agent_config.get("boss_model"):
            logger.error("No multi-agent configuration found. Please configure multi-agent models in Settings.")
            raise HTTPException(
                status_code=500,
                detail="Multi-agent not configured. Please configure multi-agent models in Settings → Research Agent Configuration"
            )
        
        # Create orchestration configuration
        orchestration_config_obj = OrchestrationConfig(
            parallel_agents=multi_agent_config.get("parallel_agents", 4),
            task_timeout=multi_agent_config.get("task_timeout", 300),
            max_concurrent_agents=min(multi_agent_config.get("parallel_agents", 4), 6),
            synthesis_config=multi_agent_config.get("synthesis_config", {})
        )
        
        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(
            model_config=orchestration_config,
            search_tool=unified_search_tool,
            config=orchestration_config_obj
        )
        
        logger.info(f"Created multi-agent orchestrator with {orchestration_config_obj.parallel_agents} agents")
        return orchestrator
        
    except Exception as e:
        logger.error(f"Error creating multi-agent orchestrator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize multi-agent orchestrator: {str(e)}")


def get_effective_mode(request_mode: Optional[str] = None) -> str:
    """
    Get the effective research agent mode for a request.
    
    Args:
        request_mode: Mode specified in the request (optional)
        
    Returns:
        Effective mode to use ("single" or "multi")
    """
    if request_mode and request_mode in ["single", "multi"]:
        return request_mode
    
    # Use system default
    return SettingsRepository.get_research_agent_mode()


@router.post("/run", response_model=ResearchTaskResponse)
async def start_research_task(
    request: ResearchTaskRequest,
    background_tasks: BackgroundTasks
) -> ResearchTaskResponse:
    """
    Start a new research task using either single-agent or multi-agent mode.
    
    Args:
        request: Research task request with question, mode, and configuration
        background_tasks: FastAPI background tasks for async execution
        
    Returns:
        Research task response with task ID and status
    """
    try:
        # Determine effective mode
        effective_mode = get_effective_mode(request.mode)
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Initialize task in global dictionaries for WebSocket tracking
        research_tasks[task_id] = {
            "task_id": task_id,
            "research_question": request.research_question,
            "mode": effective_mode,
            "status": "pending",
            "created_at": created_at,
            "progress": {"status": "Task created and queued for execution"}
        }
        
        # Create specific research run entry for Research Library
        ResearchRunRepository.insert_research_run(
            task_id=task_id,
            research_question=request.research_question,
            status="pending",
            config={"mode": effective_mode, "config": request.config or {}},
            save_to_library=request.save_to_library
        )

        # Create task in general tasks table for job history
        TaskRepository.insert_task(
            task_id=task_id,
            task_type=f"research-agent-{effective_mode}",
            status="pending",
            config_json={"mode": effective_mode, "config": request.config or {}, "research_question": request.research_question, "save_to_library": request.save_to_library},
            start_time=created_at.isoformat(),
            progress=0.0,
            current_step="Initializing research workflow",
            message=f"Research task created for {effective_mode}-agent mode"
        )

        from ..tasks import task_manager
        from ..task_handlers.recoverable import run_research_task
        await task_manager.enqueue_task(run_research_task, task_id)
        
        logger.info(f"Started {effective_mode}-agent research task {task_id} for question: {request.research_question[:100]}")
        
        return ResearchTaskResponse(
            task_id=task_id,
            status="pending",
            created_at=_convert_datetime_to_string(created_at),
            research_question=request.research_question,
            mode=effective_mode
        )
        
    except Exception as e:
        logger.error(f"Error starting research task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start research task: {str(e)}")


# New endpoints for dual-mode configuration management

@router.get("/modes", response_model=ResearchConfigResponse)
async def get_research_modes() -> ResearchConfigResponse:
    """
    Get current research agent mode configuration and available modes.
    
    Returns:
        Current mode configuration and validation status
    """
    try:
        # Ensure dual-mode configuration exists
        SettingsRepository.ensure_dual_mode_config()
        
        current_mode = SettingsRepository.get_research_agent_mode()
        single_config = SettingsRepository.get_single_agent_config()
        multi_config = SettingsRepository.get_multi_agent_config()
        validation = SettingsRepository.validate_research_agent_config()
        
        return ResearchConfigResponse(
            current_mode=current_mode,
            single_agent_config=single_config,
            multi_agent_config=multi_config,
            validation=validation,
            available_modes=["single", "multi"]
        )
        
    except Exception as e:
        logger.error(f"Error getting research modes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get research modes: {str(e)}")


@router.put("/mode", response_model=ResearchModeResponse)
async def set_research_mode(request: ResearchModeRequest) -> ResearchModeResponse:
    """
    Switch research agent mode between single and multi-agent.
    
    Args:
        request: New mode to switch to
        
    Returns:
        Mode switch results and validation status
    """
    try:
        validation_results = SettingsRepository.switch_research_agent_mode(request.mode)
        
        return ResearchModeResponse(
            current_mode=request.mode,
            validation=validation_results,
            success=validation_results["valid"],
            message=f"Successfully switched to {request.mode}-agent mode" if validation_results["valid"] 
                   else f"Switched to {request.mode}-agent mode but configuration has issues: {', '.join(validation_results['issues'])}"
        )
        
    except Exception as e:
        logger.error(f"Error setting research mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set research mode: {str(e)}")


@router.get("/config/{mode}")
async def get_mode_config(mode: Literal["single", "multi"]) -> Dict[str, Any]:
    """
    Get configuration for a specific research agent mode.
    
    Args:
        mode: Mode to get configuration for
        
    Returns:
        Configuration dictionary for the specified mode
    """
    try:
        if mode == "single":
            return SettingsRepository.get_single_agent_config()
        elif mode == "multi":
            return SettingsRepository.get_multi_agent_config()
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'single' or 'multi'")
            
    except Exception as e:
        logger.error(f"Error getting {mode} config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get {mode} configuration: {str(e)}")


@router.put("/config/{mode}")
async def set_mode_config(mode: Literal["single", "multi"], config: Dict[str, Any]) -> Dict[str, str]:
    """
    Set configuration for a specific research agent mode.
    
    Args:
        mode: Mode to set configuration for
        config: Configuration dictionary
        
    Returns:
        Success confirmation
    """
    try:
        if mode == "single":
            SettingsRepository.set_single_agent_config(config)
        elif mode == "multi":
            SettingsRepository.set_multi_agent_config(config)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'single' or 'multi'")
        
        return {"message": f"Successfully updated {mode}-agent configuration"}
        
    except Exception as e:
        logger.error(f"Error setting {mode} config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set {mode} configuration: {str(e)}")


@router.get("/status/{task_id}", response_model=ResearchTaskStatus)
async def get_research_task_status(task_id: str) -> ResearchTaskStatus:
    """
    Get the status of a research task.
    
    Args:
        task_id: Unique identifier for the research task
        
    Returns:
        Current status of the research task
    """
    # Try to get from research runs first (more detailed)
    research_run = ResearchRunRepository.get_research_run(task_id)
    if research_run:
        # Extract mode from config if available
        config = research_run.get("config", {})
        mode = config.get("mode") if isinstance(config, dict) else None
        
        return ResearchTaskStatus(
            task_id=task_id,
            status=research_run["status"],
            mode=mode,
            progress=research_run.get("progress"),
            created_at=_convert_datetime_to_string(research_run["created_at"]),
            started_at=_convert_datetime_to_string(research_run.get("started_at")),
            completed_at=_convert_datetime_to_string(research_run.get("completed_at")),
            error_message=research_run.get("error_message")
        )
    
    # Fallback to general task table
    task = TaskRepository.get_task(task_id)
    if task:
        # Extract mode from config if available
        config = task.get("config_json", {})
        mode = config.get("mode") if isinstance(config, dict) else None
        
        return ResearchTaskStatus(
            task_id=task_id,
            status=task["status"],
            mode=mode,
            progress={"progress": task.get("progress", 0)},
            created_at=_convert_datetime_to_string(task["start_time"]),
            started_at=_convert_datetime_to_string(task["start_time"]) if task["status"] != "pending" else None,
            completed_at=_convert_datetime_to_string(task.get("end_time")),
            error_message=task.get("error")
        )
    
    raise HTTPException(status_code=404, detail="Research task not found")


@router.get("/result/{task_id}", response_model=ResearchTaskResult)
async def get_research_task_result(task_id: str) -> ResearchTaskResult:
    """
    Get the result of a completed research task.
    
    Args:
        task_id: Unique identifier for the research task
        
    Returns:
        Complete research task results
    """
    research_run = ResearchRunRepository.get_research_run(task_id)
    if not research_run:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    if research_run["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Research task is not yet completed")
    
    # Extract mode from config
    config = research_run.get("config", {})
    mode = config.get("mode", "single") if isinstance(config, dict) else "single"
    
    return ResearchTaskResult(
        task_id=task_id,
        status=research_run["status"],
        research_question=research_run["research_question"],
        mode=mode,
        final_answer=research_run.get("final_answer"),
        generation_summary=research_run.get("generation_summary"),
        statistics=research_run.get("statistics"),
        sub_queries=research_run.get("sub_queries", []),
        sources_gathered=research_run.get("sources_gathered", []),
        judged_sources=research_run.get("judged_sources", []),
        evidence=research_run.get("evidence", []),
        compressed_notes=research_run.get("compressed_notes", ""),
        workflow_messages=research_run.get("workflow_messages", []),
        created_at=_convert_datetime_to_string(research_run["created_at"]),
        started_at=_convert_datetime_to_string(research_run.get("started_at")),
        completed_at=_convert_datetime_to_string(research_run.get("completed_at")),
        error_message=research_run.get("error_message")
    )


@router.get("/history", response_model=ResearchHistoryResponse)
async def get_research_history(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
    mode_filter: Optional[Literal["single", "multi"]] = None
) -> ResearchHistoryResponse:
    """
    Get research task history from the database.
    
    Args:
        limit: Maximum number of items to return
        offset: Number of items to skip
        status_filter: Optional status filter ("completed", "failed", etc.)
        mode_filter: Optional mode filter ("single", "multi")
        
    Returns:
        Paginated list of research history items
    """
    try:
        # Get research runs from database
        research_runs = ResearchRunRepository.get_research_runs_history(
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )
        
        # Apply mode filter if specified
        if mode_filter:
            filtered_runs = []
            for run in research_runs:
                config = run.get("config", {})
                mode = config.get("mode", "single") if isinstance(config, dict) else "single"
                if mode == mode_filter:
                    filtered_runs.append(run)
            research_runs = filtered_runs
        
        # Get total count for pagination
        from ...db import get_cursor
        with get_cursor() as cursor:
            query = "SELECT COUNT(*) FROM research_runs"
            params = []
            
            conditions = []
            if status_filter:
                conditions.append("status = %s")
                params.append(status_filter)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            cursor.execute(query, params)
            total = cursor.fetchone()["count"]
        
        # Convert to response model
        history_items = []
        for run in research_runs:
            # Extract mode from config
            config = run.get("config", {})
            mode = config.get("mode", "single") if isinstance(config, dict) else "single"
            
            history_items.append(ResearchHistoryItem(
                task_id=run["task_id"],
                research_question=run["research_question"],
                status=run["status"],
                mode=mode,
                created_at=_convert_datetime_to_string(run["created_at"]),
                started_at=_convert_datetime_to_string(run.get("started_at")),
                completed_at=_convert_datetime_to_string(run.get("completed_at")),
                statistics=run.get("statistics")
            ))
        
        return ResearchHistoryResponse(
            items=history_items,
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Error getting research history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get research history: {str(e)}")


@router.delete("/{task_id}")
async def delete_research_task(task_id: str) -> Dict[str, str]:
    """
    Delete a research task (cancel if running, remove if completed/failed).
    
    Args:
        task_id: Unique identifier for the research task
        
    Returns:
        Deletion/cancellation confirmation
    """
    research_run = ResearchRunRepository.get_research_run(task_id)
    if not research_run:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    # If task is still running, cancel it
    if research_run["status"] in ["pending", "running"]:
        # Update global dictionaries
        if task_id in research_tasks:
            research_tasks[task_id]["status"] = "cancelled"
            research_tasks[task_id]["completed_at"] = datetime.utcnow()
        
        # Update both tables
        completed_at = datetime.utcnow().isoformat()
        
        ResearchRunRepository.update_research_run_status(
            task_id=task_id,
            status="cancelled",
            completed_at=completed_at
        )
        
        TaskRepository.update_task_status(
            task_id=task_id,
            status="cancelled",
            end_time=completed_at,
            message="Task cancelled by user request"
        )
        
        logger.info(f"Cancelled research task {task_id}")
        return {"message": f"Research task {task_id} has been cancelled"}
    
    # If task is completed/failed/cancelled, delete it from database
    else:
        # Remove from global dictionary if present
        if task_id in research_tasks:
            del research_tasks[task_id]
        
        # Delete from both tables using the existing database methods
        try:
            # Delete from research_runs table (and associated state records)
            ResearchRunRepository.delete_research_run(task_id)
            # Delete from tasks table
            TaskRepository.delete_task(task_id)
            
            logger.info(f"Deleted research task {task_id}")
            return {"message": f"Research task {task_id} has been deleted"}
        
        except Exception as e:
            logger.error(f"Error deleting research task {task_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete research task: {str(e)}")


@router.get("/workflow/info")
async def get_workflow_info() -> Dict[str, Any]:
    """
    Get information about the research workflow configuration.
    
    Returns:
        Workflow configuration information for both modes
    """
    try:
        current_mode = SettingsRepository.get_research_agent_mode()
        status = SettingsRepository.get_dual_mode_status()
        
        workflow_info = {
            "current_mode": current_mode,
            "dual_mode_status": status,
            "modes": {
                "single": {
                    "description": "Sequential workflow with research loops",
                    "workflow_type": "langgraph", 
                    "available": status["has_single_config"]
                },
                "multi": {
                    "description": "Parallel multi-agent orchestration",
                    "workflow_type": "orchestrator",
                    "available": status["has_multi_config"]
                }
            }
        }
        
        # Add mode-specific details
        if current_mode == "single":
            try:
                single_workflow = await get_research_workflow()
                workflow_info["current_workflow"] = single_workflow.get_workflow_info()
            except Exception as e:
                workflow_info["current_workflow"] = {"error": str(e)}
        else:
            try:
                multi_orchestrator = await get_multi_agent_orchestrator()
                workflow_info["current_workflow"] = multi_orchestrator.get_orchestration_info()
            except Exception as e:
                workflow_info["current_workflow"] = {"error": str(e)}
        
        return workflow_info
        
    except Exception as e:
        logger.error(f"Error getting workflow info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow info: {str(e)}")


async def _run_single_agent_research_task(
    task_id: str,
    research_question: str,
    config: Optional[Dict[str, Any]],
    save_to_library: bool
) -> None:
    """
    Background task to run the single-agent research workflow.
    
    Args:
        task_id: Unique identifier for the research task
        research_question: The research question to investigate
        config: Optional configuration overrides
        save_to_library: Whether to save results to research library
    """
    try:
        # Get the workflow
        workflow = await get_research_workflow()
        
        # Run the existing single-agent workflow logic
        await _run_research_task_workflow(task_id, research_question, config, save_to_library, workflow)
        
    except Exception as e:
        logger.error(f"Single-agent research task {task_id} failed: {e}")
        await _handle_research_task_failure(task_id, str(e))


async def _run_multi_agent_research_task(
    task_id: str,
    research_question: str,
    config: Optional[Dict[str, Any]],
    save_to_library: bool
) -> None:
    """
    Background task to run the multi-agent research orchestration.
    
    Args:
        task_id: Unique identifier for the research task
        research_question: The research question to investigate
        config: Optional configuration overrides
        save_to_library: Whether to save results to research library
    """
    try:
        # Get the orchestrator
        orchestrator = await get_multi_agent_orchestrator()
        
        # Initialize task progress
        started_at = datetime.utcnow()
        research_tasks[task_id] = {
            "task_id": task_id,
            "research_question": research_question,
            "mode": "multi",
            "status": "running",
            "created_at": started_at,
            "started_at": started_at,
            "progress": {"phase": "initializing", "status": "Starting multi-agent orchestration"}
        }
        
        # Update task status to running
        started_at_iso = started_at.isoformat()
        
        TaskRepository.update_task_status(
            task_id=task_id,
            status="running",
            progress=0.1,
            current_step="Starting multi-agent orchestration",
            message="Multi-agent research workflow initialized"
        )
        
        ResearchRunRepository.update_research_run_status(
            task_id=task_id,
            status="running",
            started_at=started_at_iso
        )
        
        logger.info(f"Starting multi-agent orchestration for task {task_id}")
        
        # Define progress callback for orchestrator
        def orchestration_progress_callback(task_id_progress: str, progress):
            # Update global progress tracking
            if task_id in research_tasks:
                research_tasks[task_id]["progress"] = {
                    "phase": progress.phase,
                    "overall_progress": progress.overall_progress,
                    "status": progress.current_status,
                    "agents_progress": {
                        agent_id: {
                            "status": agent_progress.status.value,
                            "current_task": agent_progress.current_task,
                            "progress_percentage": agent_progress.progress_percentage,
                            "sources_found": agent_progress.sources_found
                        }
                        for agent_id, agent_progress in progress.agents_progress.items()
                    }
                }
        
        # Run orchestration
        result: OrchestrationResult = await orchestrator.orchestrate_research(
            research_question=research_question,
            task_id=task_id,
            progress_callback=orchestration_progress_callback,
            config_overrides=config
        )
        
        # Update completion status
        completed_at = datetime.utcnow().isoformat()
        
        if result.success:
            # Update global dictionaries
            if task_id in research_tasks:
                research_tasks[task_id]["status"] = "completed"
                research_tasks[task_id]["completed_at"] = datetime.utcnow()
            
            task_results[task_id] = result
            
            # Update general task status
            TaskRepository.update_task_status(
                task_id=task_id,
                status="completed",
                progress=1.0,
                current_step="Multi-agent research completed",
                message="Multi-agent orchestration completed successfully",
                result_json=_serialize_paper_info(result),
                end_time=completed_at
            )
            
            # Update research run with results
            ResearchRunRepository.update_research_run_status(
                task_id=task_id,
                status="completed",
                completed_at=completed_at
            )
            
            # Save detailed results
            ResearchRunRepository.update_research_run_results(
                task_id=task_id,
                final_answer=result.final_answer,
                generation_summary=result.generation_summary,
                statistics=result.statistics,
                sub_queries=result.sub_queries,
                sources_gathered=_serialize_paper_info(result.sources_gathered),
                judged_sources=_serialize_paper_info(result.judged_sources),
                evidence=result.evidence,
                compressed_notes=result.compressed_notes,
                workflow_messages=_serialize_paper_info(result.workflow_messages),
                research_loop_count=result.statistics.get("research_loops", 1),
                is_sufficient=result.success
            )
            
            logger.info(f"Multi-agent research task {task_id} completed successfully")
        else:
            await _handle_research_task_failure(task_id, result.error_message or "Multi-agent orchestration failed")
        
    except Exception as e:
        logger.error(f"Multi-agent research task {task_id} failed: {e}")
        await _handle_research_task_failure(task_id, str(e))


async def _run_research_task_workflow(
    task_id: str,
    research_question: str,
    config: Optional[Dict[str, Any]],
    save_to_library: bool,
    workflow: ResearchAgentWorkflow
) -> None:
    """
    Run the existing single-agent research workflow (reusing existing implementation).
    """
    # This is the existing implementation from the original file
    # We keep it unchanged to maintain backward compatibility
    try:
        # Initialize task in global dictionaries for WebSocket tracking
        started_at = datetime.utcnow()
        research_tasks[task_id] = {
            "task_id": task_id,
            "research_question": research_question,
            "mode": "single",
            "status": "running",
            "created_at": started_at,
            "started_at": started_at,
            "progress": {"current_node": "initializing", "status": "Starting research workflow"}
        }
        
        # Update task status to running in both tables
        started_at_iso = started_at.isoformat()
        
        TaskRepository.update_task_status(
            task_id=task_id,
            status="running",
            progress=0.1,
            current_step="Starting research workflow",
            message="Research workflow initialized"
        )
        
        ResearchRunRepository.update_research_run_status(
            task_id=task_id,
            status="running",
            started_at=started_at_iso
        )
        
        logger.info(f"Starting research workflow for task {task_id}")
        
        # Run the research workflow with task_id for progress tracking
        results = await workflow.run_research(research_question, config, task_id)
        
        # Update task completion status
        completed_at = datetime.utcnow().isoformat()
        
        if results.get("success", False):
            # Update global dictionaries
            if task_id in research_tasks:
                research_tasks[task_id]["status"] = "completed"
                research_tasks[task_id]["completed_at"] = datetime.utcnow()
            
            task_results[task_id] = results
            
            # Update general task status (serialize PaperInfo objects in results)
            TaskRepository.update_task_status(
                task_id=task_id,
                status="completed",
                progress=1.0,
                current_step="Research completed",
                message="Research workflow completed successfully",
                result_json=_serialize_paper_info(results),
                end_time=completed_at
            )
            
            # Update research run with detailed results
            ResearchRunRepository.update_research_run_status(
                task_id=task_id,
                status="completed",
                completed_at=completed_at
            )
            
            # Save detailed results to research library (serialize PaperInfo objects)
            ResearchRunRepository.update_research_run_results(
                task_id=task_id,
                final_answer=results.get("final_answer"),
                generation_summary=results.get("generation_summary"),
                statistics=results.get("statistics"),
                sub_queries=results.get("sub_queries", []),
                sources_gathered=_serialize_paper_info(results.get("sources_gathered", [])),
                judged_sources=_serialize_paper_info(results.get("judged_sources", [])),
                evidence=results.get("evidence", []),
                compressed_notes=results.get("compressed_notes", ""),
                workflow_messages=_serialize_paper_info(results.get("workflow_messages", [])),
                research_loop_count=results.get("statistics", {}).get("research_loops", 0),
                is_sufficient=results.get("statistics", {}).get("evidence_sufficient", False)
            )
            
            logger.info(f"Research task {task_id} completed successfully and saved to research library")
        else:
            error_message = results.get("error", "Unknown error")
            await _handle_research_task_failure(task_id, error_message)
        
    except Exception as e:
        logger.error(f"Error in research task {task_id}: {e}")
        await _handle_research_task_failure(task_id, str(e))


async def _handle_research_task_failure(task_id: str, error_message: str) -> None:
    """
    Handle research task failure for both single and multi-agent modes.
    """
    # Update global dictionaries
    if task_id in research_tasks:
        research_tasks[task_id]["status"] = "failed"
        research_tasks[task_id]["completed_at"] = datetime.utcnow()
        research_tasks[task_id]["error_message"] = error_message
    
    # Update task status to failed in both tables
    completed_at = datetime.utcnow().isoformat()
    
    TaskRepository.update_task_status(
        task_id=task_id,
        status="failed",
        progress=0.0,
        current_step="Research failed",
        message="Research workflow encountered an error",
        error=error_message,
        end_time=completed_at
    )
    
    ResearchRunRepository.update_research_run_status(
        task_id=task_id,
        status="failed",
        completed_at=completed_at,
        error_message=error_message
    )


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for the research agent service."""
    return {
        "status": "healthy",
        "service": "research_agent",
        "timestamp": _convert_datetime_to_string(datetime.utcnow()),
        "dual_mode_ready": True,
        "current_mode": SettingsRepository.get_research_agent_mode()
    }

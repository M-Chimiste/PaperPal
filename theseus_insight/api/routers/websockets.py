from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import asyncio
import json

from ..tasks import task_manager, TaskStatus

router = APIRouter(tags=["websockets"])

# Enhance the WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        """Connect a new WebSocket client.

        This method connects a new WebSocket client to the task manager.
        It accepts the task ID and the WebSocket connection.

        Args:
            task_id (str): The ID of the task to connect to.
            websocket (WebSocket): The WebSocket connection to connect to.

        Raises:
            Exception: If an error occurs while connecting the WebSocket client.
        """
        try:
            await websocket.accept()
            if task_id not in self.active_connections:
                self.active_connections[task_id] = []
            self.active_connections[task_id].append(websocket)
        except Exception as e:
            print(f"Error connecting WebSocket: {e}")
            try:
                await websocket.close(code=4000, reason=str(e))
            except Exception:
                pass
            raise

    def disconnect(self, task_id: str, websocket: WebSocket):
        """Disconnect a WebSocket client.

        This method disconnects a WebSocket client from the task manager.
        It accepts the task ID and the WebSocket connection.

        Args:
            task_id (str): The ID of the task to disconnect from.
            websocket (WebSocket): The WebSocket connection to disconnect.

        Raises:
            Exception: If an error occurs while disconnecting the WebSocket client.
        """
        try:
            if task_id in self.active_connections:
                if websocket in self.active_connections[task_id]:
                    self.active_connections[task_id].remove(websocket)
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]
        except Exception as e:
            print(f"Error disconnecting WebSocket: {e}")

    async def broadcast_status(self, task_id: str, status):
        """Broadcast status to all connected clients for a task.

        This method broadcasts the status of a task to all connected clients.
        It accepts the task ID and the status to broadcast.

        Args:
            task_id (str): The ID of the task to broadcast the status for.
            status: The status to broadcast.

        Raises:
            Exception: If an error occurs while broadcasting the status.
        """
        if task_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(status.dict())
                except WebSocketDisconnect:
                    dead_connections.append(connection)
                except Exception as e:
                    print(f"Error broadcasting to WebSocket: {e}")
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                self.disconnect(task_id, dead)

    async def close_all(self, task_id: str):
        """Close all connections for a task.

        This method closes all connections for a task.
        It accepts the task ID.

        Args:
            task_id (str): The ID of the task to close all connections for.

        Raises:
            Exception: If an error occurs while closing the connections.
        """
        if task_id in self.active_connections:
            connections_to_close = self.active_connections[task_id].copy()
            for connection in connections_to_close:
                try:
                    await connection.close(code=1000)
                except Exception:
                    pass
            del self.active_connections[task_id]
    
    async def cleanup_all(self):
        """Close all active connections.

        This method closes all active connections.
        """
        task_ids = list(self.active_connections.keys())
        for task_id in task_ids:
            await self.close_all(task_id)

manager = ConnectionManager()

async def handle_websocket_connection(websocket: WebSocket, task_id: str, endpoint_name: str):
    """Generic WebSocket handler for all task types.

    This method handles the WebSocket connection for all task types.
    It accepts the WebSocket connection, the task ID, and the endpoint name.

    Args:
        websocket (WebSocket): The WebSocket connection to handle.
        task_id (str): The ID of the task to handle.
        endpoint_name (str): The name of the endpoint to handle.

    Raises:
        Exception: If an error occurs while handling the WebSocket connection.
    """
    status_queue = None
    try:
        await manager.connect(task_id, websocket)
        
        # Subscribe to task updates
        status_queue = await task_manager.subscribe_to_updates(task_id)
        
        # --------------------------------------------
        # Send an immediate snapshot of current status
        # --------------------------------------------
        try:
            from datetime import datetime
            from ..models import RunStatus, NodeStatus  # Local import to avoid circular issues

            current_task = await asyncio.to_thread(task_manager.get_task_status, task_id)
            if current_task:
                snapshot_status = RunStatus(
                    taskId=task_id,
                    nodes=[
                        NodeStatus(
                            nodeId="main",
                            status=current_task["status"],
                            message=current_task.get("message", ""),
                            progress=current_task.get("progress", 0),
                            timestamp=datetime.now().isoformat(),
                        )
                    ],
                    overallStatus=current_task["status"],
                    currentStep=current_task.get("current_step"),
                    progress=current_task.get("progress", 0),
                    message=current_task.get("message", ""),
                    result=current_task.get("result_json"),
                    error=current_task.get("error"),
                    metadata=current_task.get("metadata"),  # Include metadata in snapshot
                )
                # Push snapshot to the front of the queue so it will be sent first
                await status_queue.put(snapshot_status)
        except Exception as snap_err:
            # Log but don't fail the connection
            print(f"Error sending snapshot status for task {task_id}: {snap_err}")

        # Main loop: forward subsequent updates
        while True:
            try:
                status = await asyncio.wait_for(status_queue.get(), timeout=2)
            except asyncio.TimeoutError:
                # Another API instance may own this task. Persisted state is
                # authoritative; reconnecting clients must continue progressing.
                current_task = await asyncio.to_thread(task_manager.get_task_status, task_id)
                if not current_task:
                    break
                state = current_task['status']
                if state in {'cancelled', 'canceled'}:
                    state = 'failed'
                status = RunStatus(taskId=task_id, nodes=[], overallStatus=state,
                    currentStep=current_task.get('current_step'), progress=current_task.get('progress') or 0,
                    message=current_task.get('message') or '', result=current_task.get('result_json'),
                    error=current_task.get('error'), metadata=current_task.get('metadata'))
            # Sentinel for shutdown
            if status is None:
                break

            await websocket.send_json(status.dict())

            if status.overallStatus in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                # Add small delay to ensure message is sent before closing
                await asyncio.sleep(0.1)
                break

    except WebSocketDisconnect:
        pass
    except ValueError as e:
        try:
            await websocket.close(code=4004, reason=str(e))
        except Exception:
            pass
    except Exception as e:
        try:
            await websocket.close(code=4000, reason=str(e))
        except Exception:
            pass
    finally:
        if status_queue:
            await task_manager.unsubscribe_from_updates(task_id, status_queue)
        manager.disconnect(task_id, websocket)
        # Ensure the connection is properly closed
        try:
            await websocket.close(code=1000)
        except Exception:
            pass

async def handle_research_agent_connection(websocket: WebSocket, task_id: str):
    """Specialized WebSocket handler for research agent tasks.

    This method handles the WebSocket connection specifically for research agent tasks,
    providing real-time progress updates for both single-agent and multi-agent workflows.

    Args:
        websocket (WebSocket): The WebSocket connection to handle.
        task_id (str): The ID of the research task to handle.

    Raises:
        Exception: If an error occurs while handling the WebSocket connection.
    """
    try:
        await manager.connect(task_id, websocket)
        
        # Import here to avoid circular imports
        from .research_agent import research_tasks, task_results
        
        # Check if task exists
        if task_id not in research_tasks:
            await websocket.close(code=4004, reason="Research task not found")
            return
        
        # Send initial status
        task_info = research_tasks[task_id]
        mode = task_info.get("mode", "single")
        
        initial_message = {
            "type": "status_update",
            "task_id": task_id,
            "status": task_info["status"],
            "mode": mode,
            "progress": task_info.get("progress", {}),
            "timestamp": task_info["created_at"].isoformat()
        }
        await websocket.send_json(initial_message)
        
        # Monitor task progress
        last_progress_hash = None
        while True:
            await asyncio.sleep(1)  # Poll every second
            
            # Get current task info
            current_task_info = research_tasks.get(task_id)
            if not current_task_info:
                break
            
            # Check if progress has actually changed to avoid unnecessary updates
            current_progress = current_task_info.get("progress", {})
            current_progress_hash = str(hash(str(current_progress)))
            
            # Only send update if progress changed or status changed
            should_send_update = (
                last_progress_hash != current_progress_hash or
                task_info.get("status") != current_task_info["status"]
            )
            
            if should_send_update:
                # Prepare status message with mode information
                status_message = {
                    "type": "status_update",
                    "task_id": task_id,
                    "status": current_task_info["status"],
                    "mode": current_task_info.get("mode", "single"),
                    "progress": current_progress,
                    "timestamp": current_task_info.get("started_at", current_task_info["created_at"]).isoformat()
                }
                
                # Add error message if failed
                if current_task_info.get("error_message"):
                    status_message["error_message"] = current_task_info["error_message"]
                
                await websocket.send_json(status_message)
                last_progress_hash = current_progress_hash
                task_info = current_task_info  # Update reference for status comparison
            
            # If task is completed, send final results and close
            if current_task_info["status"] in ["completed", "failed", "cancelled"]:
                if current_task_info["status"] == "completed":
                    results = task_results.get(task_id, {})
                    
                    # Prepare results based on mode
                    if current_task_info.get("mode") == "multi":
                        # Multi-agent results
                        result_data = {
                            "final_answer": results.get("final_answer"),
                            "generation_summary": results.get("generation_summary"),
                            "statistics": results.get("statistics", {}),
                            "sub_queries": results.get("sub_queries", []),
                            "sources_count": len(results.get("sources_gathered", [])),
                            "evidence_count": len(results.get("evidence", [])),
                            "agents_used": results.get("statistics", {}).get("agents_used", 0),
                            "synthesis_confidence": results.get("statistics", {}).get("synthesis_confidence", 0.0)
                        }
                    else:
                        # Single-agent results (backward compatible)
                        result_data = {
                            "final_answer": results.get("final_answer"),
                            "statistics": results.get("statistics"),
                            "sub_queries": results.get("sub_queries", []),
                            "sources_count": len(results.get("sources_gathered", [])),
                            "evidence_count": len(results.get("evidence", []))
                        }
                    
                    final_message = {
                        "type": "task_completed",
                        "task_id": task_id,
                        "status": "completed",
                        "mode": current_task_info.get("mode", "single"),
                        "results": result_data,
                        "timestamp": current_task_info.get("completed_at", current_task_info["created_at"]).isoformat()
                    }
                    await websocket.send_json(final_message)
                elif current_task_info["status"] == "failed":
                    # Send failure notification
                    failure_message = {
                        "type": "task_completed",
                        "task_id": task_id,
                        "status": "failed",
                        "mode": current_task_info.get("mode", "single"),
                        "error_message": current_task_info.get("error_message", "Research task failed"),
                        "timestamp": current_task_info.get("completed_at", current_task_info["created_at"]).isoformat()
                    }
                    await websocket.send_json(failure_message)
                
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close(code=4000, reason=str(e))
        except Exception:
            pass
    finally:
        manager.disconnect(task_id, websocket)

async def handle_bulk_judge_connection(websocket: WebSocket, job_id: str):
    """Specialized WebSocket handler for bulk judge jobs with enhanced monitoring."""
    try:
        print(f"🔌 WebSocket connection attempt for job_id: {job_id}")
        
        # Validate job_id format first
        try:
            from uuid import UUID
            job_uuid = UUID(job_id)
            print(f"✅ Valid UUID format: {job_uuid}")
        except ValueError as e:
            print(f"❌ Invalid UUID format for job_id: {job_id}, error: {e}")
            await websocket.close(code=4004, reason=f"Invalid job ID format: {job_id}")
            return

        await manager.connect(job_id, websocket)
        print(f"✅ WebSocket connected for job: {job_id}")

        # Send initial job status
        try:
            from ..routers.bulk_operations import get_job_metrics

            # Get initial job metrics
            print(f"🔍 Getting metrics for job: {job_uuid}")
            metrics = await get_job_metrics(job_uuid)
            print(f"✅ Got metrics for job: {job_id}")

            initial_message = {
                "type": "bulk_judge_status",
                "job_id": job_id,
                "status": "connected",
                "message": "Connected to bulk judge monitoring",
                "data": metrics
            }

            await websocket.send_json(initial_message)
            print(f"✅ Sent initial message for job: {job_id}")

        except Exception as e:
            print(f"❌ Failed to get initial job status for {job_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to get initial job status: {str(e)}"
            })

        # Main monitoring loop
        while True:
            try:
                # Check for updated job status
                try:
                    from ..routers.bulk_operations import get_job_metrics

                    # Get current job metrics
                    metrics = await get_job_metrics(job_uuid)

                    from datetime import datetime
                    
                    status_message = {
                        "type": "bulk_judge_update",
                        "job_id": job_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "data": metrics
                    }

                    await websocket.send_json(status_message)

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to get job metrics: {str(e)}"
                    })

                # Wait before next update (more frequent for bulk judge due to distributed nature)
                await asyncio.sleep(5)  # 5 second intervals for real-time monitoring

            except WebSocketDisconnect:
                break
            except Exception as e:
                error_message = {
                    "type": "error",
                    "message": f"Monitoring error: {str(e)}"
                }
                try:
                    await websocket.send_json(error_message)
                except:
                    break
                await asyncio.sleep(10)  # Back off on errors

    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
        try:
            await websocket.close(code=4000, reason=str(e))
        except:
            pass
    finally:
        manager.disconnect(job_id, websocket)

# WebSocket endpoints
@router.websocket("/ws/newsletter/{task_id}")
async def newsletter_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for newsletter generation status updates."""
    await handle_websocket_connection(websocket, task_id, "newsletter")

@router.websocket("/ws/podcast/{task_id}")
async def podcast_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for podcast generation status updates."""
    await handle_websocket_connection(websocket, task_id, "podcast")

@router.websocket("/ws/visualizer/{task_id}")
async def visualizer_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for visualizer generation status updates."""
    await handle_websocket_connection(websocket, task_id, "visualizer")

@router.websocket("/ws/database-import/{task_id}")
async def database_import_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for database import status updates."""
    await handle_websocket_connection(websocket, task_id, "database-import")

@router.websocket("/ws/database-export/{task_id}")
async def database_export_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for database export status updates."""
    await handle_websocket_connection(websocket, task_id, "database-export")

@router.websocket("/ws/research-agent/{task_id}")
async def research_agent_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for research agent status updates."""
    await handle_research_agent_connection(websocket, task_id)

@router.websocket("/ws/mindmap/{task_id}")
async def mindmap_expand_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for mind-map expansion task status updates."""
    await handle_websocket_connection(websocket, task_id, "mindmap_expand")

@router.websocket("/ws/mindmap-pdf-parse/{task_id}")
async def mindmap_pdf_parse_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for mind-map PDF parsing task status updates."""
    await handle_websocket_connection(websocket, task_id, "mindmap_pdf_parse")

@router.websocket("/ws/trends/{task_id}")
async def trends_recompute_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for trends recomputation status updates."""
    await handle_websocket_connection(websocket, task_id, "trends")

@router.websocket("/ws/star-map/{task_id}")
async def star_map_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for star map recomputation status updates."""
    await handle_websocket_connection(websocket, task_id, "star-map")

@router.websocket("/ws/bulk-judge/{job_id}")
async def bulk_judge_status(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for bulk judge job status updates."""
    await handle_bulk_judge_connection(websocket, job_id)

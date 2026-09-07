"""API router for research profiles management."""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Body
from typing import Optional, List, Dict, Any
import json
import uuid
import asyncio
import os
from datetime import datetime

from ..models import (
    ProfileResponse, ProfileCreateRequest, ProfileUpdateRequest,
    ProfileWithStatsResponse, ProfileTagSearchResponse,
    ProfileInterestResponse, ProfileInterestCreateRequest,
    BulkJudgeRunRequest, BulkJudgeRunResponse,
    ProfileNewsletterRequest, ProfileNewsletterResponse
)
from ...data_access.profiles import (
    ProfileRepository, ProfileInterestsRepository, ProfileScoreRepository
)
from ...data_access.papers import PaperRepository
from ...data_access.settings import SettingsRepository
from ...data_access.star_map import ProfileStarMapRepository
from ..tasks import task_manager, TaskStatus
from ..helpers.profile_filtering import parse_id_csv, parse_tag_csv
from ..helpers.serialization import decode_json_fields, isoformat_fields
from ...theseus_insight import TheseusInsight


def _convert_profile_timestamps(profile_data: dict) -> dict:
    """Convert PostgreSQL datetime objects to ISO strings for API response."""
    converted = isoformat_fields(profile_data, ('created_at', 'updated_at'))
    return decode_json_fields(converted, ('tags', 'email_recipients', 'arxiv_filters'))


router = APIRouter(prefix="/api/profiles", tags=["profiles"])


# =====================================================================
# Profile Management Endpoints
# =====================================================================

@router.get("", response_model=List[ProfileResponse])
async def get_profiles(include_inactive: bool = Query(False, description="Include inactive profiles")):
    """
    Get all research profiles with paper counts.
    
    Args:
        include_inactive: Whether to include inactive profiles in the response
        
    Returns:
        List of research profiles with paper statistics
    """
    try:
        profiles = ProfileRepository.get_all(include_inactive=include_inactive)
        
        # Add paper counts for each profile
        enriched_profiles = []
        for profile in profiles:
            # Get basic paper stats for this profile
            paper_stats = ProfileScoreRepository.get_profile_paper_stats(profile['id'])
            
            # Add total_papers to the profile data
            profile_data = _convert_profile_timestamps(profile)
            profile_data['total_papers'] = paper_stats['basic_stats']['total_papers']
            
            enriched_profiles.append(ProfileResponse(**profile_data))
        
        return enriched_profiles
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profiles: {str(e)}")


@router.post("", response_model=ProfileResponse)
async def create_profile(request: ProfileCreateRequest):
    """
    Create a new research profile.
    
    Args:
        request: Profile creation data
        
    Returns:
        Created profile data
    """
    try:
        # Check if profile name already exists
        if ProfileRepository.exists_by_name(request.name):
            raise HTTPException(status_code=400, detail=f"Profile with name '{request.name}' already exists")
        
        profile = ProfileRepository.create(
            name=request.name,
            description=request.description,
            color=request.color,
            tags=request.tags,
            email_recipients=request.email_recipients,
            arxiv_filters=request.arxiv_filters,
            is_default=False  # Only migration can create default profile
        )
        
        # Create research interests if provided
        if request.research_interests:
            ProfileInterestsRepository.bulk_create(profile["id"], request.research_interests)
        
        return ProfileResponse(**_convert_profile_timestamps(profile))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating profile: {str(e)}")


# =====================================================================
# Tag Management Endpoints (must come before /{profile_id} route)
# =====================================================================

@router.get("/tags/search", response_model=ProfileTagSearchResponse)
async def search_tags(
    q: str = Query(..., min_length=2, description="Search query for tags"),
    limit: int = Query(10, gt=0, le=50, description="Maximum number of suggestions")
):
    """
    Search for tags with auto-complete functionality.
    
    Args:
        q: Search query (minimum 2 characters)
        limit: Maximum number of suggestions
        
    Returns:
        Tag search results with usage counts
    """
    try:
        suggestions = ProfileRepository.search_tags(q, limit)
        return ProfileTagSearchResponse(
            query=q,
            suggestions=suggestions,
            exact_match=any(tag["tag"].lower() == q.lower() for tag in suggestions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching tags: {str(e)}")


@router.get("/tags", response_model=List[str])
async def get_all_tags():
    """
    Get all unique tags across all profiles.
    
    Returns:
        List of unique tags
    """
    try:
        return ProfileRepository.get_all_tags()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tags: {str(e)}")


@router.get("/by-tag/{tag}", response_model=List[ProfileResponse])
async def get_profiles_by_tag(tag: str):
    """
    Get all profiles that have a specific tag.
    
    Args:
        tag: Tag name
        
    Returns:
        List of profiles with the specified tag
    """
    try:
        profiles = ProfileRepository.get_by_tag(tag)
        return [
            ProfileResponse(**_convert_profile_timestamps(profile))
            for profile in profiles
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profiles by tag: {str(e)}")


@router.get("/by-tags", response_model=List[ProfileResponse])
async def get_profiles_by_tags(tags: List[str] = Query(..., description="List of tags")):
    """
    Get all profiles that have any of the specified tags.
    
    Args:
        tags: List of tag names
        
    Returns:
        List of profiles with any of the specified tags
    """
    try:
        profiles = ProfileRepository.get_by_tags(tags)
        return [
            ProfileResponse(**_convert_profile_timestamps(profile))
            for profile in profiles
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profiles by tags: {str(e)}")


@router.get("/default", response_model=ProfileResponse)
async def get_default_profile():
    """
    Get the default profile.
    
    If no default profile exists, automatically creates one using current settings
    with fallback to research_interests.txt.
    
    Returns:
        Default profile data
    """
    try:
        profile = ProfileRepository.get_default()
        if not profile:
            # Create default profile if it doesn't exist
            profile = await _create_default_profile()
        
        return ProfileResponse(**_convert_profile_timestamps(profile))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching default profile: {str(e)}")


async def _create_default_profile():
    """
    Create a default profile populated with existing settings and fallback logic.
    
    Returns:
        Created default profile data
    """
    from ...utils.path_resolver import get_config_path
    import os
    
    # Get research interests from settings with fallback to file
    research_interests_text = SettingsRepository.get("research_interests")
    if not research_interests_text:
        # Fallback to research_interests.txt file
        config_path = get_config_path('research_interests.txt')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                research_interests_text = f.read().strip()
        else:
            research_interests_text = ""
    
    # Parse research interests into array
    research_interests = []
    if research_interests_text:
        research_interests = [
            line.strip() for line in research_interests_text.split('\n') 
            if line.strip() and not line.strip().startswith('#')
        ]
    
    # Get email recipients from settings
    email_recipients = []
    email_recipients_json = SettingsRepository.get("email_recipients")
    if email_recipients_json:
        try:
            email_recipients = json.loads(email_recipients_json)
            if not isinstance(email_recipients, list):
                email_recipients = []
        except json.JSONDecodeError:
            email_recipients = []
    
    # Get arxiv categories from settings with fallback to defaults
    arxiv_filters = {}
    arxiv_categories_json = SettingsRepository.get("arxiv_search_categories")
    if arxiv_categories_json:
        try:
            arxiv_data = json.loads(arxiv_categories_json)
            if isinstance(arxiv_data, dict):
                arxiv_filters = arxiv_data
        except json.JSONDecodeError:
            pass
    
    # Use default arxiv categories if none found
    if not arxiv_filters:
        arxiv_filters = {
            "main_category": "cs",
            "filter_categories": ["cs.ai", "cs.cl", "cs.lg", "cs.ir", "cs.ma", "cs.cv"]
        }
    
    # Create the default profile
    profile = ProfileRepository.create(
        name="Default",
        description="Default profile created from existing system settings",
        color="#1f77b4",
        tags=["default", "auto-created"],
        email_recipients=email_recipients,
        arxiv_filters=arxiv_filters,
        is_default=True
    )
    
    # Create research interests if we have any
    if research_interests:
        ProfileInterestsRepository.bulk_create(profile["id"], research_interests)
    
    return profile


# =====================================================================
# Profile Management Endpoints (parameterized routes come last)
# =====================================================================

@router.get("/{profile_id}", response_model=ProfileWithStatsResponse)
async def get_profile(profile_id: int):
    """
    Get a specific profile by ID with statistics.
    
    Args:
        profile_id: Profile ID
        
    Returns:
        Profile data with statistics
    """
    try:
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        stats = ProfileRepository.get_stats(profile_id)
        interests = ProfileInterestsRepository.get_by_profile(profile_id)
        
        profile_data = _convert_profile_timestamps(profile)
        profile_data.update(stats)
        profile_data["research_interests"] = [
            interest["interest_text"] for interest in interests
        ]
        
        return ProfileWithStatsResponse(**profile_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")


@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(profile_id: int, request: ProfileUpdateRequest):
    """
    Update a profile.
    
    Args:
        profile_id: Profile ID
        request: Profile update data
        
    Returns:
        Updated profile data
    """
    try:
        # Check if profile exists
        existing = ProfileRepository.get_by_id(profile_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        # Check name uniqueness if changing name
        if request.name and request.name != existing["name"]:
            if ProfileRepository.exists_by_name(request.name, exclude_id=profile_id):
                raise HTTPException(status_code=400, detail=f"Profile with name '{request.name}' already exists")
        
        profile = ProfileRepository.update(
            profile_id,
            name=request.name,
            description=request.description,
            color=request.color,
            tags=request.tags,
            email_recipients=request.email_recipients,
            arxiv_filters=request.arxiv_filters,
            is_active=request.is_active
        )
        
        # Update research interests if provided
        if request.research_interests is not None:
            # Delete existing interests and recreate
            ProfileInterestsRepository.delete_by_profile(profile_id)
            if request.research_interests:
                ProfileInterestsRepository.bulk_create(profile_id, request.research_interests)
        
        return ProfileResponse(**_convert_profile_timestamps(profile))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating profile: {str(e)}")


@router.delete("/{profile_id}")
async def delete_profile(profile_id: int):
    """
    Delete a profile.
    
    Args:
        profile_id: Profile ID
        
    Returns:
        Success message
    """
    try:
        success = ProfileRepository.delete(profile_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        return {"message": f"Profile {profile_id} deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting profile: {str(e)}")


@router.post("/{profile_id}/clone", response_model=ProfileResponse)
async def clone_profile(profile_id: int, new_name: str = Query(..., description="Name for the cloned profile")):
    """
    Clone an existing profile.
    
    Args:
        profile_id: Source profile ID
        new_name: Name for the new profile
        
    Returns:
        Cloned profile data
    """
    try:
        cloned_profile = ProfileRepository.clone(profile_id, new_name)
        return ProfileResponse(**_convert_profile_timestamps(cloned_profile))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cloning profile: {str(e)}")


@router.put("/{profile_id}/deactivate", response_model=ProfileResponse)
async def deactivate_profile(profile_id: int):
    """
    Deactivate a profile instead of deleting it.
    
    Args:
        profile_id: Profile ID
        
    Returns:
        Deactivated profile data
    """
    try:
        profile = ProfileRepository.deactivate(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        return ProfileResponse(**_convert_profile_timestamps(profile))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deactivating profile: {str(e)}")


# =====================================================================
# Research Interests Endpoints
# =====================================================================

@router.get("/{profile_id}/interests", response_model=List[ProfileInterestResponse])
async def get_profile_interests(profile_id: int):
    """
    Get all research interests for a profile.
    
    Args:
        profile_id: Profile ID
        
    Returns:
        List of research interests
    """
    try:
        # Verify profile exists
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        interests = ProfileInterestsRepository.get_by_profile(profile_id)
        return [
            ProfileInterestResponse(
                id=interest["id"],
                interest_text=interest["interest_text"],
                embedding_model=interest.get("embedding_model"),
                created_at=interest["created_at"].isoformat() if interest["created_at"] else None,
                updated_at=interest["updated_at"].isoformat() if interest["updated_at"] else None
            )
            for interest in interests
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching interests: {str(e)}")


@router.post("/{profile_id}/interests", response_model=ProfileInterestResponse)
async def create_profile_interest(profile_id: int, request: ProfileInterestCreateRequest):
    """
    Create a new research interest for a profile.
    
    Args:
        profile_id: Profile ID
        request: Interest creation data
        
    Returns:
        Created interest data
    """
    try:
        # Verify profile exists
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        interest = ProfileInterestsRepository.create(
            profile_id,
            request.interest_text
        )
        
        return ProfileInterestResponse(
            id=interest["id"],
            interest_text=interest["interest_text"],
            embedding_model=interest.get("embedding_model"),
            created_at=interest["created_at"].isoformat() if interest["created_at"] else None,
            updated_at=interest["updated_at"].isoformat() if interest["updated_at"] else None
        )
    except HTTPException:
        raise
    except Exception as e:
        if "unique constraint" in str(e).lower():
            raise HTTPException(status_code=400, detail="Interest already exists for this profile")
        raise HTTPException(status_code=500, detail=f"Error creating interest: {str(e)}")


@router.delete("/{profile_id}/interests/{interest_id}")
async def delete_profile_interest(profile_id: int, interest_id: int):
    """
    Delete a research interest.
    
    Args:
        profile_id: Profile ID
        interest_id: Interest ID
        
    Returns:
        Success message
    """
    try:
        # Verify profile exists
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        success = ProfileInterestsRepository.delete(interest_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Interest {interest_id} not found")
        
        return {"message": f"Interest {interest_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting interest: {str(e)}")


# =====================================================================
# Profile Papers Endpoints
# =====================================================================

@router.get("/{profile_id}/star-map", response_model=Dict[str, Any])
async def get_profile_star_map(
    profile_id: int,
    limit: int = Query(10000, gt=0, le=10000, description="Maximum number of points to return"),
):
    """Get cached star map points for a profile."""
    try:
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

        data = ProfileStarMapRepository.get_points(profile_id, limit=limit)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching star map: {str(e)}")


@router.get("/{profile_id}/star-map/status", response_model=Dict[str, Any])
async def get_profile_star_map_status(profile_id: int):
    """Get cached star map status (when computed, how many points)."""
    try:
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

        return ProfileStarMapRepository.get_status(profile_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching star map status: {str(e)}")


@router.post("/{profile_id}/star-map/recompute", response_model=Dict[str, Any])
async def recompute_profile_star_map(
    profile_id: int,
    request: Dict[str, Any] = Body(default={}),
):
    """Start a background task to recompute cached star map points for a profile."""
    try:
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

        payload = request or {}
        limit = int(payload.get("limit", 10000))
        if limit <= 0 or limit > 10000:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 10000")

        task_id = str(uuid.uuid4())
        config = {"profile_id": profile_id, "limit": limit}

        await task_manager.create_task(task_id, "star-map", config)

        from ...star_map.task import run_profile_star_map_task

        await task_manager.enqueue_task(run_profile_star_map_task, task_id)

        return {"task_id": task_id, "message": f"Star map recomputation started for profile {profile_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start star map recompute: {str(e)}")


@router.get("/{profile_id}/papers", response_model=Dict[str, Any])
async def get_profile_papers(
    profile_id: int,
    page: int = Query(1, gt=0),
    page_size: int = Query(10, gt=0, le=100),
    min_profile_score: Optional[float] = Query(None, description="Minimum profile-specific score"),
    max_profile_score: Optional[float] = Query(None, description="Maximum profile-specific score"),
    profile_related_only: bool = Query(False, description="Only show papers marked as related"),
    sort_field: Optional[str] = Query("profile_score", enum=['date', 'score', 'profile_score']),
    sort_direction: Optional[str] = Query("desc", enum=['asc', 'desc']),
    search: Optional[str] = Query(None, description="Search in title and abstract"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get papers for a specific profile with profile-specific scoring.
    
    Returns papers that have been scored for the specified profile,
    including profile-specific scores, relatedness, and rationale.
    """
    try:
        # Verify profile exists
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        # Get papers with profile filtering
        from ...data_access.papers import PaperRepository
        
        papers_data = PaperRepository.paginate(
            page=page,
            page_size=page_size,
            profile_ids=[profile_id],  # Filter by this specific profile
            min_profile_score=min_profile_score,
            max_profile_score=max_profile_score,
            profile_related_only=profile_related_only,
            sort_field=sort_field or "profile_score",
            sort_direction=sort_direction or "desc",
            search=search,
            from_date=from_date,
            to_date=to_date
        )
        
        # Convert timestamps for API response
        papers = []
        for p in papers_data['items']:
            converted_p = _convert_paper_timestamps(p)
            papers.append(converted_p)
        
        return {
            "profile": {
                "id": profile['id'],
                "name": profile['name'],
                "description": profile['description']
            },
            "papers": papers,
            "pagination": {
                "total_items": papers_data['total_items'],
                "total_pages": papers_data['total_pages'],
                "current_page": page,
                "page_size": page_size,
                "has_next_page": papers_data['has_next_page']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profile papers: {str(e)}")


@router.get("/{profile_id}/papers/stats")
async def get_profile_paper_stats(profile_id: int):
    """
    Get statistics about papers for a specific profile.
    
    Returns metrics like total papers, score distribution, related papers count, etc.
    """
    try:
        # Verify profile exists
        profile = ProfileRepository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        from ...data_access.profiles import ProfileScoreRepository
        
        stats = ProfileScoreRepository.get_profile_paper_stats(profile_id)
        
        return {
            "profile": {
                "id": profile['id'],
                "name": profile['name']
            },
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profile paper stats: {str(e)}")


# =====================================================================
# Newsletter Generation Endpoints
# =====================================================================

@router.post("/{profile_id}/newsletter", response_model=ProfileNewsletterResponse)
async def generate_profile_newsletter(
    profile_id: int,
    request: ProfileNewsletterRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a newsletter for a specific profile.
    
    This endpoint generates a newsletter tailored to the specified profile's
    research interests and configuration. It supports:
    - Using profile's research interests (or override with custom ones)
    - Using profile's email recipients (or override with custom ones)  
    - Profile-specific arxiv filters and preferences
    - Optional podcast generation
    
    Args:
        profile_id: The ID of the profile to generate newsletter for
        request: Newsletter generation parameters
        background_tasks: Background task manager
        
    Returns:
        Task information for tracking newsletter generation progress
    """
    try:
        # Verify profile exists and is active (run in thread to avoid blocking event loop)
        profile = await asyncio.to_thread(ProfileRepository.get_by_id, profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        if not profile.get('is_active', True):
            raise HTTPException(status_code=400, detail=f"Profile {profile_id} is not active")
        
        # Get profile's research interests if not provided in request
        research_interests_text = request.research_interests
        if not research_interests_text:
            interests = await asyncio.to_thread(ProfileInterestsRepository.get_by_profile, profile_id)
            if interests:
                research_interests_text = "\n".join([interest["interest_text"] for interest in interests])
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Profile {profile_id} has no research interests configured and none provided in request"
                )
        
        # Get email recipients with proper override logic
        email_recipients = []
        
        # Priority 1: Use explicitly provided email_recipients from request
        if request.email_recipients:
            email_recipients = request.email_recipients
        # Priority 2: Use profile recipients if use_profile_recipients is True and available
        elif request.use_profile_recipients and profile.get('email_recipients'):
            profile_recipients = profile['email_recipients']
            if isinstance(profile_recipients, str):
                try:
                    profile_recipients = json.loads(profile_recipients)
                except json.JSONDecodeError:
                    profile_recipients = []
            email_recipients = profile_recipients if profile_recipients else []
        
        if not email_recipients:
            raise HTTPException(
                status_code=400,
                detail="No email recipients specified. Either provide email_recipients in request or configure them in the profile."
            )
        
        # Create task ID and prepare configuration
        task_id = str(uuid.uuid4())
        run_db_path = os.getenv("DATABASE_URL", "postgresql://theseus:theseus@localhost:5432/theseusdb")
        # Prepare profile-specific orchestration config with ArXiv filters
        # Get base orchestration config from settings (run in thread to avoid blocking event loop)
        base_orchestration_json = await asyncio.to_thread(SettingsRepository.get, "orchestration")
        if base_orchestration_json:
            orchestration_config = json.loads(base_orchestration_json)
        else:
            # Fallback to default file-based config
            from ...utils.path_resolver import get_config_path
            config_path = get_config_path('orchestration.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    orchestration_config = json.load(f)
            else:
                # Minimal default config
                orchestration_config = {}
        
        # Override ArXiv categories with profile's filters if available
        profile_arxiv_filters = profile.get('arxiv_filters')
        if profile_arxiv_filters:
            # Parse JSON if it's a string
            if isinstance(profile_arxiv_filters, str):
                try:
                    profile_arxiv_filters = json.loads(profile_arxiv_filters)
                except json.JSONDecodeError:
                    profile_arxiv_filters = None
            
            if profile_arxiv_filters and isinstance(profile_arxiv_filters, dict):
                # Apply profile's ArXiv filters to orchestration config
                orchestration_config['arxiv_search_categories'] = profile_arxiv_filters
                print(f"[DEBUG] Using profile ArXiv filters: {profile_arxiv_filters}")
            else:
                print("[DEBUG] Profile has invalid arxiv_filters, using defaults")
        else:
            print("[DEBUG] Profile has no arxiv_filters, using defaults")

        config = {
            "profile_id": profile_id,
            "start_date": str(request.start_date), "end_date": str(request.end_date),
            "email_recipients": email_recipients, "research_interests": research_interests_text,
            "generate_podcast_run": request.generate_podcast_run,
            "orchestration_config": orchestration_config,
        }
        await task_manager.create_task(task_id, "profile_newsletter", config)
        from ..task_handlers.recoverable import run_profile_newsletter_task
        await task_manager.enqueue_task(run_profile_newsletter_task, task_id)
        
        return ProfileNewsletterResponse(
            task_id=task_id,
            message=f"Newsletter generation started for profile '{profile['name']}'",
            profile_id=profile_id,
            profile_name=profile['name']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting profile newsletter generation: {str(e)}")


# =====================================================================
# Bulk Operations Endpoints
# =====================================================================

@router.post("/bulk/judge-run", response_model=BulkJudgeRunResponse)
async def run_bulk_judge(request: BulkJudgeRunRequest):
    """
    Run LLM judge scoring across multiple profiles.
    
    This endpoint allows running LLM judge scoring for historical papers
    across multiple profiles, with support for:
    - Profile selection by IDs or tags
    - Date range filtering
    - Batch processing
    - Overwrite existing scores option
    """
    try:
        # Create job for tracking
        from ...data_access.job_tracker import get_job_tracker
        
        job_tracker = get_job_tracker()
        
        # Create job description
        profile_desc = ""
        if request.profile_ids:
            profile_desc = f"Profile IDs: {request.profile_ids}"
        elif request.profile_tags:
            profile_desc = f"Profile Tags: {request.profile_tags}"
        else:
            profile_desc = "All active profiles"
            
        date_desc = ""
        if request.from_date or request.to_date:
            date_desc = f" | Date range: {request.from_date or 'start'} to {request.to_date or 'end'}"
            
        job_description = f"Bulk LLM Judge Run - {profile_desc}{date_desc}"
        
        job_id = job_tracker.create_job(
            job_type="bulk_judge",
            description=job_description,
            metadata={
                "request": request.dict(),
                "profiles": profile_desc,
                "date_range": date_desc
            }
        )
        
        # Start the job
        job_tracker.start_job(job_id)
        
        try:
            # Get orchestration config
            from ...data_access import SettingsRepository
            import json
            
            orch_json = SettingsRepository.get("orchestration")
            if not orch_json:
                job_tracker.fail_job(job_id, "Orchestration configuration not found")
                raise HTTPException(
                    status_code=500, 
                    detail="Orchestration configuration not found"
                )
            
            orch_config = json.loads(orch_json)
            
            # Create and run bulk judge runner with progress tracking
            from ...data_access.bulk_judge import create_bulk_judge_runner
            
            def progress_callback(message: str, current: int, total: int):
                job_tracker.update_progress(job_id, current, total, message)
            
            runner = create_bulk_judge_runner(orch_config, verbose=True)
            result = runner.run_bulk_judge(request, progress_callback)
            
            # Complete the job with results
            job_tracker.complete_job(job_id, result.dict())
            
            # Return the result with job ID
            result.job_id = job_id
            return result
            
        except Exception as e:
            job_tracker.fail_job(job_id, str(e))
            raise HTTPException(status_code=500, detail=f"Bulk judge run failed: {str(e)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bulk judge run: {str(e)}")


@router.get("/jobs")
async def list_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(50, gt=0, le=200, description="Maximum number of jobs to return")
):
    """
    List background jobs with optional filtering.
    
    Returns jobs related to profiles, such as bulk judge runs.
    """
    try:
        from ...data_access.job_tracker import get_job_tracker, JobStatus
        
        job_tracker = get_job_tracker()
        
        # Convert status string to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = JobStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        jobs = job_tracker.list_jobs(
            job_type=job_type,
            status=status_enum,
            limit=limit
        )
        
        return {
            "jobs": jobs,
            "count": len(jobs),
            "filters": {
                "job_type": job_type,
                "status": status,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status and details of a specific job.
    
    Returns real-time progress information for background tasks
    like bulk judge runs.
    """
    try:
        from ...data_access.job_tracker import get_job_tracker
        
        job_tracker = get_job_tracker()
        job = job_tracker.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return job
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching job status: {str(e)}")


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running or pending job.
    
    Note: This only marks the job as cancelled. For long-running operations
    that are already in progress, cancellation may not be immediate.
    """
    try:
        from ...data_access.job_tracker import get_job_tracker
        
        job_tracker = get_job_tracker()
        
        if not job_tracker.cancel_job(job_id):
            # Check if job exists
            job = job_tracker.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            else:
                raise HTTPException(status_code=400, detail=f"Job {job_id} cannot be cancelled (status: {job['status']})")
        
        return {"message": f"Job {job_id} has been cancelled"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")


@router.get("/bulk/judge-run/estimate")
async def estimate_bulk_judge_run(
    profile_ids: Optional[str] = Query(None, description="Comma-separated profile IDs"),
    profile_tags: Optional[str] = Query(None, description="Comma-separated profile tags"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    overwrite_existing: bool = Query(False, description="Include papers with existing scores")
):
    """
    Estimate the scope of a bulk judge run without actually running it.
    
    Returns estimates for:
    - Number of profiles that would be processed
    - Number of papers in date range
    - Total scoring operations
    """
    try:
        # Parse profile IDs and tags (no detail= : a bad value propagates to
        # this endpoint's blanket handler, as before)
        profile_id_list = parse_id_csv(profile_ids) or []
        profile_tag_list = parse_tag_csv(None, profile_tags)
        
        # Create mock request for estimation
        mock_request = BulkJudgeRunRequest(
            profile_ids=profile_id_list if profile_id_list else None,
            profile_tags=profile_tag_list if profile_tag_list else None,
            from_date=from_date,
            to_date=to_date,
            overwrite_existing=overwrite_existing
        )
        
        # Get orchestration config
        from ...data_access import SettingsRepository
        import json
        
        orch_json = SettingsRepository.get("orchestration")
        if not orch_json:
            raise HTTPException(
                status_code=500, 
                detail="Orchestration configuration not found"
            )
        
        orch_config = json.loads(orch_json)
        
        # Create runner for estimation
        from ...data_access.bulk_judge import create_bulk_judge_runner
        
        runner = create_bulk_judge_runner(orch_config, verbose=False)
        
        # Get target profiles
        target_profiles = runner._resolve_target_profiles(mock_request)
        
        # Get papers to score
        papers_to_score = runner._get_papers_to_score(target_profiles, mock_request)
        
        # Calculate estimates
        total_operations = len(target_profiles) * len(papers_to_score)
        
        profile_details = []
        for profile in target_profiles:
            interests_count = len(ProfileInterestsRepository.get_by_profile(profile['id']))
            profile_details.append({
                "id": profile['id'],
                "name": profile['name'],
                "interests_count": interests_count
            })
        
        return {
            "estimated_profiles": len(target_profiles),
            "estimated_papers": len(papers_to_score),
            "total_operations": total_operations,
            "date_range": {
                "from_date": from_date,
                "to_date": to_date
            },
            "overwrite_existing": overwrite_existing,
            "profile_details": profile_details
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")


# =====================================================================
# Utility Endpoints
# =====================================================================


def _convert_paper_timestamps(paper_data: dict) -> dict:
    """Convert PostgreSQL datetime objects to ISO strings for API response."""
    return isoformat_fields(paper_data, ('date', 'date_run', 'created_at', 'updated_at'))

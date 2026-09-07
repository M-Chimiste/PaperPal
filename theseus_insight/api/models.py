from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from uuid import UUID

# Basic domain entities
class Model(BaseModel):
    id: str
    name: str
    provider: str

class ModelCreate(BaseModel):
    provider_id: int
    name: str
    config_json: dict

class Paper(BaseModel):
    id: int
    title: str
    abstract: str
    score: float
    date: str  # ISO 8601
    url: str

class Run(BaseModel):
    id: int
    date: str  # ISO 8601
    pipeline_type: str
    status: str
    duration: float
    artifact_path: Optional[str] = None
    artifact_size: Optional[int] = None

# Settings models
class VisualizerSettings(BaseModel):
    resolution: tuple = (1920, 1080)
    fps: int = 30
    matrix_count: int = 200
    matrix_head_color: str = "#e0ffe7"
    matrix_tail_color: str = "0x00b000"
    matrix_char_size: int = 24
    head_step_time: float = 0.25
    random_x_jitter: float = 2.0
    fade_time: float = 3.0
    head_glow_passes: int = 3
    head_glow_alpha_decay: int = 50
    head_spawn_delay_range: tuple = (1.0, 3.0)
    head_saw_period: float = 1.5
    wave_color: str = "#d703fc"
    trail_colors: List[str] = ["#fc03b6", "#ba03fc", "#ce6bf2"]
    glow_passes: int = 3
    glow_alpha_decay: int = 40
    line_width: int = 6

    class Config:
        json_schema_extra = {
            "example": {
                "theme": "dark",
                "font_size": 12,
                "node_color": "#1f77b4",
                "edge_color": "#aec7e8",
                "show_labels": True,
                "layout_type": "force-directed"
            }
        }

class ModelConfig(BaseModel):
    model_name: str
    model_type: str # This refers to the provider name e.g., "ollama", "lmstudio", "openai"
    max_new_tokens: Optional[int] = Field(None, example=2048)
    temperature: Optional[float] = Field(None, example=0.7)
    num_ctx: Optional[int] = Field(None, example=4096) # Context window size
    trust_remote_code: Optional[bool] = Field(None, example=False)

    # Host configuration (supports Ollama, LMStudio, Custom-OAI)
    host: Optional[str] = Field(
        None,
        example="athena.local:11434",
        description="Custom host for Ollama, LMStudio, or Custom-OAI providers. "
                   "Format: 'host:port' or 'http://host:port'. "
                   "Priority: configured host > env variable > provider default. "
                   "Leave empty to use OLLAMA_URL, LMSTUDIO_HOST, or provider defaults."
    )

    # LMStudio-specific fields
    context_length: Optional[int] = Field(None, example=32768, description="Context window for LMStudio")
    gpu_offload: Optional[str] = Field(None, example="max", description="GPU offload: 'max', 'off', or ratio 0-1")

    # Any other model-specific parameters can be added here or in a Dict[str, Any]

class TTSModelConfig(BaseModel):
    tts_provider: str
    tts_model_name: str
    speaker_1_voice: str
    speaker_1_speed: float = Field(..., ge=0.5, le=3.5)
    speaker_2_voice: str
    speaker_2_speed: float = Field(..., ge=0.5, le=3.5)

    class Config:
        json_schema_extra = {
            "example": {
                "tts_provider": "openai",
                "tts_model_name": "tts-1",
                "speaker_1_voice": "sage",
                "speaker_1_speed": 1.0,
                "speaker_2_voice": "ash",
                "speaker_2_speed": 1.0
            }
        }

class MindMapConfig(BaseModel):
    """Configuration for Mind-Map Explorer."""
    k: int = Field(15, ge=5, le=50, description="Number of neighbors to retrieve")
    similarity_threshold: float = Field(0.3, ge=0.1, le=0.95, description="Minimum similarity threshold")
    layout_algorithm: str = Field("force", description="Layout algorithm: force, circular, hierarchical")
    summarization_model: ModelConfig = Field(..., description="Model used for paper summarization")
    expansion_order: int = Field(1, ge=1, le=5, description="Order of expansion (1-5). Higher orders expand from each retrieved paper.")
    max_nodes_per_order: int = Field(20, ge=5, le=50, description="Maximum number of nodes to expand from each paper in multi-order expansion")

class OrchestrationConfig(BaseModel):
    embedding_model: ModelConfig = Field(..., example={"model_name": "Alibaba-NLP/gte-modernbert-base", "model_type": "sentence-transformers", "trust_remote_code": True})
    judge_model: ModelConfig = Field(..., example={"model_name": "phi4-mini:3.8b-q8_0", "model_type": "ollama", "max_new_tokens": 512, "temperature": 0.1, "num_ctx": 4096})
    content_extraction_model: ModelConfig = Field(..., example={"model_name": "gemma3:27b-it-qat", "model_type": "ollama", "max_new_tokens": 4096, "temperature": 0.1, "num_ctx": 131072})
    newsletter_sections_model: ModelConfig = Field(..., example={"model_name": "gemma3:27b-it-qat", "model_type": "ollama", "max_new_tokens": 4096, "temperature": 0.1, "num_ctx": 131072})
    newsletter_intro_model: ModelConfig = Field(..., example={"model_name": "gemini-2.0-flash", "model_type": "gemini", "max_new_tokens": 4096, "temperature": 0.1, "num_ctx": 131072})
    podcast_model: Optional[ModelConfig] = Field(None, example={"model_name": "gemini-2.0-flash", "model_type": "gemini", "max_new_tokens": 8192, "temperature": 0.1, "num_ctx": 131072})
    tts_model: Optional[TTSModelConfig] = Field(None, example={"tts_provider": "openai", "tts_model_name": "tts-1", "speaker_1_voice": "sage", "speaker_1_speed": 1.0, "speaker_2_voice": "ash", "speaker_2_speed": 1.0})
    research_agent_model_config: Optional[ResearchAgentModelConfigApi] = Field(None, description="Research Agent model configuration for automated literature review")  
    mind_map_config: Optional[MindMapConfig] = Field(None, description="Mind-Map Explorer configuration")
    # Add other orchestration settings as needed

class ModelProvider(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

# Pagination
class PaginatedResponse(BaseModel):
    items: List[Any]  # Can be List[Paper], List[Run], etc.
    total_items: int
    total_pages: int
    current_page: int
    nextPage: Optional[int] = None # Kept for compatibility, but might be derived

# Newsletter/Podcast config models
class DateRange(BaseModel):
    from_date: str = Field(..., alias="from")
    to: str

class PodcastConfig(BaseModel):
    scriptModel: str
    ttsModel: str
    addVisualization: bool = False
    visualizerConfig: Optional[Dict] = None

# WebSocket status models
class NodeStatus(BaseModel):
    nodeId: str
    status: str  # 'pending' | 'processing' | 'completed' | 'failed'
    message: str
    progress: float  # 0-100
    timestamp: str  # ISO 8601

class RunStatus(BaseModel):
    """Combined status model used by WebSocket clients."""

    taskId: str
    nodes: List[NodeStatus]
    overallStatus: str  # 'pending' | 'processing' | 'completed' | 'failed'
    currentStep: Optional[str] = None
    progress: Optional[float] = None  # 0-100
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Settings for ArXiv
class ArxivCategoriesConfig(BaseModel):
    main_category: str = Field(..., example="cs")
    filter_categories: List[str] = Field(..., example=["cs.ai", "cs.cl", "cs.lg"])

# Settings for Newsletter Generation
class NewsletterConfig(BaseModel):
    start_date: date
    end_date: date
    research_summary_model: ModelConfig
    title_generation_model: ModelConfig
    intro_generation_model: ModelConfig
    layout_optimization_model: ModelConfig
    # Additional fields like template_path, max_papers, etc.
    max_papers_to_select: int = Field(10, gt=0)
    min_score_threshold: float = Field(0.5, ge=0, le=1)
    search_query: Optional[str] = None
    topic_id: Optional[int] = Field(None, description="Generate newsletter from specific topic papers")
    send_email: bool = False
    intro_music_path: Optional[str] = None # Path to intro music file if any

# If you have other specific setting structures, define them here
# e.g., EmailRecipients, ResearchInterests (if more complex than simple list/string)

class EmailRecipients(BaseModel):
    recipients: List[str] = Field(..., example=["test@example.com", "user@example.org"])

class ResearchInterests(BaseModel):
    interests: str = Field(..., example="Large language models, reinforcement learning, and generative AI.")


# --- Podcast Generation Specific Models --- #

class PodcastVisualizerParams(BaseModel):
    """Detailed parameters for the podcast video visualizer."""
    matrix_count: int
    matrix_head_color: str
    matrix_tail_color: str
    matrix_char_size: int
    head_step_time: float
    random_x_jitter: float
    fade_time: float
    head_glow_passes: int
    head_glow_alpha_decay: int
    head_spawn_delay_range_min: float
    head_spawn_delay_range_max: float
    head_saw_period: float
    line_width: int
    wave_color: str
    trail_color_1: str
    trail_color_2: str
    trail_color_3: str
    glow_passes: int
    glow_alpha_decay: int
    font_path: str
    resolution_width: int
    resolution_height: int
    fps: int

class PodcastGenerationParams(BaseModel):
    """Parameters for initiating a podcast generation task."""
    input_type: str = Field(..., description="Input type, either 'urls' or 'pdfs'.")
    urls: Optional[List[str]] = Field(None, description="List of URLs if input_type is 'urls'.")
    # PDF files will be handled as UploadFile list directly in the endpoint, not part of this JSON body.
    podcast_model_config: ModelConfig
    tts_model_config: TTSModelConfig
    create_visualization: bool
    visualizer_params: Optional[PodcastVisualizerParams] = None
    # Intro music file will also be handled as UploadFile directly in the endpoint.

# New models for Podcast History
class PodcastScriptItem(BaseModel):
    text: str
    speaker: str
    segment_label: Optional[str] = None

class PodcastDetailResponse(BaseModel):
    id: int
    title: str
    date: str
    description: str
    script: List[PodcastScriptItem]

class PodcastListItemResponse(BaseModel):
    id: int
    title: str
    date: str
    description_snippet: str

# --- Pydantic Model for Newsletter Run Pipeline ---
class NewsletterRunParams(BaseModel):
    start_date: str = Field(..., example=date.today().strftime("%Y-%m-%d"))
    end_date: str = Field(..., example=(date.today() - timedelta(days=6)).strftime("%Y-%m-%d"))
    email_recipients: List[str] = Field(default_factory=list, example=["test@example.com"])
    research_interests: str = Field(..., example="AI in healthcare")
    topic_id: Optional[int] = Field(None, description="Generate newsletter from specific topic papers (overrides research_interests filtering)")
    generate_podcast_run: bool = Field(False, description="Whether to generate a podcast as part of this run.")
    
    # Profile filtering parameters
    profile_id: Optional[int] = Field(None, description="Generate newsletter for specific profile")
    profile_ids: Optional[List[int]] = Field(None, description="Generate newsletter for multiple profiles")
    profile_tag: Optional[str] = Field(None, description="Generate newsletter for profiles with specific tag")
    profile_tags: Optional[List[str]] = Field(None, description="Generate newsletter for profiles with any of the tags")
    use_profile_recipients: bool = Field(False, description="Use profile-specific email recipients instead of email_recipients")

    # Multi-server judge configuration
    use_multi_server_judge: bool = Field(False, description="Use multi-server worker pool for LLM judge scoring")
    judge_server_ids: Optional[List[int]] = Field(None, description="List of inference server IDs to use for multi-server judge scoring")
    judge_request_timeout_sec: Optional[int] = Field(None, description="Timeout in seconds for judge LLM requests")
    judge_max_retries: Optional[int] = Field(None, description="Maximum number of retries for failed judge tasks")

    # Newsletter section configuration
    num_sections: Optional[int] = Field(5, ge=1, le=15, description="Number of newsletter sections to generate")

# --- Pydantic Model for Profile-Specific Newsletter Generation ---
class ProfileNewsletterRequest(BaseModel):
    start_date: str = Field(..., example=date.today().strftime("%Y-%m-%d"))
    end_date: str = Field(..., example=(date.today() - timedelta(days=6)).strftime("%Y-%m-%d"))
    email_recipients: Optional[List[str]] = Field(default=None, example=["test@example.com"], description="Email recipients (if not provided, uses profile's configured recipients)")
    research_interests: Optional[str] = Field(None, example="AI in healthcare", description="Research interests (if not provided, uses profile's research interests)")
    topic_id: Optional[int] = Field(None, description="Generate newsletter from specific topic papers (overrides research_interests filtering)")
    generate_podcast_run: bool = Field(False, description="Whether to generate a podcast as part of this run.")
    use_profile_recipients: bool = Field(True, description="Use profile-specific email recipients (default: true for profile-specific requests)")

class ProfileNewsletterResponse(BaseModel):
    task_id: str = Field(..., description="Task ID for tracking newsletter generation progress")
    message: str = Field(..., description="Success message")
    profile_id: int = Field(..., description="Profile ID for which newsletter is being generated")
    profile_name: str = Field(..., description="Name of the profile")

# Models for Papers Page
class PaperApiResponse(BaseModel):
    id: int
    title: str
    abstract: str
    date: str
    date_run: str
    score: Optional[float] = Field(default=None, description="Legacy paper score (use profile_score for profile-specific scores)")
    rationale: Optional[str] = Field(default=None, description="Legacy rationale (use profile rationale for profile-specific rationales)")
    related: Optional[bool] = Field(default=False, description="Legacy related flag")
    cosine_similarity: Optional[float] = Field(default=None, description="Cosine similarity score")
    url: str
    embedding_model: str
    keywords: Optional[List[str]] = Field(default=None, description="Top keywords extracted from title/abstract")
    similarity_score: Optional[float] = Field(default=None, description="Semantic similarity score when returned from similarity search")
    profile_score: Optional[float] = Field(default=None, description="Profile-specific relevance score")
    semantic_score: Optional[float] = Field(default=None, description="Semantic similarity score in hybrid search")
    keyword_score: Optional[float] = Field(default=None, description="Keyword matching score in hybrid search")
    hybrid_score: Optional[float] = Field(default=None, description="Combined hybrid search score")

class PaginatedPapersResponse(PaginatedResponse): # Inherit from existing PaginatedResponse
    items: List[PaperApiResponse]
    # total_items, total_pages, current_page, nextPage are inherited

class SimilaritySearchResponse(BaseModel):
    query_text: str
    results: List[PaperApiResponse]
    total_results: int

class PaperUpdateRequest(BaseModel):
    """Request payload for updating a paper's editable fields."""
    score: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="New score (0-10)")
    related: Optional[bool] = Field(default=None, description="Whether the paper is considered relevant")
    profile_ids: Optional[List[int]] = Field(default=None, description="If provided, update profile-specific score/related for these profiles instead of base paper")

class SimilaritySearchRequest(BaseModel):
    query_text: str = Field(..., description="Text to search for semantically similar papers")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score (0-1)")

class SimilarPapersRequest(BaseModel):
    limit: Optional[int] = Field(10, description="Maximum number of similar papers to return")
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score (0-1)")

class SimilarPapersResponse(BaseModel):
    reference_paper: PaperApiResponse
    similar_papers: List[PaperApiResponse]
    total_similar: int

class HybridSearchRequest(BaseModel):
    profile_ids: Optional[List[int]] = None
    min_profile_score: Optional[float] = Field(None, ge=0, le=10)
    max_profile_score: Optional[float] = Field(None, ge=0, le=10)
    profile_related: Optional[bool] = None
    query_text: str = Field(..., description="Search query text")
    page: int = Field(1, gt=0, description="Page number")
    page_size: int = Field(10, gt=0, le=100, description="Number of results per page")
    semantic_weight: float = Field(0.6, ge=0.0, le=1.0, description="Weight for semantic similarity")
    keyword_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for keyword matching")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum semantic similarity threshold")
    min_score: Optional[float] = Field(None, description="Minimum paper score filter")
    max_score: Optional[float] = Field(None, description="Maximum paper score filter")
    from_date: Optional[str] = Field(None, description="Start date filter (YYYY-MM-DD)")
    to_date: Optional[str] = Field(None, description="End date filter (YYYY-MM-DD)")

class HybridSearchResponse(BaseModel):
    candidate_limit: int = 500
    count_scope: str = "retrieved_candidates"
    query_text: str
    results: List[PaperApiResponse]
    total_results: int
    total_pages: int
    current_page: int
    semantic_weight: float
    keyword_weight: float

# Research Agent Model Configuration
class ResearchAgentModelConfigApi(BaseModel):
    """API model for Research Agent model configuration (FR-16, FR-17)."""
    boss_model: ModelConfig = Field(..., description="Main coordinating model")
    worker_models: Dict[str, ModelConfig] = Field(
        default_factory=dict, 
        description="Task-specific worker models (summary, analysis, search)"
    )
    default_worker: str = Field("summary", description="Default worker model role")
    max_retries: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
    
    # LangGraph workflow configuration
    max_research_context_tokens: int = Field(15000, ge=1000, le=100000, description="Budget before compression")
    compress_to_ratio: float = Field(0.2, ge=0.1, le=0.8, description="Target ratio for scratch-pad compression")
    initial_rerank_top_k: int = Field(40, ge=5, le=100, description="Input to cross-encoder re-ranking")
    max_research_loops: int = Field(3, ge=1, le=10, description="Maximum research iteration loops")
    
    # Node-specific model configurations
    query_planner_model: Optional[ModelConfig] = Field(None, description="Model for query decomposition")
    evidence_selector_model: Optional[ModelConfig] = Field(None, description="Model for evidence sufficiency evaluation")
    compression_model: Optional[ModelConfig] = Field(None, description="Model for context compression")
    answer_generator_model: Optional[ModelConfig] = Field(None, description="Model for final report generation")
    
    # External API configuration
    arxiv_rate_limit: float = Field(3.0, ge=0.5, le=10.0, description="ArXiv API requests per second")

    class Config:
        json_schema_extra = {
            "example": {
                "boss_model": {
                    "model_name": "gemini-2.0-flash",
                    "model_type": "gemini",
                    "max_new_tokens": 4096,
                    "temperature": 0.1,
                    "num_ctx": 131072
                },
                "worker_models": {
                    "summary": {
                        "model_name": "gemma3:27b-it-qat",
                        "model_type": "ollama",
                        "max_new_tokens": 4096,
                        "temperature": 0.1,
                        "num_ctx": 131072
                    },
                    "analysis": {
                        "model_name": "phi4-mini:3.8b-q8_0",
                        "model_type": "ollama",
                        "max_new_tokens": 2048,
                        "temperature": 0.1,
                        "num_ctx": 4096
                    }
                },
                "default_worker": "summary",
                "max_retries": 3,
                "max_research_context_tokens": 15000,
                "compress_to_ratio": 0.2,
                "initial_rerank_top_k": 40,
                "max_research_loops": 3,
                "arxiv_rate_limit": 3.0,
            }
        }

# Research Agent Request/Response models
class ResearchAgentRunRequest(BaseModel):
    """Request model for starting a research agent run (FR-18)."""
    research_question: str = Field(..., min_length=10, description="Research question to investigate")
    num_papers_target: int = Field(5, ge=1, le=20, description="Target number of papers to collect")
    max_steps: int = Field(10, ge=1, le=50, description="Maximum agent iteration steps")
    model_config_override: Optional[ResearchAgentModelConfigApi] = Field(
        None, description="Optional model configuration override"
    )

class ResearchAgentRunResponse(BaseModel):
    """Response model for research agent run initiation."""
    task_id: str = Field(..., description="Unique task identifier")
    message: str = Field(..., description="Status message")

# Literature Review result models
class LiteratureReviewSummary(BaseModel):
    """Summary of a literature review result."""
    paper_id: int
    title: str
    summary: str
    rationale: str
    relevance_score: float

class LiteratureReviewResult(BaseModel):
    """Complete literature review result."""
    id: int
    research_question: str
    summaries: List[LiteratureReviewSummary]
    created_ts: str
    total_papers: int
    trace_log: List[Dict[str, Any]] = Field(default_factory=list)
    report_text: Optional[str] = Field(None, description="Full markdown report generated by the research agent")

# Model Catalog models
class ModelCatalogEntry(BaseModel):
    """Model catalog entry."""
    id: int
    alias: str = Field(..., description="Display name for the model")
    model_string: str = Field(..., description="Model identifier string")
    provider_name: str = Field(..., description="Provider name (e.g., openai, ollama)")
    model_type: str = Field(..., description="Model type (e.g., chat, completion, embedding)")
    description: Optional[str] = Field(None, description="Model description (markdown supported)")
    max_new_tokens: Optional[int] = Field(None, description="Maximum new tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature setting")
    num_ctx: Optional[int] = Field(None, description="Context window size")
    trust_remote_code: Optional[bool] = Field(False, description="Whether to trust remote code")
    tags: Optional[List[str]] = Field(default_factory=list, description="Model tags")
    is_favorite: Optional[bool] = Field(False, description="Whether model is marked as favorite")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")

class ModelCatalogCreateRequest(BaseModel):
    """Request model for creating a model catalog entry."""
    alias: str = Field(..., description="Display name for the model")
    model_string: str = Field(..., description="Model identifier string") 
    provider_name: str = Field(..., description="Provider name")
    model_type: str = Field(..., description="Model type")
    description: Optional[str] = Field("", description="Model description")
    max_new_tokens: Optional[int] = Field(None, description="Maximum new tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature setting")
    num_ctx: Optional[int] = Field(None, description="Context window size")
    trust_remote_code: bool = Field(False, description="Whether to trust remote code")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    is_favorite: bool = Field(False, description="Whether model is marked as favorite")

class ModelCatalogUpdateRequest(BaseModel):
    """Request model for updating a model catalog entry."""
    alias: Optional[str] = Field(None, description="Display name for the model")
    model_string: Optional[str] = Field(None, description="Model identifier string")
    provider_name: Optional[str] = Field(None, description="Provider name")
    model_type: Optional[str] = Field(None, description="Model type")
    description: Optional[str] = Field(None, description="Model description")
    max_new_tokens: Optional[int] = Field(None, description="Maximum new tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature setting")
    num_ctx: Optional[int] = Field(None, description="Context window size")
    trust_remote_code: Optional[bool] = Field(None, description="Whether to trust remote code")
    tags: Optional[List[str]] = Field(None, description="Model tags")
    is_favorite: Optional[bool] = Field(None, description="Whether model is marked as favorite")

class ModelCatalogSearchRequest(BaseModel):
    """Request model for searching model catalog."""
    search: Optional[str] = Field(None, description="Search query")
    provider: Optional[str] = Field(None, description="Filter by provider")
    model_type: Optional[str] = Field(None, description="Filter by model type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    is_favorite: Optional[bool] = Field(None, description="Filter by favorite status")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Page size")

class ModelCatalogSearchResponse(BaseModel):
    """Response model for model catalog search."""
    models: List[ModelCatalogEntry] = Field(..., description="List of models")
    total_count: int = Field(..., description="Total number of models")
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")

# Mind-Map specific models
class MindMapNode(BaseModel):
    """A node in the mind-map visualization."""
    id: str = Field(..., description="Unique node identifier (paper_id)")
    title: str = Field(..., description="Paper title")
    summary: str = Field(..., description="LLM-generated summary")
    similarity_score: float = Field(..., description="Similarity score to seed paper")
    position: Dict[str, float] = Field(..., description="Node position {x, y}")
    is_seed: bool = Field(False, description="Whether this is the seed paper")
    has_fulltext: bool = Field(False, description="Whether full-text is available")
    url: Optional[str] = Field(None, description="Paper URL")
    date: Optional[str] = Field(None, description="Publication date")
    keywords: Optional[List[str]] = Field(None, description="Top keywords extracted from the paper title/abstract")

class MindMapEdge(BaseModel):
    """An edge connecting two nodes in the mind-map."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    similarity_score: float = Field(..., description="Similarity score between papers")

class MindMapData(BaseModel):
    """Complete mind-map data structure."""
    nodes: List[MindMapNode] = Field(..., description="List of nodes")
    edges: List[MindMapEdge] = Field(..., description="List of edges")
    seed_paper_id: str = Field(..., description="ID of the seed paper")
    layout_algorithm: str = Field(..., description="Layout algorithm used")
    generation_timestamp: str = Field(..., description="When the mind-map was generated")

class MindMapExpandRequest(BaseModel):
    """Request model for expanding a mind-map."""
    paper_id: Optional[str] = Field(None, description="Seed paper ID (required if topic_id not provided)")
    topic_id: Optional[int] = Field(None, description="Seed from topic's representative papers (required if paper_id not provided)")
    k: int = Field(15, ge=1, le=50, description="Number of similar papers to retrieve")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")
    layout_algorithm: str = Field("force", description="Layout algorithm: force, circular, hierarchical")
    model_config_override: Optional[ModelConfig] = Field(None, description="Override default summarization model")
    expansion_order: int = Field(1, ge=1, le=5, description="Order of expansion (1-5). Higher orders expand from each retrieved paper.")
    max_nodes_per_order: int = Field(20, ge=5, le=50, description="Maximum number of nodes to expand from each paper in multi-order expansion")
    # Profile filtering parameters
    profile_id: Optional[int] = Field(None, description="Filter papers to specific profile ID")
    profile_ids: Optional[List[int]] = Field(None, description="Filter papers to multiple profile IDs")
    profile_tag: Optional[str] = Field(None, description="Filter papers to profiles with specific tag")
    profile_tags: Optional[List[str]] = Field(None, description="Filter papers to profiles with any of these tags")
    
    @model_validator(mode='after')
    def validate_seed_input(self):
        """Ensure either paper_id or topic_id is provided, but not both."""
        if not self.paper_id and not self.topic_id:
            raise ValueError('Either paper_id or topic_id must be provided')
        
        if self.paper_id and self.topic_id:
            raise ValueError('Cannot specify both paper_id and topic_id - choose one')
        
        return self

class MindMapExpandResponse(BaseModel):
    """Response model for mind-map expansion."""
    task_id: str = Field(..., description="Task ID for tracking progress")
    message: str = Field(..., description="Status message")

class PDFParseRequest(BaseModel):
    """Request model for parsing PDFs on-demand."""
    paper_ids: List[str] = Field(..., max_items=20, description="List of paper IDs to parse (max 20)")

class PDFParseResponse(BaseModel):
    """Response model for PDF parsing."""
    task_id: str = Field(..., description="Task ID for tracking progress")
    message: str = Field(..., description="Status message")
    papers_to_parse: int = Field(..., description="Number of papers queued for parsing")

class MindMapSeedSearchRequest(BaseModel):
    """Request model for searching papers to use as mind-map seeds."""
    query: str = Field(..., min_length=3, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")

class MindMapSeedSearchResponse(BaseModel):
    """Response model for seed paper search."""
    query: str = Field(..., description="Search query used")
    results: List[PaperApiResponse] = Field(..., description="List of matching papers")
    total_results: int = Field(..., description="Total number of results")

# Mind-Map Report models for saving and loading mind-maps
class MindMapReport(BaseModel):
    """A saved mind-map report."""
    id: int = Field(..., description="Unique report ID")
    title: str = Field(..., description="Report title")
    description: Optional[str] = Field(None, description="Optional report description")
    seed_paper_id: int = Field(..., description="ID of the seed paper")
    seed_paper_title: str = Field(..., description="Title of the seed paper")
    parameters: Dict[str, Any] = Field(..., description="Generation parameters used")
    mindmap_data: Dict[str, Any] = Field(..., description="Complete mind-map data")
    statistics: Dict[str, Any] = Field(..., description="Generation statistics")
    created_at: str = Field(..., description="Creation timestamp")

class MindMapReportSaveRequest(BaseModel):
    """Request to save a mind-map as a report."""
    title: str = Field(..., min_length=1, max_length=200, description="Report title (required)")
    description: Optional[str] = Field(None, max_length=1000, description="Optional description")
    mindmap_data: Dict[str, Any] = Field(..., description="Complete mind-map data to save")
    parameters: Dict[str, Any] = Field(..., description="Generation parameters")

class MindMapReportListResponse(BaseModel):
    """Response with list of saved mind-map reports."""
    reports: List[MindMapReport] = Field(..., description="List of saved reports")
    total_count: int = Field(..., description="Total number of reports")

class MindMapReportSaveResponse(BaseModel):
    """Response when saving a mind-map report."""
    id: int = Field(..., description="Saved report ID")
    title: str = Field(..., description="Report title")
    message: str = Field(..., description="Success message")

# === Topic Evolution & Trends Models ===

class TopicApiResponse(BaseModel):
    """API response model for a topic."""
    id: int = Field(..., description="Topic ID")
    label: str = Field(..., description="Topic label")
    keywords: List[str] = Field(..., description="Top keywords for the topic")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    
    # Latest metrics (if available)
    latest_doc_count: Optional[int] = Field(None, description="Most recent document count")
    latest_growth_rate: Optional[float] = Field(None, description="Most recent growth rate")
    total_papers: Optional[int] = Field(None, description="Total papers associated with topic")
    
    # Forecasts (if available)
    forecast_1m: Optional[int] = Field(None, description="1-month forecast")
    forecast_3m: Optional[int] = Field(None, description="3-month forecast")
    forecast_6m: Optional[int] = Field(None, description="6-month forecast")

class TopicMetricResponse(BaseModel):
    """API response model for topic metrics over time."""
    id: int = Field(..., description="Metric ID")
    topic_id: int = Field(..., description="Topic ID")
    period_start: str = Field(..., description="Period start date")
    period_end: str = Field(..., description="Period end date")
    period_type: str = Field(..., description="Period type (week/month/quarter)")
    doc_count: int = Field(..., description="Number of documents in this period")
    avg_score: Optional[float] = Field(None, description="Average score of papers")
    growth_rate: Optional[float] = Field(None, description="Growth rate compared to previous period")
    forecast_1m: Optional[int] = Field(None, description="1-month forecast")
    forecast_3m: Optional[int] = Field(None, description="3-month forecast")
    forecast_6m: Optional[int] = Field(None, description="6-month forecast")
    created_at: str = Field(..., description="Creation timestamp")

class TopicDetailResponse(BaseModel):
    """Detailed response for a specific topic including timeline and papers."""
    topic: TopicApiResponse = Field(..., description="Topic information")
    timeline: List[TopicMetricResponse] = Field(..., description="Historical metrics")
    representative_papers: List[PaperApiResponse] = Field(..., description="Most relevant papers")
    total_papers: int = Field(..., description="Total papers for this topic")

class TrendsListRequest(BaseModel):
    """Request model for listing trending topics."""
    limit: int = Field(20, ge=1, le=100, description="Maximum number of topics to return")
    period_type: str = Field("month", description="Display granularity: week, month, quarter")
    duration_months: int = Field(6, ge=1, le=24, description="Duration to analyze: 1, 3, 6, 12, 24 months")
    min_doc_count: int = Field(5, ge=1, description="Minimum document count filter")
    sort_by: str = Field("growth_rate", description="Sort by: growth_rate, doc_count, forecast_3m")

class TrendsListResponse(BaseModel):
    """Response model for trending topics list."""
    topics: List[TopicApiResponse] = Field(..., description="List of trending topics")
    total_topics: int = Field(..., description="Total number of topics in system")
    total_papers_with_topics: int = Field(..., description="Total papers with topic assignments")
    period_type: str = Field(..., description="Display granularity used")
    duration_months: int = Field(..., description="Duration analyzed")

class TrendsSearchRequest(BaseModel):
    """Request model for searching topics."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")

class TrendsSearchResponse(BaseModel):
    """Response model for topic search."""
    query: str = Field(..., description="Search query used")
    topics: List[TopicApiResponse] = Field(..., description="Matching topics")
    total_results: int = Field(..., description="Total number of results")

class TrendsRecomputeRequest(BaseModel):
    """Request model for trends recomputation with weekly-first analysis."""
    lookback_months: int = Field(24, ge=1, le=48, description="Months of historical data to analyze")
    duration_months: int = Field(6, ge=1, le=24, description="Duration for analysis views (1, 3, 6, 12, 24 months)")
    min_papers: int = Field(100, ge=10, description="Minimum papers required to run analysis")
    validate_accuracy: bool = Field(True, description="Whether to validate forecast accuracy")
    force_full_recalc: bool = Field(False, description="Force full recalculation instead of incremental processing")
    clear_all_data: bool = Field(False, description="NUCLEAR OPTION: Clear all topics, metrics, and relationships before recalculation (for development/testing)")

class TrendsRecomputeResponse(BaseModel):
    """Response model for trends recomputation."""
    task_id: str = Field(..., description="Task ID for tracking progress")
    message: str = Field(..., description="Status message")
    estimated_duration_minutes: int = Field(..., description="Estimated processing time")

class TopicPapersRequest(BaseModel):
    """Request model for getting papers for a topic."""
    limit: int = Field(50, ge=1, le=200, description="Maximum number of papers")
    min_relevance: float = Field(0.1, ge=0.0, le=1.0, description="Minimum relevance score")
    sort_by: str = Field("relevance", description="Sort by: relevance, score, date")

class TopicPapersResponse(BaseModel):
    """Response model for topic papers."""
    topic_id: int = Field(..., description="Topic ID")
    topic_label: str = Field(..., description="Topic label")
    papers: List[PaperApiResponse] = Field(..., description="Papers for this topic")
    total_papers: int = Field(..., description="Total papers for this topic")

class TrendsValidateAccuracyRequest(BaseModel):
    """Request model for forecast accuracy validation."""
    period_type: str = Field(default="month", description="Period type for validation")

# Performance Configuration Models
class PerformanceConfig(BaseModel):
    """Performance configuration for computationally intensive operations."""
    
    # Hardware Resources
    max_cores: int = Field(default=4, ge=1, le=128, description="Maximum CPU cores to use")
    max_memory_gb: int = Field(default=16, ge=4, le=2048, description="Maximum memory in GB")
    
    # HDBSCAN Clustering Optimization
    hdbscan_n_jobs: int = Field(default=-1, ge=-1, le=128, description="HDBSCAN parallel jobs (-1 = use all cores)")
    clustering_batch_size: int = Field(default=50000, ge=1000, le=1000000, description="Batch size for clustering operations")
    
    # Embedding & Vector Operations
    embedding_batch_size: int = Field(default=512, ge=32, le=2048, description="Embedding batch size (used when auto-tuning is disabled)")
    auto_tune_batch_size: bool = Field(default=True, description="Auto-tune embedding batch size for the hardware on first run")
    vector_processing_workers: int = Field(default=8, ge=1, le=64, description="Workers for vector processing")
    
    # Memory Management
    enable_memory_mapping: bool = Field(default=True, description="Use memory mapping for large datasets")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings in memory")
    aggressive_garbage_collection: bool = Field(default=False, description="Force garbage collection between stages")
    
    # Development vs Production Mode
    development_mode: bool = Field(default=False, description="Enable development optimizations (smaller datasets, faster iterations)")
    development_max_papers: int = Field(default=5000, ge=100, le=50000, description="Max papers in development mode")
    
    class Config:
        json_schema_extra = {
            "example": {
                "max_cores": 16,
                "max_memory_gb": 128,
                "hdbscan_n_jobs": -1,
                "clustering_batch_size": 100000,
                "embedding_batch_size": 1024,
                "auto_tune_batch_size": True,
                "vector_processing_workers": 16,
                "enable_memory_mapping": True,
                "cache_embeddings": True,
                "aggressive_garbage_collection": False,
                "development_mode": False,
                "development_max_papers": 5000
            }
        }

class SystemInfoResponse(BaseModel):
    """System hardware information for performance configuration."""
    cpu_count_physical: int = Field(description="Physical CPU cores")
    cpu_count_logical: int = Field(description="Logical CPU cores (with hyperthreading)")
    memory_total_gb: float = Field(description="Total system memory in GB")
    memory_available_gb: float = Field(description="Available system memory in GB")
    gpu_available: bool = Field(description="GPU acceleration available")
    gpu_name: Optional[str] = Field(default=None, description="GPU name if available")
    recommended_config: PerformanceConfig = Field(description="Recommended performance configuration")

# === Research Interest Clustering API Models ===
# Separate from automatic topic discovery, these handle research interest based analysis

class ResearchInterestApiResponse(BaseModel):
    """Response model for research interest data."""
    id: int
    interest_text: str
    embedding_model: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    latest_doc_count: Optional[int] = 0
    latest_growth_rate: Optional[float] = None
    total_papers: int = 0
    latest_avg_relevance: Optional[float] = None
    latest_avg_score: Optional[float] = None
    forecast_1m: Optional[int] = None
    forecast_3m: Optional[int] = None
    forecast_6m: Optional[int] = None


class ResearchInterestMetricResponse(BaseModel):
    """Response model for research interest metrics."""
    id: int
    research_interest_id: int
    period_start: str
    period_end: str
    period_type: str
    doc_count: int
    avg_relevance_score: Optional[float] = None
    avg_paper_score: Optional[float] = None
    growth_rate: Optional[float] = None
    forecast_1m: Optional[int] = None
    forecast_3m: Optional[int] = None
    forecast_6m: Optional[int] = None
    created_at: str


class ResearchInterestDetailResponse(BaseModel):
    """Response model for detailed research interest information."""
    interest: ResearchInterestApiResponse
    timeline: List[ResearchInterestMetricResponse]
    representative_papers: List[PaperApiResponse]
    total_papers: int


class ResearchInterestsListRequest(BaseModel):
    """Request model for listing research interests."""
    limit: int = Field(20, ge=1, le=100, description="Maximum number of interests to return")
    period_type: str = Field("week", description="Display granularity: week, month, quarter")
    duration_months: int = Field(6, ge=1, le=24, description="Duration to analyze: 1, 3, 6, 12, 24 months")
    min_doc_count: int = Field(1, ge=1, description="Minimum document count filter")
    sort_by: str = Field("growth_rate", description="Sort by: growth_rate, doc_count, avg_relevance, forecast_3m")


class ResearchInterestsListResponse(BaseModel):
    """Response model for research interests list."""
    interests: List[ResearchInterestApiResponse]
    total_interests: int
    total_papers_with_interests: int
    period_type: str
    duration_months: int


class ResearchInterestsSearchRequest(BaseModel):
    """Request model for searching research interests."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")


class ResearchInterestsSearchResponse(BaseModel):
    """Response model for research interest search results."""
    query: str
    interests: List[ResearchInterestApiResponse]
    total_results: int


class ResearchInterestRecomputeRequest(BaseModel):
    """Request model for research interest recomputation."""
    lookback_months: int = Field(24, ge=1, le=60, description="Historical data window (months)")
    duration_months: int = Field(6, ge=1, le=24, description="Analysis duration (1, 3, 6, 12, 24 months)")
    min_papers: int = Field(100, ge=50, le=10000, description="Minimum papers required")
    similarity_threshold: float = Field(0.3, ge=0.1, le=1.0, description="Minimum similarity threshold")
    clear_all_data: bool = Field(False, description="NUCLEAR: Clear all research interest data")


class ResearchInterestRecomputeResponse(BaseModel):
    """Response model for research interest recomputation."""
    task_id: str
    message: str
    estimated_duration_minutes: int


class ResearchInterestPapersRequest(BaseModel):
    """Request model for getting papers for a research interest."""
    limit: int = Field(50, ge=1, le=200, description="Maximum number of papers")
    min_similarity: float = Field(0.1, ge=0.0, le=1.0, description="Minimum similarity score")
    sort_by: str = Field("similarity", description="Sort by: similarity, score, date")


class ResearchInterestPapersResponse(BaseModel):
    """Response model for research interest papers."""
    research_interest_id: int
    interest_text: str
    papers: List[PaperApiResponse]
    total_papers: int


# =====================================================================
# Research Profiles Models
# =====================================================================

class ProfileResponse(BaseModel):
    """Response model for research profiles."""
    id: int = Field(..., description="Profile ID")
    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    color: Optional[str] = Field(None, description="UI color coding")
    tags: Optional[List[str]] = Field(None, description="Organization tags")
    email_recipients: Optional[List[str]] = Field(None, description="Email distribution list")
    arxiv_filters: Optional[Dict[str, Any]] = Field(None, description="ArXiv category filters")
    is_active: bool = Field(True, description="Whether profile is active")
    is_default: bool = Field(False, description="Whether this is the default profile")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    total_papers: Optional[int] = Field(None, description="Total papers scored for this profile")


class ProfileWithStatsResponse(ProfileResponse):
    """Profile response with statistics."""
    interest_count: int = Field(0, description="Number of research interests")
    total_scored_papers: int = Field(0, description="Total papers scored for this profile")
    relevant_papers: int = Field(0, description="Papers marked as relevant")
    average_score: Optional[float] = Field(None, description="Average relevance score")
    research_interests: List[str] = Field(default_factory=list, description="List of research interest texts")


class ProfileCreateRequest(BaseModel):
    """Request model for creating a profile."""
    name: str = Field(..., min_length=1, max_length=100, description="Profile name")
    description: Optional[str] = Field(None, max_length=500, description="Profile description")
    color: Optional[str] = Field(None, description="UI color coding")
    tags: Optional[List[str]] = Field(None, description="Organization tags")
    email_recipients: Optional[List[str]] = Field(None, description="Email distribution list")
    arxiv_filters: Optional[Dict[str, Any]] = Field(None, description="ArXiv category filters")
    research_interests: Optional[List[str]] = Field(None, description="Initial research interests")


class ProfileUpdateRequest(BaseModel):
    """Request model for updating a profile."""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Profile name")
    description: Optional[str] = Field(None, max_length=500, description="Profile description")
    color: Optional[str] = Field(None, description="UI color coding")
    tags: Optional[List[str]] = Field(None, description="Organization tags")
    email_recipients: Optional[List[str]] = Field(None, description="Email distribution list")
    arxiv_filters: Optional[Dict[str, Any]] = Field(None, description="ArXiv category filters")
    is_active: Optional[bool] = Field(None, description="Whether profile is active")
    research_interests: Optional[List[str]] = Field(None, description="Research interests")


class ProfileTagSearchResponse(BaseModel):
    """Response model for tag search/auto-complete."""
    query: str = Field(..., description="Search query")
    suggestions: List[Dict[str, Any]] = Field(..., description="Tag suggestions with usage counts")
    exact_match: bool = Field(..., description="Whether an exact match was found")


class ProfileInterestResponse(BaseModel):
    """Response model for profile research interests."""
    id: int = Field(..., description="Interest ID")
    interest_text: str = Field(..., description="Interest text")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class ProfileInterestCreateRequest(BaseModel):
    """Request model for creating a research interest."""
    interest_text: str = Field(..., min_length=1, max_length=500, description="Research interest text")


class BulkJudgeRunRequest(BaseModel):
    """Request model for bulk LLM judge runs across profiles."""
    profile_ids: Optional[List[int]] = Field(None, description="Specific profile IDs to process")
    profile_tags: Optional[List[str]] = Field(None, description="Process profiles with these tags")
    from_date: Optional[str] = Field(None, description="Start date for paper filtering (YYYY-MM-DD)")
    to_date: Optional[str] = Field(None, description="End date for paper filtering (YYYY-MM-DD)")
    batch_size: int = Field(100, ge=10, le=1000, description="Papers to process per batch")
    overwrite_existing: bool = Field(False, description="Overwrite existing scores")
    paper_ids: Optional[List[int]] = Field(None, description="Specific paper IDs to process (overrides date range)")


class BulkJudgeRunResponse(BaseModel):
    """Response model for bulk judge run initiation."""
    job_id: str = Field(..., description="Job tracking ID")
    status: str = Field(..., description="Job status")
    profile_count: int = Field(..., description="Number of profiles to process")
    estimated_papers: int = Field(..., description="Estimated papers to score")
    message: str = Field(..., description="Status message")


class ProfileAwareIngestRequest(BaseModel):
    """Request model for profile-aware paper ingestion."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    profile_ids: Optional[List[int]] = None
    profile_tags: Optional[List[str]] = None
    score_all_profiles: bool = False
    overwrite_existing: bool = False
    cosine_threshold: float = 0.5
    arxiv_categories: Optional[List[str]] = None
    use_profile_arxiv_filters: bool = True  # Use arXiv filters from selected profiles
    batch_size: int = 10
    send_error_notifications: bool = False
    # Multi-server configuration (only used when LLM-as-Judge uses Ollama)
    use_multi_server: bool = False
    server_ids: Optional[List[int]] = None

class ProfileAwareIngestResponse(BaseModel):
    """Response model for profile-aware paper ingestion."""
    task_id: str
    message: str
    profile_count: int
    estimated_papers: int
    status: str

# Job monitoring models
class JobStatus(BaseModel):
    """Enum-like class for job statuses."""
    PENDING: str = "pending"
    RUNNING: str = "running"
    COMPLETED: str = "completed"
    FAILED: str = "failed"
    CANCELLED: str = "cancelled"

class JobResponse(BaseModel):
    """Response model for job details."""
    id: UUID = Field(..., description="Job ID")
    job_type: str = Field(..., description="Type of job")
    status: str = Field(..., description="Current job status")
    configuration: Dict[str, Any] = Field(..., description="Job configuration")
    state: Optional[Dict[str, Any]] = Field(None, description="Current job state")
    progress_current: int = Field(0, description="Current progress count")
    progress_total: Optional[int] = Field(None, description="Total items to process")
    progress_percent: float = Field(0.0, description="Progress percentage")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_count: int = Field(0, description="Number of errors")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    last_checkpoint_at: Optional[datetime] = Field(None, description="Last checkpoint time")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")

class JobListResponse(BaseModel):
    """Response model for paginated job list."""
    jobs: List[JobResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")

class JobStatisticsResponse(BaseModel):
    """Response model for job statistics."""
    statistics: List[Dict[str, Any]] = Field(..., description="Statistics by job type")
    overall: Dict[str, Any] = Field(..., description="Overall statistics")


# =====================================================================
# Research Timeline Visualization Models
# =====================================================================

class TimelineKeyPaper(BaseModel):
    """Simplified paper model for timeline key papers."""
    id: int = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    date: str = Field(..., description="Publication date")
    score: Optional[float] = Field(None, description="Relevance score")
    relevance_score: Optional[float] = Field(None, description="Topic relevance score")


class TimelinePeriodData(BaseModel):
    """Data for a single time period in the timeline."""
    period_start: str = Field(..., description="Period start date (ISO format)")
    period_end: str = Field(..., description="Period end date (ISO format)")
    period_type: str = Field(..., description="Period granularity: week, month, quarter, year")
    doc_count: int = Field(..., description="Number of papers in this period")
    growth_rate: Optional[float] = Field(None, description="Growth rate vs previous period")
    phase: str = Field(..., description="Growth phase: emerging, growth, stable, declining")
    key_papers: Optional[List[TimelineKeyPaper]] = Field(None, description="Top papers for this period")
    forecast_1m: Optional[int] = Field(None, description="1-period forecast")
    forecast_3m: Optional[int] = Field(None, description="3-period forecast")
    forecast_6m: Optional[int] = Field(None, description="6-period forecast")
    is_forecast: bool = Field(False, description="Whether this is a forecast period")


class TopicTimelineData(BaseModel):
    """Timeline data for a single topic."""
    topic_id: int = Field(..., description="Topic ID")
    topic_label: str = Field(..., description="Topic label/name")
    keywords: List[str] = Field(..., description="Topic keywords")
    total_papers: int = Field(..., description="Total papers for this topic")
    periods: List[TimelinePeriodData] = Field(..., description="Time periods with data")


class TimelineDataRequest(BaseModel):
    """Request model for timeline data."""
    topic_ids: Optional[List[int]] = Field(None, description="Topic IDs to include (None = all topics)")
    period_type: str = Field("month", description="Period granularity: week, month, quarter, year")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    include_key_papers: bool = Field(True, description="Include key papers per period")
    key_papers_limit: int = Field(3, ge=1, le=10, description="Max key papers per period")
    include_forecasts: bool = Field(True, description="Include forecast periods")


class TimelineDataResponse(BaseModel):
    """Response model for timeline visualization data."""
    topics: List[TopicTimelineData] = Field(..., description="Timeline data per topic")
    date_range: Dict[str, str] = Field(..., description="Actual date range: {start, end}")
    period_type: str = Field(..., description="Period granularity used")
    available_zoom_levels: List[str] = Field(
        default=["year", "quarter", "month", "week"],
        description="Available zoom levels for this data"
    )
    total_topics: int = Field(..., description="Total topics returned")


# ---------------------------------------------------------------------------
# Generic operation responses (added in refactor B7 to complete response_model
# coverage — these unblock OpenAPI->TypeScript codegen on the frontend).
# ---------------------------------------------------------------------------

class MessageResponse(BaseModel):
    """Simple message acknowledgement."""
    message: str


class StatusMessageResponse(BaseModel):
    """Status acknowledgement with optional detail message."""
    status: str
    message: Optional[str] = None


class TaskQueuedResponse(BaseModel):
    """Returned when a background task has been queued."""
    task_id: str
    message: str


class CleanupHungJobsResponse(BaseModel):
    cleaned_jobs: List[str]
    count: int


class ActiveTasksResponse(BaseModel):
    active_tasks: List[Dict[str, Any]]


class CompletedTasksResponse(BaseModel):
    completed_tasks: List[Dict[str, Any]]


class EmbeddingUpdateResponse(BaseModel):
    message: str
    updated: bool


class PapersWithoutEmbeddingsResponse(BaseModel):
    papers: List[Dict[str, Any]]
    count: int


class BulkEmbedStartResponse(BaseModel):
    task_id: str
    message: str
    estimated_papers: int
    status: str


class BulkDataCheckResponse(BaseModel):
    paper_count: int
    embedded_count: int
    missing_embeddings: int
    start_date: str
    end_date: str
    has_all_embeddings: bool


# --- Bulk-operations request/response models (moved from the router in B7) ---
class HarvestJudgeRequest(BaseModel):
    date_from: str = Field(..., description="Start date in YYYY-MM-DD format")
    date_to: str = Field(..., description="End date in YYYY-MM-DD format")
    top_n: int = Field(5, description="Number of top papers to select")
    cosine_threshold: float = Field(0.5, description="Cosine similarity threshold")
    update_existing: bool = Field(False, description="Whether to update existing papers")
    batch_size: int = Field(100, description="Batch size for processing")
    max_workers: int = Field(10, description="Maximum number of workers")
    rate_limit_requests: int = Field(5, description="Rate limit requests per second")
    
class BulkJudgeRequest(BaseModel):
    # Profile selection
    profile_ids: Optional[List[str]] = Field(None, description="List of profile IDs to judge")
    all_profiles: bool = Field(False, description="Judge all profiles")

    # Paper filtering
    limit: Optional[int] = Field(None, description="Limit number of papers to judge")
    start_date: Optional[str] = Field(None, description="Start date for papers")
    end_date: Optional[str] = Field(None, description="End date for papers")
    use_profile_arxiv_filters: bool = Field(True, description="Use arXiv category filters from selected profiles (if available)")

    # Multi-server configuration
    use_multi_server: bool = Field(False, description="Use multiple Ollama servers for distributed processing")
    server_ids: Optional[List[int]] = Field(None, description="Specific server IDs to use (if not provided, uses all enabled servers)")
    request_timeout_sec: Optional[int] = Field(None, description="Request timeout in seconds (uses global default if not provided)")
    max_retries: Optional[int] = Field(None, description="Max retries per error type (uses global default if not provided)")

    # Scheduler and job management
    suspend_scheduled_tasks: bool = Field(True, description="Suspend background scheduled tasks during bulk job")
    overwrite_existing: bool = Field(False, description="Overwrite existing scores instead of skipping")

    # Processing configuration
    batch_size: int = Field(100, description="Batch size for processing")
    
class BackfillEmbeddingsRequest(BaseModel):
    limit: Optional[int] = Field(None, description="Limit number of papers to process")
    batch_size: int = Field(100, description="Batch size for processing")
    start_date: Optional[str] = Field(None, description="Start date for papers")
    end_date: Optional[str] = Field(None, description="End date for papers")
    
class BackfillKeywordsRequest(BaseModel):
    limit: Optional[int] = Field(None, description="Limit number of papers to process")
    batch_size: int = Field(100, description="Batch size for processing")
    start_date: Optional[str] = Field(None, description="Start date for papers")
    end_date: Optional[str] = Field(None, description="End date for papers")

# Response models
class JobStartResponse(BaseModel):
    job_id: UUID
    job_type: str
    status: str
    message: str
    
class JobStatusResponse(BaseModel):
    job_id: UUID
    job_type: str
    status: str
    progress_current: int
    progress_total: Optional[int]
    progress_percent: float
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    last_checkpoint_at: Optional[datetime]

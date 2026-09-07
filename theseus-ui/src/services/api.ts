import axios from 'axios';

// --- Types generated from the backend OpenAPI schema (F2) ---
// These replaced hand-written duplicates; regenerate via `npm run generate:api`.
import type {
    ModelConfig,
    MindMapConfig,
    ServerTestResult,
    GlobalDefaults,
    ProfileCreateRequest,
    ProfileUpdateRequest,
    ProfileAwareIngestResponse,
    BulkJudgeRunResponse,
    ProfileInterestResponse,
    LogEntry,
    TaskHistoryEntry,
    PodcastScriptItem,
    PodcastDetailResponse,
    PodcastListItemResponse,
    PaperApiResponse,
    PaginatedPapersResponse,
    SimilarPapersResponse,
    HybridSearchResponse,
    TopicApiResponse,
    TopicMetricResponse,
    TopicDetailResponse,
    TrendsListResponse,
    TrendsSearchResponse,
    TopicPapersResponse,
    TimelineKeyPaper,
    TimelinePeriodData,
    TopicTimelineData,
    TimelineDataResponse,
    PerformanceConfig,
} from './generated/types';
export type {
    ModelConfig,
    MindMapConfig,
    ServerTestResult,
    GlobalDefaults,
    ProfileCreateRequest,
    ProfileUpdateRequest,
    ProfileAwareIngestResponse,
    BulkJudgeRunResponse,
    ProfileInterestResponse,
    LogEntry,
    TaskHistoryEntry,
    PodcastScriptItem,
    PodcastDetailResponse,
    PodcastListItemResponse,
    PaperApiResponse,
    PaginatedPapersResponse,
    SimilarPapersResponse,
    HybridSearchResponse,
    TopicApiResponse,
    TopicMetricResponse,
    TopicDetailResponse,
    TrendsListResponse,
    TrendsSearchResponse,
    TopicPapersResponse,
    TimelineKeyPaper,
    TimelinePeriodData,
    TopicTimelineData,
    TimelineDataResponse,
    PerformanceConfig,
};

// TODO(codegen): the following types conflict with the generated schema
// (required-vs-optional request fields or Dict[str,Any] backend models).
// Re-alias them as backend models tighten.

export interface ProfileAwareIngestRequest {
  start_date?: string;
  end_date?: string;
  profile_ids?: number[];
  profile_tags?: string[];
  score_all_profiles?: boolean;
  overwrite_existing?: boolean;
  cosine_threshold?: number;
  arxiv_categories?: string[];
  use_profile_arxiv_filters?: boolean;
  batch_size?: number;
  send_error_notifications?: boolean;
  // Multi-server configuration (only used when LLM-as-Judge is configured for Ollama)
  use_multi_server?: boolean;
  server_ids?: number[];
}

export interface ResearchTaskRequest {
  research_question: string;
  mode?: 'single' | 'multi';
  config?: {
    search_config?: {
      local_limit?: number;
      external_limit?: number;
    };
    evidence_config?: {
      min_evidence_threshold?: number;
      quality_threshold?: number;
    };
    compression_config?: {
      compression_ratio?: number;
      max_tokens?: number;
    };
    answer_config?: {
      citation_style?: 'academic' | 'numbered' | 'apa';
      include_methodology?: boolean;
      include_limitations?: boolean;
    };
    synthesis_config?: {
      strategy?: 'weighted_consensus' | 'evidence_based' | 'confidence_weighted';
      confidence_threshold?: number;
    };
  };
  save_to_library?: boolean;
}

export interface ResearchTaskResponse {
  task_id: string;
  status: string;
  created_at: string;
  research_question: string;
  mode: 'single' | 'multi';
}

export interface ResearchTaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  mode?: 'single' | 'multi';
  progress?: any;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface ResearchTaskResult {
  task_id: string;
  status: string;
  research_question: string;
  mode: 'single' | 'multi';
  final_answer?: string;
  generation_summary?: string;
  statistics?: {
    research_loops: number;
    total_sources_found: number;
    selected_sources: number;
    evidence_pieces: number;
    evidence_sufficient: boolean;
    compression_used: boolean;
    agents_used?: number;
    synthesis_confidence?: number;
  };
  sub_queries: string[];
  sources_gathered: any[];
  judged_sources: any[];
  evidence: string[];
  compressed_notes: string;
  workflow_messages: any[];
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface ResearchHistoryItem {
  task_id: string;
  research_question: string;
  status: string;
  mode?: 'single' | 'multi';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  statistics?: {
    research_loops: number;
    total_sources_found: number;
    selected_sources: number;
    evidence_pieces: number;
    evidence_sufficient: boolean;
    compression_used: boolean;
    agents_used?: number;
    synthesis_confidence?: number;
  };
}

export interface ResearchModeResponse {
  current_mode: 'single' | 'multi';
  validation: {
    valid: boolean;
    issues: string[];
    has_single_config: boolean;
    has_multi_config: boolean;
  };
  success: boolean;
  message: string;
}

export interface ResearchConfigResponse {
  current_mode: 'single' | 'multi';
  single_agent_config: any;
  multi_agent_config: any;
  validation: {
    valid: boolean;
    issues: string[];
    has_single_config: boolean;
    has_multi_config: boolean;
  };
  available_modes: string[];
}

export interface MindMapExpandRequest {
  paper_id?: string;
  topic_id?: number;
  k?: number;
  similarity_threshold?: number;
  layout_algorithm?: 'force' | 'circular' | 'hierarchical';
  model_config_override?: any;
  expansion_order?: number;
  max_nodes_per_order?: number;
}

export interface MindMapExpandResponse {
  task_id: string;
  message: string;
}

export interface MindMapSeedSearchResponse {
  papers: PaperApiResponse[];
  total_results: number;
}

export interface MindMapReport {
  id: number;
  title: string;
  description?: string;
  seed_paper_id: number;
  seed_paper_title: string;
  parameters: Record<string, any>;
  mindmap_data: MindMapData;
  statistics: Record<string, any>;
  created_at: string;
}

export interface MindMapReportSaveRequest {
  title: string;
  description?: string;
  mindmap_data: MindMapData;
  parameters: Record<string, any>;
}

export interface MindMapReportSaveResponse {
  id: number;
  title: string;
  message: string;
}

export interface MindMapReportListResponse {
  reports: MindMapReport[];
  total_count: number;
}

export interface ScheduledTaskConfig {
  emailRecipients?: string[];
  researchInterests?: string[];
  use_profile_recipients?: boolean;
  lookback_months?: number;
  duration_months?: number;
  min_papers?: number;
  months_to_keep?: number;
}

export interface ScheduledTaskCreate {
  name: string;
  task_type: 'newsletter' | 'trends_recomputation' | 'database_cleanup' | 'profile_ingestion' | 'bulk_embedding';
  profile_id?: number;
  is_enabled: boolean;
  frequency: 'hourly' | 'daily' | 'weekly' | 'monthly';
  day_of_week?: number; // 0=Monday, 6=Sunday
  day_of_month?: number; // 1-31
  hour: number; // 0-23
  minute?: number; // 0-59
  timezone?: string;
  config: ScheduledTaskConfig;
}

export interface ScheduledTaskUpdate {
  name?: string;
  is_enabled?: boolean;
  frequency?: 'hourly' | 'daily' | 'weekly' | 'monthly';
  day_of_week?: number;
  day_of_month?: number;
  hour?: number;
  minute?: number;
  timezone?: string;
  config?: ScheduledTaskConfig;
}

export interface NewsletterRunParams {
  start_date: string;
  end_date: string;
  email_recipients?: string[];
  research_interests: string;
  topic_id?: number;
  generate_podcast_run?: boolean;
  // Profile filtering
  profile_id?: number;
  profile_ids?: number[];
  profile_tag?: string;
  profile_tags?: string[];
  use_profile_recipients?: boolean;
  // Multi-server judge configuration
  use_multi_server_judge?: boolean;
  judge_server_ids?: number[];
  judge_request_timeout_sec?: number;
  judge_max_retries?: number;
  // Newsletter section configuration
  num_sections?: number;
}

export interface BulkJudgeRunRequest {
  profile_ids?: number[];
  profile_tags?: string[];
  start_date?: string;
  end_date?: string;
  overwrite_existing?: boolean;
  cosine_threshold?: number;
  batch_size?: number;
}

export interface InferenceServerCreate {
  name: string;
  url: string;
  provider: 'ollama' | 'lmstudio';
  config_json?: {
    context_length?: number;
    gpu_offload?: string;
  };
  model_name?: string;  // Override model name for this server
  model_config?: {
    temperature?: number;
    max_new_tokens?: number;
    num_ctx?: number;
    context_length?: number;
    gpu_offload?: string;
  };
  notes?: string;
}

export interface InferenceServerUpdate {
  name: string;
  url: string;
  provider: 'ollama' | 'lmstudio';
  enabled: boolean;
  config_json?: {
    context_length?: number;
    gpu_offload?: string;
  };
  model_name?: string;  // Override model name for this server
  model_config?: {
    temperature?: number;
    max_new_tokens?: number;
    num_ctx?: number;
    context_length?: number;
    gpu_offload?: string;
  };
  notes?: string;
}

import type { AxiosResponse } from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

// Configuration interfaces


export interface OrchestrationConfig {
  embedding_model: ModelConfig;
  judge_model: ModelConfig;
  content_extraction_model: ModelConfig;
  newsletter_sections_model: ModelConfig;
  newsletter_intro_model: ModelConfig;
  podcast_model?: ModelConfig;
  tts_model?: any;
  research_agent_model_config?: any;
  mind_map_config?: MindMapConfig;
}

// Trends API Parameter Interfaces
export interface GetTrendingTopicsParams {
  limit?: number;
  period_type?: string;
  duration_months?: number;
  min_doc_count?: number;
  sort_by?: string;
}

export interface SearchTopicsParams {
  query: string;
  limit?: number;
}

export interface GetTopicDetailParams {
  period_type?: string;
  timeline_limit?: number;
  papers_limit?: number;
}

export interface GetTopicPapersParams {
  limit?: number;
  min_relevance?: number;
  sort_by?: 'relevance' | 'score' | 'date';
}

export interface GetResearchInterestPapersParams {
    limit?: number;
    min_similarity?: number;
    sort_by?: 'similarity' | 'score' | 'date';
}

// Newsletter Run Parameters


const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Settings API
export const settingsApi = {
  getOrchestrationConfig: () => api.get('/settings/orchestration'),
  updateOrchestrationConfig: (config: any) => api.put('/settings/orchestration', config),
  getArxivCategories: () => api.get('/settings/arxiv-categories'),
  updateArxivCategories: (config: any) => api.put('/settings/arxiv-categories', config),
  getResearchInterests: () => api.get('/settings/research-interests'),
  updateResearchInterests: (data: any) => api.put('/settings/research-interests', data),
  getEmailRecipients: () => api.get('/settings/email-recipients'),
  updateEmailRecipients: (data: any) => api.put('/settings/email-recipients', data),
  getVisualizerSettings: () => api.get('/settings/visualizer-settings'),
  sendTestEmail: () => api.post('/settings/send-test-email'),
  getCredentials: () => api.get<Record<string, { configured: boolean; value: string }>>('/settings/credentials'),
  updateCredentials: (data: any) => api.put('/settings/credentials', data),
  getModelProviders: () => api.get('/model-providers'),
  getModels: () => api.get('/models'),
  runNewsletterPipeline: (params: NewsletterRunParams) => api.post('/actions/run-newsletter-pipeline', params),
  abortTask: (taskId: string) => api.post(`/tasks/${taskId}/abort`),
  exportDatabase: (onProgress?: (percent: number) => void) =>
    api.get('/settings/database/export', {
      responseType: 'blob',
      onDownloadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    }),
  startExportDatabase: (options?: {
    incremental?: boolean;
    since_timestamp?: string;
    tables?: string[];
    batch_size?: number;
    streaming?: boolean;
    parallel?: boolean;
    max_workers?: number;
  }) => api.post('/settings/database/export-task', options || {}),
  downloadExportDatabase: (
    taskId: string,
    onProgress?: (percent: number) => void
  ) =>
    api.get(`/settings/database/export-task/${taskId}/download`, {
      responseType: 'blob',
      onDownloadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    }),
  importDatabase: (file: File, importMode: 'merge' | 'overwrite' = 'merge') => {
    const formData = new FormData();
    formData.append('backup_file', file);
    formData.append('import_mode', importMode);
    return api.post('/settings/database/import', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

// Inference Servers API (Ollama & LMStudio)
export interface InferenceServer {
  id: number;
  name: string;
  url: string;
  provider: 'ollama' | 'lmstudio';
  enabled: boolean;
  config_json?: {
    context_length?: number;
    gpu_offload?: string;
  };
  model_name?: string;  // Override model name for this server
  model_config?: {
    temperature?: number;
    max_new_tokens?: number;
    num_ctx?: number;
    context_length?: number;
    gpu_offload?: string;
  };
  notes?: string;
  last_tested_at?: string;
  last_test_latency_ms?: number;
  last_test_ok?: boolean;
  created_at?: string;
  updated_at?: string;
}


// Backward compatibility aliases
export type OllamaServer = InferenceServer;
export type OllamaServerCreate = InferenceServerCreate;
export type OllamaServerUpdate = InferenceServerUpdate;


export interface JobMetrics {
  job_id: string;
  job_type: string;
  status: string;
  progress: {
    current: number;
    total: number;
    percent: number;
  };
  timestamps: {
    started_at?: string;
    completed_at?: string;
    last_checkpoint_at?: string;
  };
  error_message?: string;
  queue_metrics?: {
    pending_tasks: number;
    in_progress_tasks: number;
    completed_tasks: number;
    failed_tasks: number;
    total_tasks: number;
  };
  worker_metrics?: Array<{
    worker_id: string;
    server_url: string;
    status: string;
    tasks_processed: number;
    last_heartbeat?: string;
  }>;
}

export interface ActiveJob {
  job_id: string;
  job_type: string;
  status: string;
  progress: {
    current: number;
    total: number;
    percent: number;
  };
  started_at?: string;
  last_checkpoint_at?: string;
  multi_server?: boolean;
  servers?: Array<{
    id: number;
    name: string;
    url: string;
  }>;
  profile_count?: number | string;
}

// Inference Servers API (supports both Ollama and LMStudio)
export const inferenceServersApi = {
  getAllServers: (provider?: 'ollama' | 'lmstudio') =>
    api.get('/settings/inference-servers/', provider ? { params: { provider } } : undefined),
  getServer: (id: number) => api.get(`/settings/inference-servers/${id}`),
  createServer: (server: InferenceServerCreate) => api.post('/settings/inference-servers/', server),
  updateServer: (id: number, server: InferenceServerUpdate) => api.put(`/settings/inference-servers/${id}`, server),
  deleteServer: (id: number) => api.delete(`/settings/inference-servers/${id}`),
  testServer: (id: number) => api.post(`/settings/inference-servers/${id}/test`),
  toggleServer: (id: number) => api.post(`/settings/inference-servers/${id}/toggle`),
  getHealthOverview: () => api.get('/settings/inference-servers/health/overview'),
  getGlobalDefaults: () => api.get('/settings/inference-servers/defaults'),
  updateGlobalDefaults: (defaults: GlobalDefaults) => api.put('/settings/inference-servers/defaults', defaults),
  testUrl: (url: string, provider: 'ollama' | 'lmstudio' = 'ollama') =>
    api.post('/settings/inference-servers/test-url', { url, provider }),
};

// Backward compatibility alias
export const ollamaServersApi = inferenceServersApi;

export interface ServerMetrics {
  id: number;
  name: string;
  url: string;
  enabled: boolean;
  status: string;
  latency_ms?: number;
  last_tested?: string;
}

export interface ServerMetricsSummary {
  total: number;
  enabled: number;
  healthy: number;
  unhealthy: number;
}

export interface QueueStats {
  pending_tasks: number;
  in_progress_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  total_tasks: number;
}

export interface ValidationResult {
  valid: boolean;
  warnings: string[];
  errors: string[];
  recommendations: string[];
  estimated_duration?: {
    minutes: number;
    hours: number;
    formatted: string;
  };
  estimated_tasks?: number;
}

export interface ConflictCheck {
  has_conflicts: boolean;
  bulk_judge_running: boolean;
  conflicts: Array<{
    job_id: string;
    job_type: string;
    started_at?: string;
    description: string;
  }>;
  recommendations: string[];
}

// Bulk Operations API
export const bulkOperationsApi = {
  // Server CRUD operations
  getAllServers: (): Promise<AxiosResponse<OllamaServer[]>> => {
    return api.get('/settings/ollama-servers/');
  },

  getEnabledServers: (): Promise<AxiosResponse<OllamaServer[]>> => {
    return api.get('/settings/ollama-servers/?enabled_only=true');
  },

  createServer: (server: OllamaServerCreate): Promise<AxiosResponse<OllamaServer>> => {
    return api.post('/settings/ollama-servers/', server);
  },

  updateServer: (serverId: number, server: OllamaServerUpdate): Promise<AxiosResponse<OllamaServer>> => {
    return api.put(`/settings/ollama-servers/${serverId}`, server);
  },

  deleteServer: (serverId: number): Promise<AxiosResponse<{ message: string }>> => {
    return api.delete(`/settings/ollama-servers/${serverId}`);
  },

  // Server testing
  testServer: (serverId: number): Promise<AxiosResponse<ServerTestResult>> => {
    return api.post(`/settings/ollama-servers/${serverId}/test`);
  },

  testServerUrl: (url: string, timeoutSeconds: number = 10): Promise<AxiosResponse<ServerTestResult>> => {
    return api.post('/settings/ollama-servers/test-url', { url, timeout_seconds: timeoutSeconds });
  },

  // Server management
  toggleServer: (serverId: number): Promise<AxiosResponse<{ message: string; enabled: boolean }>> => {
    return api.post(`/settings/ollama-servers/${serverId}/toggle`);
  },

  getHealthOverview: (): Promise<AxiosResponse<{
    servers: Array<{
      id: number;
      name: string;
      url: string;
      enabled: boolean;
      status: string;
      last_tested?: string;
      latency_ms?: number;
    }>;
    summary: any;
  }>> => {
    return api.get('/settings/ollama-servers/health/overview');
  },

  // Global defaults
  getGlobalDefaults: (): Promise<AxiosResponse<GlobalDefaults>> => {
    return api.get('/settings/ollama-servers/defaults');
  },

  updateGlobalDefaults: (defaults: GlobalDefaults): Promise<AxiosResponse<GlobalDefaults>> => {
    return api.put('/settings/ollama-servers/defaults', defaults);
  },

  // Job control endpoints
  pauseJob: (jobId: string): Promise<AxiosResponse<{ success: boolean; job_id: string; status: string; message: string }>> => {
    return api.post(`/bulk-operations/job/${jobId}/pause`);
  },

  resumeJob: (jobId: string): Promise<AxiosResponse<{ success: boolean; job_id: string; status: string; message: string }>> => {
    return api.post(`/bulk-operations/job/${jobId}/resume`);
  },

  cancelJob: (jobId: string): Promise<AxiosResponse<{ success: boolean; job_id: string; status: string; message: string }>> => {
    return api.post(`/bulk-operations/job/${jobId}/cancel`);
  },

  getJobMetrics: (jobId: string): Promise<AxiosResponse<JobMetrics>> => {
    return api.get(`/bulk-operations/job/${jobId}/metrics`);
  },

  getActiveJobs: (): Promise<AxiosResponse<{ active_jobs: ActiveJob[]; count: number }>> => {
    return api.get('/bulk-operations/active-jobs');
  },

  getServerMetrics: (): Promise<AxiosResponse<{ servers: ServerMetrics[]; summary: ServerMetricsSummary }>> => {
    return api.get('/bulk-operations/server-metrics');
  },

  getQueueStatus: (): Promise<AxiosResponse<{ queue_stats: QueueStats; active_jobs: ActiveJob[]; timestamp: string }>> => {
    return api.get('/bulk-operations/queue-status');
  },

  checkJobConflicts: (): Promise<AxiosResponse<ConflictCheck>> => {
    return api.post('/bulk-operations/conflict-check');
  },

  validateBulkJudgeJob: (request: unknown): Promise<AxiosResponse<ValidationResult>> => {
    return api.post('/bulk-operations/bulk-judge/validate', request);
  },

  // Job History and Queue Management
  getJobHistory: (params?: { limit?: number; job_type?: string }): Promise<AxiosResponse<{
    jobs: Array<{
      job_id: string;
      job_type: string;
      status: string;
      configuration?: any;
      state?: any;
      started_at?: string;
      completed_at?: string;
      created_at?: string;
      duration_seconds?: number;
      error_message?: string;
    }>;
    summary: {
      total_jobs: number;
      completed_jobs: number;
      failed_jobs: number;
      canceled_jobs: number;
      avg_duration_seconds?: number;
    };
    limit: number;
    job_type_filter?: string;
  }>> => {
    return api.get('/bulk-operations/job-history', { params });
  },

  clearQueue: (params?: { job_id?: string; status_filter?: string }): Promise<AxiosResponse<{
    success: boolean;
    cleared_tasks: number;
    job_id?: string;
    status_filter?: string;
    message: string;
  }>> => {
    return api.delete('/bulk-operations/clear-queue', { params });
  },

  retryWorker: (workerId: string, serverUrl: string, jobId: string): Promise<AxiosResponse<{
    message: string;
    worker_id: string;
    server_url: string;
    requeued_tasks: number;
    status: string;
  }>> => {
    return api.post(`/bulk-operations/worker/${workerId}/retry`, {}, {
      params: { server_url: serverUrl, job_id: jobId }
    });
  },
};

// Newsletter API
export const newsletterApi = {
  runNewsletter: (config: any, introMusicFile?: File) => {
    const formData = new FormData();
    formData.append('config', JSON.stringify(config));
    if (introMusicFile) {
      formData.append('intro_music_file', introMusicFile);
    }
    return api.post('/newsletter/run', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

// Podcast API
export const podcastApi = {
  generatePodcast: (params: any, introMusicFile?: File, pdfFiles?: File[]) => {
    const formData = new FormData();
    formData.append('params_json', JSON.stringify(params));
    if (introMusicFile) {
      formData.append('intro_music_file', introMusicFile);
    }
    if (pdfFiles && pdfFiles.length > 0) {
      pdfFiles.forEach(file => formData.append('pdf_files', file));
    }
    return api.post('/podcast/generate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

// Task API
export const taskApi = {
  getTaskStatus: (taskId: string) => api.get(`/tasks/${taskId}/status`),
  getTaskResult: (taskId: string) => api.get(`/tasks/${taskId}/result`),
  getActiveTasks: (taskTypes?: string[]) => {
    const params = taskTypes ? { task_types: taskTypes.join(',') } : {};
    return api.get('/tasks/active', { params });
  },
  getRecentCompletedTasks: (taskTypes?: string[]) => {
    const params = taskTypes ? { task_types: taskTypes.join(',') } : {};
    return api.get('/tasks/recent-completed', { params });
  },
  downloadTaskArtifact: (taskId: string, fileType: 'markdown' | 'audio' | 'video') => 
    api.get(`/tasks/${taskId}/download/${fileType}`, { responseType: 'blob' }),
  runVisualizerPipeline: (audioFile: File, visualizerParams: any) => {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('visualizer_params_json', JSON.stringify(visualizerParams));
    return api.post('/actions/run-visualizer-pipeline', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
  },
};

// Profile API Types
export interface ProfileApiResponse {
  id: number;
  name: string;
  description?: string;
  color?: string;
  tags?: string[];
  email_recipients?: string[];
  arxiv_filters?: string[];
  is_active: boolean;
  is_default: boolean;
  created_at: string;
  updated_at: string;
  total_papers?: number;
  recent_papers?: number;
}


export interface TagSearchResponse {
  query: string;
  suggestions: Array<{
    tag: string;
    usage_count: number;
  }>;
  exact_match: boolean;
}

export interface BulkEmbedRequest {
  start_date: string;
  end_date: string;
  batch_size?: number;
  skip_existing?: boolean;
  arxiv_categories?: string[];
}


// Profile Research Interest API Types


// Profile API
export const profileApi = {
  // Profile Management
  getProfiles: () => api.get<ProfileApiResponse[]>('/profiles'),
  createProfile: (profile: ProfileCreateRequest) => api.post<ProfileApiResponse>('/profiles', profile),
  getProfile: (id: number) => api.get<ProfileApiResponse>(`/profiles/${id}`),
  updateProfile: (id: number, profile: ProfileUpdateRequest) => api.put<ProfileApiResponse>(`/profiles/${id}`, profile),
  deleteProfile: (id: number) => api.delete(`/profiles/${id}`),
  cloneProfile: (id: number, newName: string) => api.post<ProfileApiResponse>(`/profiles/${id}/clone`, { name: newName }),
  
  // Profile Research Interests
  getProfileInterests: (profileId: number) => api.get<ProfileInterestResponse[]>(`/profiles/${profileId}/interests`),
  createProfileInterest: (profileId: number, interestText: string) => api.post<ProfileInterestResponse>(`/profiles/${profileId}/interests`, { interest_text: interestText }),
  deleteProfileInterest: (profileId: number, interestId: number) => api.delete(`/profiles/${profileId}/interests/${interestId}`),
  
  // Profile Tags
  getAllTags: () => api.get<string[]>('/profiles/tags'),
  searchTags: (query: string, limit: number = 10) => api.get<TagSearchResponse>(`/profiles/tags/search`, { params: { q: query, limit } }),
  getProfilesByTag: (tag: string) => api.get<ProfileApiResponse[]>(`/profiles/by-tag/${tag}`),
  
  // Profile-Scoped Operations
  getProfilePapers: (id: number, params?: any) => api.get(`/profiles/${id}/papers`, { params }),
  runProfileJudge: (id: number, params: any) => api.post(`/profiles/${id}/judge-run`, params),
  getProfileTrends: (id: number, params?: any) => api.get(`/profiles/${id}/trends`, { params }),
  generateProfileNewsletter: (id: number, params: any) => api.post(`/profiles/${id}/newsletter`, params),
  generateProfileMindmap: (id: number, params: any) => api.post(`/profiles/${id}/mindmap`, params),
  
  // Bulk Operations
  runBulkJudge: (request: BulkJudgeRunRequest) => api.post<BulkJudgeRunResponse>('/profiles/bulk/judge-run', request),
  generateBulkNewsletters: (params: any) => api.post('/profiles/bulk/newsletter', params),
  getBulkTrends: (params: any) => api.get('/profiles/bulk/trends', { params }),

  // Profile-Aware Ingestion
  runProfileAwareIngest: (request: ProfileAwareIngestRequest) => api.post<ProfileAwareIngestResponse>('/papers/profile-aware-ingest', request),

  // Bulk Embedding
  runBulkEmbed: (request: BulkEmbedRequest) => api.post('/papers/bulk-embed', request),
  checkExistingBulkData: (params: { start_date: string; end_date: string }) => api.get('/papers/check-existing-bulk-data', { params }),


};

// === Profile Star Map API Types ===
export interface StarMapPoint {
  paper_id: number;
  x: number;
  y: number;
  z?: number;
  // Optional metadata for richer UI (kept light for 10k points)
  title?: string;
  date?: string;
  profile_score?: number;
  paper_score?: number;
  cluster_id?: number;
  dominant_interest_id?: number;
  dominant_label?: string;
  dominant_short_label?: string;
}

export interface StarMapCentroid {
  interest_id: number;
  x: number;
  y: number;
  z: number;
  label: string;
  short_label: string;
  count: number;
}

export interface ProfileStarMapResponse {
  profile_id: number;
  total_points: number;
  computed_at?: string;
  points: StarMapPoint[];
  centroids?: StarMapCentroid[];
}

export interface ProfileStarMapStatusResponse {
  profile_id: number;
  computed_at?: string;
  total_points?: number;
}

export interface ProfileStarMapRecomputeResponse {
  task_id: string;
  message: string;
}

export const starMapApi = {
  getProfileStarMap: (profileId: number, limit: number = 10000) =>
    api.get<ProfileStarMapResponse>(`/profiles/${profileId}/star-map`, { params: { limit } }),
  getProfileStarMapStatus: (profileId: number) =>
    api.get<ProfileStarMapStatusResponse>(`/profiles/${profileId}/star-map/status`),
  recomputeProfileStarMap: (profileId: number, params?: { limit?: number }) =>
    api.post<ProfileStarMapRecomputeResponse>(`/profiles/${profileId}/star-map/recompute`, params || {}),
};

// Runs API
export const runsApi = {
  getRuns: (page: number = 1) => api.get('/runs', { params: { page } }),
  deleteRunArtifact: (runId: number) => api.delete(`/runs/${runId}/artifact`),
};

// Scheduled Tasks Types


export interface ScheduledTask {
  id: number;
  name: string;
  task_type: string;
  profile_id?: number;
  is_enabled: boolean;
  frequency: string;
  day_of_week?: number;
  day_of_month?: number;
  hour: number;
  minute: number;
  timezone: string;
  config: ScheduledTaskConfig;
  last_run_at?: string;
  next_run_at?: string;
  last_run_status?: string;
  last_run_task_id?: string;
  run_count: number;
  error_count: number;
  created_at: string;
  updated_at: string;
  profile_name?: string;
}

export interface ScheduledTaskRun {
  id: number;
  scheduled_task_id: number;
  task_id: string;
  started_at: string;
  completed_at?: string;
  status: string;
  error_message?: string;
  result?: any;
}

// Scheduled Tasks API
export const scheduledTasksApi = {
  // Get all scheduled tasks
  getTasks: (params?: { profile_id?: number; task_type?: string; is_enabled?: boolean }) => 
    api.get<ScheduledTask[]>('/scheduled-tasks', { params }),
  
  // Get a specific task
  getTask: (taskId: number) => api.get<ScheduledTask>(`/scheduled-tasks/${taskId}`),
  
  // Create a new scheduled task
  createTask: (task: ScheduledTaskCreate) => api.post<ScheduledTask>('/scheduled-tasks', task),
  
  // Update a scheduled task
  updateTask: (taskId: number, update: ScheduledTaskUpdate) => 
    api.put<ScheduledTask>(`/scheduled-tasks/${taskId}`, update),
  
  // Delete a scheduled task
  deleteTask: (taskId: number) => api.delete(`/scheduled-tasks/${taskId}`),
  
  // Get run history for a task
  getTaskRuns: (taskId: number, limit?: number, offset?: number) => 
    api.get<ScheduledTaskRun[]>(`/scheduled-tasks/${taskId}/runs`, { params: { limit, offset } }),
  
  // Manually run a task
  runTaskNow: (taskId: number) => api.post(`/scheduled-tasks/${taskId}/run`),
  
  // Sync all tasks (e.g., after server restart)
  syncTasks: () => api.post('/scheduled-tasks/sync'),
};

// Research Agent API
export const researchAgentApi = {
  startResearchTask: (request: any) => api.post('/research-agent/run', request),
  getTaskStatus: (taskId: string) => api.get(`/research-agent/status/${taskId}`),
  getTaskResult: (taskId: string) => api.get(`/research-agent/result/${taskId}`),
  getHistory: (limit: number = 50, offset: number = 0, statusFilter?: string, modeFilter?: string) => {
    const params: any = { limit, offset };
    if (statusFilter) params.status_filter = statusFilter;
    if (modeFilter) params.mode_filter = modeFilter;
    return api.get('/research-agent/history', { params });
  },
  cancelTask: (taskId: string) => api.delete(`/research-agent/${taskId}`),
  getWorkflowInfo: () => api.get('/research-agent/workflow/info'),
  getHealth: () => api.get('/research-agent/health'),
  
  // Dual-mode configuration endpoints
  getModes: () => api.get('/research-agent/modes'),
  setMode: (mode: 'single' | 'multi') => api.put('/research-agent/mode', { mode }),
  getModeConfig: (mode: 'single' | 'multi') => api.get(`/research-agent/config/${mode}`),
  setModeConfig: (mode: 'single' | 'multi', config: any) => api.put(`/research-agent/config/${mode}`, config),
};

// Model Catalog API
export const modelCatalogApi = {
  searchModels: (params: any) => api.get('/model-catalog/', { params }),
  createModel: (model: any) => api.post('/model-catalog/', model),
  updateModel: (modelId: number, model: any) => api.put(`/model-catalog/${modelId}`, model),
  deleteModel: (modelId: number) => api.delete(`/model-catalog/${modelId}`),
  toggleFavorite: (modelId: number) => api.post(`/model-catalog/${modelId}/toggle-favorite`),
  getModel: (modelId: number) => api.get(`/model-catalog/${modelId}`),
};

// Mind-Map API
export const mindMapApi = {
  expandMindMap: (request: MindMapExpandRequest) => 
    api.post('/mindmap/expand', request),
  parsePDFs: (request: MindMapPDFParseRequest) => 
    api.post('/mindmap/parse-pdfs', request),
  searchSeeds: (request: MindMapSeedSearchRequest) => 
    api.get('/mindmap/search-seeds', { params: request }),
  getPaper: (paperId: string) => 
    api.get(`/mindmap/paper/${paperId}`),
  
  // Report management
  getReports: async (): Promise<MindMapReportListResponse> => {
    const response: AxiosResponse<MindMapReportListResponse> = await api.get('/mindmap/reports');
    return response.data;
  },
  getReport: async (reportId: number): Promise<MindMapReport> => {
    const response: AxiosResponse<MindMapReport> = await api.get(`/mindmap/reports/${reportId}`);
    return response.data;
  },
  saveReport: async (request: MindMapReportSaveRequest): Promise<MindMapReportSaveResponse> => {
    const response: AxiosResponse<MindMapReportSaveResponse> = await api.post('/mindmap/reports', request);
    return response.data;
  },
  updateReport: async (reportId: number, request: MindMapReportSaveRequest): Promise<{ message: string; id: number }> => {
    const response: AxiosResponse<{ message: string; id: number }> = await axios.put(`/api/mindmap/reports/${reportId}`, request);
    return response.data;
  },
  deleteReport: async (reportId: number): Promise<{ status: string; message: string }> => {
    const response: AxiosResponse<{ status: string; message: string }> = await api.delete(`/mindmap/reports/${reportId}`);
    return response.data;
  },
  updateReportTitle: async (reportId: number, title: string): Promise<{ status: string; message: string; title: string }> => {
    const response: AxiosResponse<{ status: string; message: string; title: string }> = await api.put(`/mindmap/reports/${reportId}/title`, { title });
    return response.data;
  },
  updateReportDescription: async (reportId: number, description: string): Promise<{ status: string; message: string; description: string }> => {
    const response: AxiosResponse<{ status: string; message: string; description: string }> = await api.put(`/mindmap/reports/${reportId}/description`, { description });
    return response.data;
  },
};

// WebSocket connection
export const createWebSocket = (
  taskId: string,
  type:
    | 'newsletter'
    | 'podcast'
    | 'visualizer'
    | 'research-agent'
    | 'mindmap'
    | 'mindmap-pdf-parse'
    | 'trends'
    | 'star-map'
) => {
  const ws = new WebSocket(`ws://localhost:8000/ws/${type}/${taskId}`);
  return ws;
};


export const getLogs = async (limit: number = 100, fromDate?: string, toDate?: string): Promise<LogEntry[]> => {
  const params: Record<string, string | number> = { limit };
  if (fromDate) {
    params.from_date = fromDate;
  }
  if (toDate) {
    params.to_date = toDate;
  }
  const response: AxiosResponse<LogEntry[]> = await api.get<LogEntry[]>("/logs", { params });
  return response.data;
};

export const getTaskHistory = async (limit: number = 100, fromDate?: string, toDate?: string): Promise<TaskHistoryEntry[]> => {
  const params: Record<string, string | number> = { limit };
  if (fromDate) {
    params.from_date = fromDate;
  }
  if (toDate) {
    params.to_date = toDate;
  }
  const response: AxiosResponse<TaskHistoryEntry[]> = await api.get<TaskHistoryEntry[]>("/task-history", { params });
  return response.data;
};

// Task API calls
export interface RunStatusPayload { 
  taskId: string;
  status: string;
  progress?: number;
  message?: string;
}

export const getTaskStatus = async (taskId: string): Promise<RunStatusPayload | null> => {
  if (!taskId || taskId.startsWith("dummy-")) return null; // Avoid calling API with placeholder
  try {
    const response = await api.get<RunStatusPayload>(`/tasks/${taskId}/status`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching status for task ${taskId}:`, error);
    return null;
  }
};

// Interfaces for Podcast History


// Interfaces for Papers Page


// Podcast History API functions
export const podcastHistoryApi = {
    getPodcastHistoryList: async (): Promise<PodcastListItemResponse[]> => {
        const response: AxiosResponse<PodcastListItemResponse[]> = await api.get<PodcastListItemResponse[]>('/podcasts/history');
        return response.data;
    },

    getPodcastDetail: async (podcastId: string): Promise<PodcastDetailResponse> => {
        const response: AxiosResponse<PodcastDetailResponse> = await api.get<PodcastDetailResponse>(`/podcasts/history/${podcastId}`);
        return response.data;
    },

    deletePodcast: async (podcastId: number): Promise<{ status: string; message: string }> => {
        const response: AxiosResponse<{ status: string; message: string }> = await api.delete(`/podcasts/history/${podcastId}`);
        return response.data;
    },

    updatePodcastTitle: async (podcastId: number, title: string): Promise<{ status: string; message: string; title: string }> => {
        const response: AxiosResponse<{ status: string; message: string; title: string }> = await api.put(`/podcasts/history/${podcastId}/title`, { title });
        return response.data;
    },
};

// Papers API functions
export const papersApi = {
    getPapers: async (
        page: number = 1,
        pageSize: number = 10,
        sortField: string = 'score',
        sortDirection: string = 'desc',
        minScore?: number,
        maxScore?: number,
        fromDate?: string,
        toDate?: string,
        search?: string,
        topicId?: number,
        profileIds?: number[],
        minProfileScore?: number,
        maxProfileScore?: number,
        profileRelatedOnly?: boolean,
        profileInterestId?: number
    ): Promise<PaginatedPapersResponse> => {
        const params: Record<string, any> = {
            page,
            page_size: pageSize,
            sort_field: sortField,
            sort_direction: sortDirection
        };

        if (minScore !== undefined) params.score = minScore;
        if (maxScore !== undefined) params.max_score = maxScore;
        if (fromDate) params.from_date = fromDate;
        if (toDate) params.to_date = toDate;
        if (search) params.search = search;
        if (topicId !== undefined) params.topic_id = topicId;
        if (profileInterestId !== undefined) params.profile_interest_id = profileInterestId;
        
        // Profile filtering parameters
        if (profileIds && profileIds.length > 0) {
            params.profile_ids = profileIds.join(',');
        }
        if (minProfileScore !== undefined) params.min_profile_score = minProfileScore;
        if (maxProfileScore !== undefined) params.max_profile_score = maxProfileScore;
        if (profileRelatedOnly) params.profile_related_only = profileRelatedOnly;

        const response: AxiosResponse<PaginatedPapersResponse> = await api.get<PaginatedPapersResponse>('/papers', {
            params
        });
        return response.data;
    },

    updatePaper: async (
        paperId: number,
        update: { score?: number; related?: boolean; profile_ids?: number[] }
    ): Promise<PaperApiResponse> => {
        const response: AxiosResponse<PaperApiResponse> = await api.put<PaperApiResponse>(`/papers/${paperId}`, update);
        return response.data;
    },

    findSimilarPapers: async (
        paperId: number,
        limit: number = 10,
        similarityThreshold: number = 0.7
    ): Promise<SimilarPapersResponse> => {
        const params = {
            limit,
            similarity_threshold: similarityThreshold
        };

        const response: AxiosResponse<SimilarPapersResponse> = await api.get<SimilarPapersResponse>(`/papers/${paperId}/similar`, {
            params
        });
        return response.data;
    },

    hybridSearch: async (
        queryText: string,
        page: number = 1,
        pageSize: number = 10,
        semanticWeight: number = 0.6,
        keywordWeight: number = 0.4,
        similarityThreshold: number = 0.3,
        minScore?: number,
        maxScore?: number,
        fromDate?: string,
        toDate?: string,
        profileIds?: number[],
        minProfileScore?: number,
        maxProfileScore?: number,
        profileRelated?: boolean
    ): Promise<HybridSearchResponse> => {
        const requestBody = {
            query_text: queryText,
            page,
            page_size: pageSize,
            semantic_weight: semanticWeight,
            keyword_weight: keywordWeight,
            similarity_threshold: similarityThreshold,
            min_score: minScore,
            max_score: maxScore,
            from_date: fromDate,
            to_date: toDate,
            profile_ids: profileIds,
            min_profile_score: minProfileScore,
            max_profile_score: maxProfileScore,
            profile_related: profileRelated
        };

        const response: AxiosResponse<HybridSearchResponse> = await api.post<HybridSearchResponse>('/papers/hybrid-search', requestBody);
        return response.data;
    },
};

// Research Agent Interfaces


export interface ResearchWebSocketMessage {
  type: 'status_update' | 'task_completed';
  task_id: string;
  status: string;
  progress?: any;
  timestamp: string;
  error_message?: string;
  results?: {
    final_answer?: string;
    statistics?: any;
    sub_queries?: string[];
    sources_count?: number;
    evidence_count?: number;
  };
}

// Mind-Map API Types
export interface MindMapNode {
  id: string | number;
  title: string;
  abstract: string;
  date: string;
  url: string;
  score: number;
  rationale: string;
  similarity_score: number;
  summary?: string;
  keywords?: string[];
  has_fulltext?: boolean;
  is_seed?: boolean;
  colorIndex?: number;
}

export interface MindMapEdge {
  source_id: number;
  target_id: number;
  similarity_score: number;
  relationship_type?: string;
}

export interface MindMapData {
  nodes: MindMapNode[];
  edges: MindMapEdge[];
  seed_paper_id: string;
  layout_algorithm: string;
  generation_timestamp: string;
}


export interface MindMapPDFParseRequest {
  paper_ids: string[];
}

export interface MindMapPDFParseResponse {
  task_id: string;
  message: string;
  papers_count: number;
}

export interface MindMapSeedSearchRequest {
  query: string;
  limit?: number;
}


export interface MindMapWebSocketMessage {
  type: 'progress_update' | 'task_completed' | 'task_failed';
  task_id: string;
  step: string;
  progress: number;
  message: string;
  timestamp: string;
  mindmap_data?: MindMapData;
  statistics?: {
    nodes_created: number;
    edges_created: number;
    layout_algorithm: string;
  };
  error?: string;
}

// Mind-Map Reports interfaces


// === Trends API Interfaces ===


export interface ResearchInterestPapersResponse {
    research_interest_id: number;
    interest_text: string;
    papers: PaperApiResponse[];
    total_papers: number;
}

// Trends API
export const trendsApi = {
  getTrendingTopics: async (params?: GetTrendingTopicsParams): Promise<AxiosResponse<TrendsListResponse>> => {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.period_type) searchParams.append('period_type', params.period_type);
    if (params?.duration_months) searchParams.append('duration_months', params.duration_months.toString());
    if (params?.min_doc_count) searchParams.append('min_doc_count', params.min_doc_count.toString());
    if (params?.sort_by) searchParams.append('sort_by', params.sort_by);
    
    return api.get(`/trends?${searchParams.toString()}`);
  },

  searchTopics: async (params: SearchTopicsParams): Promise<AxiosResponse<TrendsSearchResponse>> => {
    const searchParams = new URLSearchParams();
    searchParams.append('query', params.query);
    if (params.limit) searchParams.append('limit', params.limit.toString());
    
    return api.get(`/trends/search?${searchParams.toString()}`);
  },

  getTopicDetail: async (topicId: number, params?: GetTopicDetailParams): Promise<AxiosResponse<TopicDetailResponse>> => {
    const searchParams = new URLSearchParams();
    if (params?.period_type) searchParams.append('period_type', params.period_type);
    if (params?.timeline_limit) searchParams.append('timeline_limit', params.timeline_limit.toString());
    if (params?.papers_limit) searchParams.append('papers_limit', params.papers_limit.toString());
    
    return api.get(`/trends/${topicId}?${searchParams.toString()}`);
  },

  getTopicPapers: async (topicId: number, params?: GetTopicPapersParams): Promise<AxiosResponse<TopicPapersResponse>> => {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.min_relevance) searchParams.append('min_relevance', params.min_relevance.toString());
    if (params?.sort_by) searchParams.append('sort_by', params.sort_by);
    return api.get(`/trends/${topicId}/papers?${searchParams.toString()}`);
  },

  summarizeLabels: async (labels: string[]): Promise<AxiosResponse<Record<string, string>>> => {
    return api.post('/trends/summarize-labels', labels);
  },

  getResearchInterestPapers: async (interestId: number, params?: GetResearchInterestPapersParams): Promise<AxiosResponse<ResearchInterestPapersResponse>> => {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.min_similarity) searchParams.append('min_similarity', params.min_similarity.toString());
    if (params?.sort_by) searchParams.append('sort_by', params.sort_by);
    return api.get(`/trends/research-interests/${interestId}/papers`, { params: searchParams });
  },

  getTimelineData: async (params?: GetTimelineDataParams): Promise<AxiosResponse<TimelineDataResponse>> => {
    const searchParams = new URLSearchParams();
    if (params?.topic_ids) searchParams.append('topic_ids', params.topic_ids);
    if (params?.profile_ids) searchParams.append('profile_ids', params.profile_ids);
    if (params?.period_type) searchParams.append('period_type', params.period_type);
    if (params?.start_date) searchParams.append('start_date', params.start_date);
    if (params?.end_date) searchParams.append('end_date', params.end_date);
    if (params?.include_key_papers !== undefined) searchParams.append('include_key_papers', params.include_key_papers.toString());
    if (params?.key_papers_limit) searchParams.append('key_papers_limit', params.key_papers_limit.toString());
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.source) searchParams.append('source', params.source);

    return api.get(`/trends/timeline-data?${searchParams.toString()}`);
  },

  generateShortLabels: async (params?: { profile_ids?: number[] }): Promise<AxiosResponse<{
    status: string;
    message: string;
    processed: number;
    total_interests?: number;
    errors?: string[];
  }>> => {
    return api.post('/trends/generate-short-labels', params || {});
  },
};

// Research Interest Clustering API
export const researchInterestsApi = {
  getResearchInterests: async (params?: GetTrendingTopicsParams): Promise<AxiosResponse<TrendsListResponse>> => {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.period_type) searchParams.append('period_type', params.period_type);
    if (params?.duration_months) searchParams.append('duration_months', params.duration_months.toString());
    if (params?.min_doc_count) searchParams.append('min_doc_count', params.min_doc_count.toString());
    if (params?.sort_by) searchParams.append('sort_by', params.sort_by);
    
    return api.get(`/trends/research-interests?${searchParams.toString()}`);
  },

  searchResearchInterests: async (params: SearchTopicsParams): Promise<AxiosResponse<TrendsSearchResponse>> => {
    const searchParams = new URLSearchParams();
    searchParams.append('query', params.query);
    if (params.limit) searchParams.append('limit', params.limit.toString());
    
    return api.get(`/trends/research-interests/search?${searchParams.toString()}`);
  },

  getResearchInterestDetail: async (interestId: number, params?: GetTopicDetailParams): Promise<AxiosResponse<ResearchInterestDetailResponse>> => {
    const searchParams = new URLSearchParams();
    if (params?.period_type) searchParams.append('period_type', params.period_type);
    if (params?.timeline_limit) searchParams.append('timeline_limit', params.timeline_limit.toString());
    if (params?.papers_limit) searchParams.append('papers_limit', params.papers_limit.toString());

    return api.get(`/trends/research-interests/${interestId}?${searchParams.toString()}`);
  },

  recomputeProfileInterests: async (params: {
    profile_ids: number[];
    lookback_months?: number;
    similarity_threshold?: number;
    clear_existing?: boolean;
  }): Promise<AxiosResponse<{ task_id: string; message: string }>> => {
    return api.post('/trends/research-interests/recompute', params);
  },
};

// === Research Interest Clustering API Interfaces ===

export interface ResearchInterestApiResponse {
  id: number;
  interest_text: string;
  embedding_model?: string;
  created_at: string;
  updated_at?: string;
  latest_doc_count?: number;
  latest_growth_rate?: number;
  total_papers?: number;
  latest_avg_relevance?: number;
  latest_avg_score?: number;
  forecast_1m?: number;
  forecast_3m?: number;
  forecast_6m?: number;
}

export interface ResearchInterestMetricResponse {
  id: number;
  research_interest_id: number;
  period_start: string;
  period_end: string;
  period_type: string;
  doc_count: number;
  avg_relevance_score?: number;
  avg_paper_score?: number;
  growth_rate?: number;
  forecast_1m?: number;
  forecast_3m?: number;
  forecast_6m?: number;
  created_at: string;
}

export interface ResearchInterestDetailResponse {
  interest: ResearchInterestApiResponse;
  timeline: ResearchInterestMetricResponse[];
  representative_papers: PaperApiResponse[];
  total_papers: number;
}

// Union type for detail responses to handle both topics and research interests
export type EntityDetailResponse = TopicDetailResponse | ResearchInterestDetailResponse;

// === Research Timeline Visualization Types ===


export interface GetTimelineDataParams {
  topic_ids?: string;
  profile_ids?: string;
  period_type?: 'week' | 'month' | 'quarter' | 'year';
  start_date?: string;
  end_date?: string;
  include_key_papers?: boolean;
  key_papers_limit?: number;
  limit?: number;
  source?: 'research_interests' | 'profile_interests' | 'topics';
}

// Type guards to differentiate between topic and research interest responses
export function isTopicDetailResponse(response: EntityDetailResponse): response is TopicDetailResponse {
  return 'topic' in response;
}

export function isResearchInterestDetailResponse(response: EntityDetailResponse): response is ResearchInterestDetailResponse {
  return 'interest' in response;
}

// Performance Configuration interfaces


export interface SystemInfo {
  cpu_count_physical: number;
  cpu_count_logical: number;
  memory_total_gb: number;
  memory_available_gb: number;
  gpu_available: boolean;
  gpu_name?: string;
  recommended_config: PerformanceConfig;
}

// Performance Configuration API
export const performanceApi = {
  getSystemInfo: async (): Promise<SystemInfo> => {
    const response = await api.get('/trends/system-info');
    return response.data;
  },

  getPerformanceConfig: async (): Promise<PerformanceConfig> => {
    const response = await api.get('/trends/performance-config');
    return response.data;
  },

  updatePerformanceConfig: async (config: PerformanceConfig): Promise<{ status: string; message: string }> => {
    const response = await api.post('/trends/performance-config', config);
    return response.data;
  },
};

export const runtimeApi = {
  diagnostics: (taskId: string) => api.get<{
    events: { stage: string; status: string; duration_ms: number | null; error_type: string | null; created_at: string }[];
    deliveries: { status: string; updated_at: string }[];
    task?: { status: string; current_step: string | null; message: string | null; error: string | null };
  }>(`/tasks/${encodeURIComponent(taskId)}/diagnostics`),
  retry: (taskId: string) => api.post(`/tasks/${encodeURIComponent(taskId)}/retry`),
  resolveDelivery: (taskId: string, resolution: 'sent' | 'retry') => api.post(`/tasks/${encodeURIComponent(taskId)}/delivery-resolution`, null, { params: { resolution } }),
};

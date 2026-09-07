import os
import json
import gc
import shutil
import pickle
import random
import datetime
import time
import concurrent.futures as cf
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import json_repair
from typing import Optional, Callable, List
import yake
import asyncio
import uuid

# Local application imports
from theseus_insight.communication import GmailCommunication, construct_email_body, upload_video
from theseus_insight.data_processing import ArxivDataProcessor, Paper, Newsletter, Podcast
from theseus_insight.pipeline.checkpoints import CheckpointAdapter
from theseus_insight.pipeline.model_loading import load_inference_model
from theseus_insight.pipeline import embedding_pipeline, profile_scoring, profiles_pipeline, ranking
from theseus_insight.pipeline.stages import download as download_stage
from theseus_insight.pipeline.stages import email as email_stage
from theseus_insight.pipeline.stages import embed as embed_stage
from theseus_insight.pipeline.stages import rank as rank_stage
from theseus_insight.pipeline.stages import newsletter_content as newsletter_content_stage
from theseus_insight.pipeline.stages import newsletter_sections as newsletter_sections_stage
from theseus_insight.pipeline.stages import podcast as podcast_stage
from theseus_insight.pdf.markdown_extraction import (
    download_pdf_to_temp_file, pdf_to_markdown,
)
from theseus_insight.data_access import (
    PaperRepository, LogsRepository, NewsletterRepository, 
    PodcastRepository, SettingsRepository
)
from theseus_insight.db import get_connection_pool
from theseus_insight.inference import SentenceTransformerInference
from theseus_insight.prompt import (
    NEWSLETTER_SYSTEM_PROMPT,
    RESEARCH_INTERESTS_SYSTEM_PROMPT,
    SYSTEM_CONTENT_EXTRACTION_SUMMARY,
    INSTRUCTION_TEMPLATES,
    general_summary_prompt,
    research_prompt,
    newsletter_context_prompt,
    newsletter_intro_prompt,
    SummaryPromptData,
    ResearchInterestsPromptData,
    NewsletterPromptData
)
from theseus_insight.utils import cosine_similarity, get_n_days_ago, TODAY, purge_ollama_cache
from theseus_insight.constants import INTRO_TEXT

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
GMAIL_SENDER_ADDRESS = os.getenv("GMAIL_SENDER_ADDRESS", None)
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", None)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
PDF_CONVERSION_TIMEOUT_SEC = int(os.getenv("PDF_CONVERSION_TIMEOUT_SEC", "120"))
PDF_DOWNLOAD_MAX_WORKERS = int(os.getenv("PDF_DOWNLOAD_MAX_WORKERS", "4"))


def _parse_optional_timeout_env(name: str, default: Optional[float]) -> Optional[float]:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"0", "none", "off", "false", "no"}:
        return None

    return float(raw_value)


NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC = _parse_optional_timeout_env(
    "NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC",
    1200.0,
)


class TheseusInsight:
    def __init__(self,
                 research_interests_path="config/research_interests.txt",
                 n_days=7,
                 top_n=5,
                 start_date_override=None,
                 end_date_override=None,
                 orchestration_config="config/orchestration.json",
                 receiver_address_override=None,
                 research_interests_override=None,
                 profile_ids_override=None,
                 max_new_tokens=1024,
                 temperature=0.1,
                 cosine_similarity_threshold=0.5,
                 db_saving=True,
                 data_path=os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/theseus"),
                 generate_podcast=False,
                 intro_music_path=None,
                 output_format: str = "mp3",
                 output_dir: str = "output_audio",
                 prefix: str = "podcast_segment",
                 final_filename: str = "podcast_final",
                 resolution: tuple = (1920, 1080),
                 fps: int = 30,
                 matrix_count=200,
                 matrix_head_color="#e0ffe7",   # short hex for bright green
                 matrix_tail_color="0x00b000",  # hex for (0,176,0)
                 matrix_char_size=24,
                 head_step_time=0.25,
                 random_x_jitter=2.0,
                 fade_time=1.5,
                 head_glow_passes=3,
                 head_glow_alpha_decay=50,
                 head_spawn_delay_range=(1.0,3.0),
                 head_saw_period=1.5,
                 wave_color="#d703fc",
                 trail_colors=["#fc03b6", "#ba03fc", "#ce6bf2"], 
                 glow_passes=3,
                 glow_alpha_decay=40,
                 line_width=6,
                 font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                 verbose=True,
                 generate_email=True,
                 publish_podcast=False,
                 visualizer=True,
                 save_dialogue=True,
                 checkpoint_dir="checkpoints",
                 task_id=None,
                 send_error_notifications=True,
                 use_database_checkpoints=True,
                 use_multi_server_judge=False,
                 judge_server_ids=None,
                 newsletter_job_id=None,
                 judge_request_timeout_sec=None,
                 judge_max_retries=None,
                 progress_callback: Optional[Callable[[str, float, str], None]] = None):
        
        # Store task_id for logging
        self.task_id = task_id or str(__import__("uuid").uuid4())

        self.verbose = verbose
        self.progress_callback = progress_callback
        self.save_dialogue = save_dialogue
        self.checkpoint_dir = os.path.join("data", "checkpoints", self.task_id) if checkpoint_dir == "checkpoints" else checkpoint_dir
        self.generate_email = generate_email
        self.publish_podcast = publish_podcast
        self.generate_podcast = generate_podcast
        self.use_database_checkpoints = use_database_checkpoints

        # Multi-server judge configuration
        self.use_multi_server_judge = use_multi_server_judge
        self.judge_server_ids = judge_server_ids
        self.newsletter_job_id = newsletter_job_id
        self.judge_request_timeout_sec = judge_request_timeout_sec
        self.judge_max_retries = judge_max_retries

        # Set error notification flag early (before any operations that might fail)
        self.send_error_notifications = send_error_notifications
        self.error_notified = False
        
        # Checkpoint persistence (file + optional DB) — see pipeline/checkpoints.py
        self._checkpoints = CheckpointAdapter(
            self.checkpoint_dir,
            use_database_checkpoints=use_database_checkpoints,
            verbose=verbose,
        )
        
        # Email/Communication
        final_receiver_address = None
        if receiver_address_override is not None:
            final_receiver_address = receiver_address_override
        
        if final_receiver_address:
            if isinstance(final_receiver_address, str) and ',' in final_receiver_address:
                self.receiver_address = [addr.strip() for addr in final_receiver_address.split(',')]
            elif isinstance(final_receiver_address, list):
                self.receiver_address = final_receiver_address
            else: # Should be a single string email
                self.receiver_address = [final_receiver_address] if isinstance(final_receiver_address, str) else []

        else:
            self.receiver_address = None

        self.communication = GmailCommunication(
            sender_address=GMAIL_SENDER_ADDRESS,
            app_password=GMAIL_APP_PASSWORD,
            receiver_address=self.receiver_address
        )

        # Dates
        if start_date_override is None and end_date_override is None:
            self.start_date = get_n_days_ago(n_days)
            self.end_date = TODAY
        else:
            def try_parse_date(date_str):
                if not date_str:
                    return None
                formats = ["%Y-%m-%d", "%m-%d-%Y"]
                for fmt in formats:
                    try:
                        return datetime.datetime.strptime(date_str, fmt).date()
                    except ValueError:
                        continue
                # If it's already a date object, return it
                if isinstance(date_str, datetime.date):
                    return date_str
                raise ValueError("Dates must be in YYYY-MM-DD or MM-DD-YYYY format or a date object")

            self.start_date = try_parse_date(start_date_override) if start_date_override else get_n_days_ago(n_days)
            self.end_date = try_parse_date(end_date_override) if end_date_override else TODAY

        self.top_n = top_n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.db_saving = db_saving
        
        # Store data_path for reference (repositories handle their own connections)
        self.data_path = data_path
        
        # Podcast settings
        self.intro_music_path = intro_music_path
        self.output_format = output_format
        self.output_dir = output_dir
        self.prefix = prefix
        self.final_filename = final_filename
        self.resolution = resolution
        self.fps = fps
        self.matrix_count = matrix_count
        self.matrix_head_color = matrix_head_color
        self.matrix_tail_color = matrix_tail_color
        self.matrix_char_size = matrix_char_size
        self.head_step_time = head_step_time
        self.random_x_jitter = random_x_jitter
        self.fade_time = fade_time
        self.head_glow_passes = head_glow_passes
        self.head_glow_alpha_decay = head_glow_alpha_decay
        self.head_spawn_delay_range = head_spawn_delay_range
        self.head_saw_period = head_saw_period
        self.wave_color = wave_color
        self.trail_colors = trail_colors
        self.glow_passes = glow_passes
        self.glow_alpha_decay = glow_alpha_decay
        self.line_width = line_width
        self.font_path = font_path
        self.visualizer = visualizer
        
        # Load research interests
        if research_interests_override is not None:
            self.research_interests = research_interests_override
        else:
            with open(research_interests_path, 'r') as f:
                self.research_interests = f.read().strip()
        
        # Store profile filtering context
        self.profile_ids_override = profile_ids_override
        
        # Inference models
        if isinstance(orchestration_config, str):
            # first see if this is a json string or a path to a file
            if orchestration_config.startswith('{') and orchestration_config.endswith('}'):
                self.orchestration_config = json.loads(orchestration_config)
            else:
                with open(orchestration_config, 'r') as f:
                    self.orchestration_config = json.load(f)
        elif isinstance(orchestration_config, dict):
            self.orchestration_config = orchestration_config
        else:
            raise ValueError("Invalid orchestration config")

        # Debug: Check the orchestration config before any modifications
        if self.verbose:
            print(f"[DEBUG] TheseusInsight received orchestration_config: {self.orchestration_config.get('arxiv_search_categories', 'NOT SET')}")
        
        # Ensure arxiv_search_categories exists with defaults if not present
        if 'arxiv_search_categories' not in self.orchestration_config:
            if self.verbose:
                print("[DEBUG] arxiv_search_categories not found, setting defaults")
            self.orchestration_config['arxiv_search_categories'] = {
                "main_category": "cs",
                "filter_categories": ["cs.ai", "cs.cl", "cs.lg", "cs.ir", "cs.ma", "cs.cv"]
            }
        else:
            if self.verbose:
                print("[DEBUG] arxiv_search_categories already exists, keeping current values")

        # 1) Embedding model
        self.embedding_model_name = self.orchestration_config['embedding_model']['model_name']
        self.embedding_model = SentenceTransformerInference(
            self.embedding_model_name, 
            remote_code=self.orchestration_config['embedding_model']['trust_remote_code']
        )
        
        # 2) Load specialized LLMs
        self.judge_model_config = self.orchestration_config['judge_model']
        self.content_extraction_model_config = self.orchestration_config['content_extraction_model']
        self.newsletter_sections_model_config = self.orchestration_config['newsletter_sections_model']
        self.newsletter_intro_model_config = self.orchestration_config['newsletter_intro_model']

        self.judge_inference = self._load_inference_model(
            self.judge_model_config['model_type'],
            self.judge_model_config['model_name'],
            self.judge_model_config['max_new_tokens'],
            self.judge_model_config['temperature'],
            self.judge_model_config.get('num_ctx'),
            host=self.judge_model_config.get('host')
        )
        self.content_extraction_inference = self._load_inference_model(
            self.content_extraction_model_config['model_type'],
            self.content_extraction_model_config['model_name'],
            self.content_extraction_model_config['max_new_tokens'],
            self.content_extraction_model_config['temperature'],
            self.content_extraction_model_config.get('num_ctx'),
            host=self.content_extraction_model_config.get('host')
        )
        self.newsletter_sections_inference = self._load_inference_model(
            self.newsletter_sections_model_config['model_type'],
            self.newsletter_sections_model_config['model_name'],
            self.newsletter_sections_model_config['max_new_tokens'],
            self.newsletter_sections_model_config['temperature'],
            self.newsletter_sections_model_config.get('num_ctx'),
            host=self.newsletter_sections_model_config.get('host')
        )
        self.newsletter_intro_inference = self._load_inference_model(
            self.newsletter_intro_model_config['model_type'],
            self.newsletter_intro_model_config['model_name'],
            self.newsletter_intro_model_config['max_new_tokens'],
            self.newsletter_intro_model_config['temperature'],
            self.newsletter_intro_model_config.get('num_ctx'),
            host=self.newsletter_intro_model_config.get('host'),
            request_timeout_sec=(
                NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC
                if self.newsletter_intro_model_config['model_type'] == "lmstudio"
                else None
            ),
        )

        # 3) Podcast model
        if self.generate_podcast:
            from theseus_insight.podcast.generator import PodcastGenerator
            self.podcast_inference = self.orchestration_config.get('podcast_model', None)
            if not self.podcast_inference:
                raise ValueError("Podcast model not set in orchestration config.")
            self.podcast_generator = PodcastGenerator(
                text_model=self.podcast_inference,
                tts_provider=self.orchestration_config.get('tts_model', {}).get('tts_provider', 'kokoro'),
                speaker_1_voice=self.orchestration_config.get('tts_model', {}).get('speaker_1_voice', 'af_bella'),
                speaker_1_speed=self.orchestration_config.get('tts_model', {}).get('speaker_1_speed', 1.0),
                speaker_2_voice=self.orchestration_config.get('tts_model', {}).get('speaker_2_voice', 'am_adam'),
                speaker_2_speed=self.orchestration_config.get('tts_model', {}).get('speaker_2_speed', 1.0),
                instructions_template=INSTRUCTION_TEMPLATES,
                intro_music_path=self.intro_music_path,
                verbose=self.verbose,
                db_url=data_path  # Pass the database URL
            )
        # 4) Arxiv search categories
        self.arxiv_main_category = self.orchestration_config['arxiv_search_categories']['main_category']
        self.arxiv_filter_categories = self.orchestration_config['arxiv_search_categories']['filter_categories']
        self.pdf_conversion_timeout_sec = PDF_CONVERSION_TIMEOUT_SEC
        self.pdf_download_max_workers = max(PDF_DOWNLOAD_MAX_WORKERS, 1)

    # PDF download/parse mechanics live in pdf/markdown_extraction.py (B8).
    def _download_pdf_to_temp_file(self, pdf_url: str) -> str:
        return download_pdf_to_temp_file(pdf_url, verbose=self.verbose)

    def _parse_downloaded_pdf_to_markdown(self, temp_pdf_path: str, source_pdf_url: str) -> str:
        return pdf_to_markdown(
            temp_pdf_path,
            source_pdf_url,
            timeout_sec=self.pdf_conversion_timeout_sec,
            verbose=self.verbose,
        )

    def _load_inference_model(
        self,
        model_type,
        model_name,
        max_new_tokens,
        temperature,
        num_ctx=None,
        host=None,
        request_timeout_sec=None,
    ):
        """Load the appropriate inference model (pipeline/model_loading.py, B8)."""
        try:
            return load_inference_model(
                model_type,
                model_name,
                max_new_tokens,
                temperature,
                num_ctx=num_ctx,
                host=host,
                request_timeout_sec=request_timeout_sec,
            )
        except Exception as e:
            self._log_error(500, e)
            raise

    def _log_error(self, status_code: int, error: Exception):
        """Helper to log errors to the database and optionally send an email."""
        import traceback
        error_msg = f"{type(error).__name__}: {str(error)}"
        
        # Log the full stack trace to console for debugging
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ERROR {status_code}: {error_msg}")
            print(f"{'='*60}")
            print("Full stack trace:")
            traceback.print_exc()
            print(f"{'='*60}\n")
        
        LogsRepository.upsert(
            task_id=self.task_id, 
            status=f"ERROR_{status_code}",
            datetime_run=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        # Send error notification email only once per run (if enabled)
        if self.send_error_notifications and not self.error_notified:
            try:
                self.communication.send_error_notification(error_msg)
                self.error_notified = True
            except Exception as e:
                print(f"Failed to send error notification: {str(e)}")

    async def _init_checkpoint_manager(self):
        """Initialize the checkpoint manager and create/resume job."""
        await self._checkpoints.init_db_job({
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "top_n": self.top_n,
            "cosine_threshold": self.cosine_similarity_threshold,
            "profile_ids": self.profile_ids_override,
            "task_id": self.task_id,
            "orchestration": self.orchestration_config,
            "research_interests": self.research_interests,
            "recipients": self.receiver_address,
        })

    # Checkpoint mechanics live in pipeline/checkpoints.py (B8); these
    # delegates keep the ~48 internal call sites unchanged.
    async def _save_checkpoint_async(self, stage: str, data: any):
        await self._checkpoints.save_async(stage, data)

    def _save_checkpoint(self, stage: str, data: any):
        self._checkpoints.save(stage, data)

    async def _load_checkpoint_async(self, stage: str) -> any:
        return await self._checkpoints.load_async(stage)

    def _load_checkpoint(self, stage: str) -> any:
        return self._checkpoints.load(stage)

    async def _cleanup_checkpoints_async(self):
        await self._checkpoints.cleanup_async()

    def _cleanup_checkpoints(self):
        self._checkpoints.cleanup()

    def _cleanup_temp_data(self):
        """Clean up temp_data folder (if it exists)."""
        try:
            temp_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'temp_data'
            )
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                if self.verbose:
                    print("Cleaned up temp_data directory")
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up temp_data directory: {cleanup_error}")
            self._log_error(500, cleanup_error)

    def _clear_judge_model_cache(self):
        """Clear model cache to resolve potential context issues (Ollama or LM Studio)."""
        try:
            if not hasattr(self, 'judge_inference'):
                return
                
            provider = getattr(self.judge_inference, 'provider', None)
            
            if provider == "ollama":
                model_name = self.judge_model_config.get('model_name')
                if model_name:
                    if self.verbose:
                        print(f"Clearing Ollama cache for judge model: {model_name}")
                    purge_ollama_cache(OLLAMA_URL, model_name)
                    
            elif provider == "lmstudio":
                # For LM Studio, verify the model is still loaded
                # SDK singleton can't be reset, so we just verify connection
                from theseus_insight.utils.lmstudio_client import verify_model_loaded
                model_name = self.judge_model_config.get('model_name')
                lmstudio_host = os.getenv('LMSTUDIO_HOST', 'localhost:1234')
                
                if self.verbose:
                    print(f"Verifying LM Studio model availability: {model_name}")
                    
                if not verify_model_loaded(host=lmstudio_host, model_name=model_name):
                    if self.verbose:
                        print(f"⚠️ LM Studio model {model_name} may not be loaded. Please check LM Studio.")
                    # Wait a bit for potential auto-reload
                    time.sleep(3)
                else:
                    if self.verbose:
                        print(f"✓ LM Studio model verified")
                        
        except Exception as e:
            if self.verbose:
                print(f"Failed to clear judge model cache: {e}")

    def _handle_no_papers_found(self, reason="no_papers_from_arxiv"):
        """Delegates to pipeline/profile_scoring.py (B9)."""
        return profile_scoring.handle_no_papers_found(self, reason=reason)

    def rank_papers_with_historical_scores(self, data_df, return_all_scored=False, progress_callback=None):
        """Delegates to pipeline/ranking.py (B9)."""
        return ranking.rank_papers_with_historical_scores(
            self, data_df, return_all_scored=return_all_scored, progress_callback=progress_callback)

    async def rank_papers_async(self, data_df):
        """Delegates to pipeline/ranking.py (B9)."""
        return await ranking.rank_papers_async(self, data_df)

    async def _rank_papers_multi_server(self, data_df):
        """Delegates to pipeline/ranking.py (B9)."""
        return await ranking.rank_papers_multi_server(self, data_df)

    def rank_papers(self, df, progress_callback=None):
        """Delegates to pipeline/ranking.py (B9)."""
        return ranking.rank_papers(self, df, progress_callback=progress_callback)

    def _rank_papers_single_server(self, df, progress_callback=None):
        """Delegates to pipeline/ranking.py (B9)."""
        return ranking.rank_papers_single_server(self, df, progress_callback=progress_callback)

    def get_profile_papers(self, profile_ids: List[int], min_score: float = 0.5) -> pd.DataFrame:
        """Delegates to pipeline/profile_scoring.py (B9)."""
        return profile_scoring.get_profile_papers(self, profile_ids, min_score=min_score)

    def get_and_score_profile_papers(
        self,
        profile_ids: List[int],
        embedded_df: pd.DataFrame = None,
        progress_callback: Optional[Callable[[str, float, str, dict], None]] = None
    ) -> pd.DataFrame:
        """Delegates to pipeline/profile_scoring.py (B9)."""
        return profile_scoring.get_and_score_profile_papers(
            self, profile_ids, embedded_df=embedded_df, progress_callback=progress_callback)

    def store_papers_without_scoring(self, data_df):
        """Delegates to pipeline/profile_scoring.py (B9)."""
        return profile_scoring.store_papers_without_scoring(self, data_df)

    def run(self, 
            start_from: str | None = None, 
            progress_callback: Callable[[str, float, str], None]|None = None
           ):
        """Synchronous wrapper for the async run method."""
        if progress_callback:
            self.progress_callback = progress_callback
        return asyncio.run(self.run_async(start_from, progress_callback))
    
    async def run_async(self, 
                       start_from: str | None = None, 
                       progress_callback: Callable[[str, float, str], None]|None = None
                      ):
        """
        Unified pipeline with checkpoints. 
        Harmonizes the old generate_newsletter_and_podcast() features:
         - Save newsletter to DB
         - Send email
         - Generate podcast (audio + optional visualizer)
         - Save dialogue JSON
         - Insert podcast into DB
        """
        data_df = None
        embedded_df = None
        top_n_df = None
        sections_data = None
        newsletter_content = None
        podcast_content = None
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING NEWSLETTER PIPELINE")
            print(f"   Task ID: {self.task_id}")
            print(f"   Use multi-server: {self.use_multi_server_judge}")
            print(f"   Date range: {self.start_date} to {self.end_date}")
            print(f"{'='*80}\n")
        
        try:
            from .observability import run_stage
            # Initialize checkpoint manager if using database checkpoints
            await self._init_checkpoint_manager()
            
            # -----------
            # Stage 1: Download Papers (pipeline/stages/download.py, B9)
            # -----------
            data_df, exit_early = await run_stage("download", download_stage.run, self, start_from, progress_callback)
            if exit_early:
                return

            # -----------
            # Stage 2: Embed Papers (pipeline/stages/embed.py, B9)
            # -----------
            embedded_df, exit_early = await run_stage("embed", embed_stage.run, self, data_df, start_from, progress_callback)
            if exit_early:
                return

            # -----------
            # Stage 3: Rank Papers (pipeline/stages/rank.py, B9)
            # -----------
            top_n_df = await run_stage("rank", rank_stage.run, self, embedded_df, start_from, progress_callback)
            embedded_df = None  # the stage releases embeddings; drop our reference too

            # -----------
            # Stage 4: Newsletter Sections (pipeline/stages/newsletter_sections.py, B9)
            # -----------
            sections_data = await run_stage("newsletter_sections", newsletter_sections_stage.run, self, top_n_df, start_from, progress_callback)

            # -----------
            # Stage 5: Newsletter Content (pipeline/stages/newsletter_content.py, B9)
            # -----------
            newsletter_content, sections_data = await run_stage("newsletter_content", newsletter_content_stage.run,
                self, sections_data, start_from, progress_callback
            )

            # -----------
            # Stage 6: Send Email (pipeline/stages/email.py, B9)
            # -----------
            newsletter_content, sections_data = await run_stage("email", email_stage.run,
                self, newsletter_content, sections_data, progress_callback
            )

            # -----------
            # Stage 7: Podcast (pipeline/stages/podcast.py, B9)
            # -----------
            await run_stage("podcast", podcast_stage.run, self, top_n_df, sections_data, progress_callback)

            # -----------
            # Final Step: Mark completion, purge + cleanup
            # -----------
            await self._save_checkpoint_async('newsletter_complete', {'status': 'complete'})
            # Try to purge the cache for all Ollama models used in the orchestration config
            try:
                # Collect all unique Ollama model names from the orchestration config
                ollama_models = set()
                for key, cfg in self.orchestration_config.items():
                    if isinstance(cfg, dict) and cfg.get('model_type', '').lower() == 'ollama':
                        model_name = cfg.get('model_name')
                        if model_name:
                            ollama_models.add(model_name)
                for model_name in ollama_models:
                    purge_ollama_cache(OLLAMA_URL, model_name)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to purge Ollama cache for some models: {e}")

            # Log final success
            LogsRepository.upsert(
                task_id=self.task_id,
                status="COMPLETED: Successfully completed Theseus Insight run",
                datetime_run=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

            # Only purge checkpoints when the run actually produced a newsletter.
            # An empty top_n_df means scoring failed silently or no papers met
            # criteria; keeping the file checkpoints lets a manual retry resume
            # from the closest viable stage instead of re-ingesting from scratch.
            produced_papers = (
                'top_n_df' in locals()
                and top_n_df is not None
                and len(top_n_df) > 0
            )
            if produced_papers:
                await self._cleanup_checkpoints_async()
            elif self.verbose:
                print(
                    "Skipping checkpoint cleanup: run completed with no papers ranked. "
                    "Checkpoints retained so a retry can resume."
                )

        except Exception as e:
            # Mark job as failed if using database checkpoints
            try:
                await self._checkpoints.fail_job(str(e))
            except Exception as checkpoint_error:
                if self.verbose:
                    print(f"Failed to mark job as failed: {checkpoint_error}")
            
            self._log_error(500, e)
            raise
        finally:
            self._cleanup_temp_data()

    def run_profiles_pipeline(self, progress_callback: Callable[[str, float, str], None]|None = None):
        """Delegates to pipeline/profiles_pipeline.py (B9)."""
        return profiles_pipeline.run(self, progress_callback)

    def run_embedding_only_pipeline(self, progress_callback: Callable[[str, float, str], None]|None = None):
        """Delegates to pipeline/embedding_pipeline.py (B9)."""
        return embedding_pipeline.run(self, progress_callback)

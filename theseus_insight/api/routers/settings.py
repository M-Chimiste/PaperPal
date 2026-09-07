from fastapi import APIRouter, HTTPException
from typing import Dict, List
import json
import os

from ..models import (
    OrchestrationConfig, ArxivCategoriesConfig, ModelProvider,
    ResearchInterests, EmailRecipients, VisualizerSettings,
    ModelConfig, TTSModelConfig, ResearchAgentModelConfigApi,
    MindMapConfig
)
from ..dependencies import CREDENTIAL_KEYS
from ...utils.path_resolver import get_config_path, config_file_exists
from ... import theseus_insight as ti_module
from ...data_access import SettingsRepository

from ..models import StatusMessageResponse

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/orchestration", response_model=OrchestrationConfig)
async def get_orchestration_config_api():
    """
    Retrieves the orchestration configuration, ensuring defaults for all fields including podcast and TTS.

    This endpoint fetches the orchestration configuration from the database.
    It returns the configuration with defaults for all fields including podcast and TTS.

    Returns:
        OrchestrationConfig: The orchestration configuration with defaults.

    Raises:
        HTTPException: If an error occurs while fetching the orchestration configuration.
    """
    try:
        db_config_json = SettingsRepository.get("orchestration")
        loaded_config_data = {}

        if db_config_json:
            loaded_config_data = json.loads(db_config_json)
        else:
            # Fallback to orchestration.json if DB is empty
            config_path = get_config_path('orchestration.json')
            if config_file_exists('orchestration.json'):
                with open(config_path, 'r') as f:
                    loaded_config_data = json.load(f)
        
        # Define comprehensive defaults that Pydantic models expect
        default_embedding_model = ModelConfig(model_name='Alibaba-NLP/gte-modernbert-base', model_type='sentence-transformers', trust_remote_code=True)
        default_judge_model = ModelConfig(model_name='phi4-mini:3.8b-q8_0', model_type='ollama', max_new_tokens=512, temperature=0.1, num_ctx=4096)
        default_content_extraction_model = ModelConfig(model_name='gemma3:27b-it-qat', model_type='ollama', max_new_tokens=4096, temperature=0.1, num_ctx=131072)
        default_newsletter_sections_model = ModelConfig(model_name='gemma3:27b-it-qat', model_type='ollama', max_new_tokens=4096, temperature=0.1, num_ctx=131072)
        default_newsletter_intro_model = ModelConfig(model_name='gemini-2.0-flash', model_type='gemini', max_new_tokens=4096, temperature=0.1, num_ctx=131072)
        default_podcast_model = ModelConfig(model_name='gemini-2.0-flash', model_type='gemini', max_new_tokens=8192, temperature=0.1, num_ctx=131072)
        default_tts_model = TTSModelConfig(tts_provider='openai', tts_model_name='tts-1', speaker_1_voice='sage', speaker_1_speed=1.0, speaker_2_voice='ash', speaker_2_speed=1.0)
        
        # Default Mind-Map configuration
        default_mind_map_config = MindMapConfig(
            k=15,
            similarity_threshold=0.3,
            layout_algorithm='force',
            summarization_model=ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=1024, temperature=0.3)
        )
        
        # Default Research Agent configuration
        default_research_agent_config = ResearchAgentModelConfigApi(
            boss_model=ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=4096, temperature=0.1),
            worker_models={
                'summary': ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=2048, temperature=0.1),
                'analysis': ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=2048, temperature=0.1),
                'search': ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=1024, temperature=0.1)
            },
            default_worker='summary',
            max_retries=3,
            max_research_context_tokens=15000,
            compress_to_ratio=0.2,
            initial_rerank_top_k=40,
            max_research_loops=3,
            query_planner_model=ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=1024, temperature=0.1),
            evidence_selector_model=ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=2048, temperature=0.1),
            compression_model=ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=4096, temperature=0.1),
            answer_generator_model=ModelConfig(model_name='gpt-4', model_type='openai', max_new_tokens=4096, temperature=0.3),
            arxiv_rate_limit=3.0,
        )

        # Create OrchestrationConfig by merging loaded data with defaults for missing top-level keys
        # Handle null values by using defaults when the loaded value is None
        embedding_model_data = loaded_config_data.get('embedding_model') or default_embedding_model.dict()
        judge_model_data = loaded_config_data.get('judge_model') or default_judge_model.dict()
        content_extraction_model_data = loaded_config_data.get('content_extraction_model') or default_content_extraction_model.dict()
        newsletter_sections_model_data = loaded_config_data.get('newsletter_sections_model') or default_newsletter_sections_model.dict()
        newsletter_intro_model_data = loaded_config_data.get('newsletter_intro_model') or default_newsletter_intro_model.dict()
        podcast_model_data = loaded_config_data.get('podcast_model') or default_podcast_model.dict()
        tts_model_data = loaded_config_data.get('tts_model') or default_tts_model.dict()
        mind_map_config_data = loaded_config_data.get('mind_map_config') or default_mind_map_config.dict()
        
        # Special handling for research agent config with nested null model configurations
        research_agent_raw = loaded_config_data.get('research_agent_model_config') or {}
        research_agent_model_config_data = default_research_agent_config.dict()
        research_agent_model_config_data.update(research_agent_raw)
        
        # Handle null nested model configurations in research agent config
        if research_agent_model_config_data.get('query_planner_model') is None:
            research_agent_model_config_data['query_planner_model'] = default_research_agent_config.query_planner_model
        if research_agent_model_config_data.get('evidence_selector_model') is None:
            research_agent_model_config_data['evidence_selector_model'] = default_research_agent_config.evidence_selector_model
        if research_agent_model_config_data.get('compression_model') is None:
            research_agent_model_config_data['compression_model'] = default_research_agent_config.compression_model
        if research_agent_model_config_data.get('answer_generator_model') is None:
            research_agent_model_config_data['answer_generator_model'] = default_research_agent_config.answer_generator_model

        final_config = OrchestrationConfig(
            embedding_model=ModelConfig(**embedding_model_data),
            judge_model=ModelConfig(**judge_model_data),
            content_extraction_model=ModelConfig(**content_extraction_model_data),
            newsletter_sections_model=ModelConfig(**newsletter_sections_model_data),
            newsletter_intro_model=ModelConfig(**newsletter_intro_model_data),
            podcast_model=ModelConfig(**podcast_model_data),
            tts_model=TTSModelConfig(**tts_model_data),
            research_agent_model_config=ResearchAgentModelConfigApi(**research_agent_model_config_data),
            mind_map_config=MindMapConfig(**mind_map_config_data)
        )
        return final_config

    except Exception as e:
        # Adding more context to the error for easier debugging
        error_detail = f"Error getting orchestration config: {str(e)}. DB JSON: {db_config_json if 'db_config_json' in locals() else 'Not fetched/Error'}. Loaded Data: {loaded_config_data if 'loaded_config_data' in locals() else 'Not loaded/Error'}."
        raise HTTPException(status_code=500, detail=error_detail)

@router.put("/orchestration")
async def update_orchestration_config_api(config: OrchestrationConfig):
    """
    Updates the orchestration configuration.

    This endpoint updates the orchestration configuration in the database.
    It also updates the orchestration.json file for legacy/fallback.

    Args:
        config (OrchestrationConfig): The orchestration configuration to update.

    Returns:
        dict: A dictionary containing the status and message of the update operation.

    Raises:
        HTTPException: If an error occurs while updating the orchestration configuration.
    """
    try:
        SettingsRepository.set("orchestration", config.json())
        # Also update orchestration.json for legacy/fallback
        config_path = get_config_path('orchestration.json')
        with open(config_path, 'w') as f:
            json.dump(json.loads(config.json()), f, indent=2)
        return {"status": "success", "message": "Orchestration configuration updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating orchestration config: {str(e)}")

@router.get("/arxiv-categories", response_model=ArxivCategoriesConfig)
async def get_arxiv_categories_api():
    """
    Retrieves the ArXiv search categories.

    This endpoint fetches the ArXiv search categories from the database.
    It returns the categories with defaults if not found in the database.
    
    Returns:
        ArxivCategoriesConfig: The ArXiv search categories.

    Raises:
        HTTPException: If an error occurs while fetching the ArXiv categories.
    """
    try:
        settings_json = SettingsRepository.get("arxiv_search_categories")
        if settings_json:
            return ArxivCategoriesConfig.parse_raw(settings_json)
        # Return default ArXivCategoriesConfig if not found in DB
        return ArxivCategoriesConfig(
            main_category="cs",
            filter_categories=["cs.ai", "cs.cl", "cs.lg", "cs.ir", "cs.ma", "cs.cv"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ArXiv categories: {str(e)}")

@router.put("/arxiv-categories")
async def update_arxiv_categories_api(config: ArxivCategoriesConfig):
    """
    Updates the ArXiv search categories.

    This endpoint updates the ArXiv search categories in the database.
    It returns a success message if the categories are updated successfully.

    Args:
        config (ArxivCategoriesConfig): The ArXiv search categories to update.

    Returns:
        dict: A dictionary containing the status and message of the update operation.

    Raises:
        HTTPException: If an error occurs while updating the ArXiv categories.
    """
    try:
        SettingsRepository.set("arxiv_search_categories", config.json())
        return {"status": "success", "message": "ArXiv categories updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating ArXiv categories: {str(e)}")

@router.get("/research-interests", response_model=ResearchInterests)
async def get_research_interests_api():
    """
    Retrieves the research interests.

    This endpoint fetches the research interests from the database.
    It returns the interests with defaults if not found in the database.

    Returns:
        ResearchInterests: The research interests.

    Raises:
        HTTPException: If an error occurs while fetching the research interests.
    """
    try:
        interests = SettingsRepository.get("research_interests")
        if interests is not None: # Check if DB returned a value (could be empty string)
            return ResearchInterests(interests=interests)
        else:
            # Fallback to research_interests.txt
            config_path = get_config_path('research_interests.txt')
            if config_file_exists('research_interests.txt'):
                with open(config_path, 'r') as f:
                    interests_from_file = f.read().strip()
                return ResearchInterests(interests=interests_from_file)
            else:
                # Default to empty string if neither DB nor file exists
                return ResearchInterests(interests="")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting research interests: {str(e)}")

@router.put("/research-interests", response_model=ResearchInterests)
async def update_research_interests_api(data: ResearchInterests):
    """
    Updates the research interests.

    This endpoint updates the research interests in the database.
    It also updates the research_interests.txt file for legacy/fallback.

    Args:
        data (ResearchInterests): The research interests to update.

    Returns:
        ResearchInterests: The updated research interests.

    Raises:
        HTTPException: If an error occurs while updating the research interests.
    """
    try:
        # Save to DB
        SettingsRepository.set("research_interests", data.interests)
        
        # Save to research_interests.txt
        config_path = get_config_path('research_interests.txt')
        try:
            with open(config_path, 'w') as f:
                f.write(data.interests)
        except IOError as e:
            # Log error but don't fail the request if DB save was successful
            print(f"Warning: Could not write to {config_path}: {e}")
            # Optionally, you could raise an HTTPException here if writing to file is critical
            # raise HTTPException(status_code=500, detail=f"Error saving research interests to file: {str(e)}")

        return data # Return the updated data
    except Exception as e:
        # Log the full error for debugging
        print(f"Full error updating research interests: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating research interests: {str(e)}")

@router.get("/email-recipients", response_model=EmailRecipients)
async def get_email_recipients():
    """
    Retrieves the email recipients list.

    This endpoint fetches the email recipients list from the database.
    It returns the recipients list with defaults if not found in the database.

    Returns:
        EmailRecipients: The email recipients list.

    Raises:
        HTTPException: If an error occurs while fetching the email recipients list.
    """
    try:
        recipients_list = SettingsRepository.get_email_recipients()
        return EmailRecipients(recipients=recipients_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/email-recipients", response_model=StatusMessageResponse)
async def update_email_recipients(data: EmailRecipients):
    """
    Updates the email recipients list.

    This endpoint updates the email recipients list in the database.
    It returns a success message if the recipients are updated successfully.
    
    Args:
        data (EmailRecipients): The email recipients list to update.

    Returns:
        dict: A dictionary containing the status and message of the update operation.

    Raises:
        HTTPException: If an error occurs while updating the email recipients list.
    """
    try:
        # Basic email validation (optional here if Pydantic model handles it, or keep for defense-in-depth)
        for email in data.recipients:
            if "@" not in email or "." not in email: # Basic check
                raise HTTPException(status_code=400, detail=f"Invalid email address: {email}")
        SettingsRepository.set_email_recipients(data.recipients)
        return {"status": "success", "message": "Email recipients updated successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualizer-settings", response_model=VisualizerSettings)
async def get_visualizer_settings():
    """
    Retrieves the visualizer settings.

    This endpoint fetches the visualizer settings from the database.
    It returns the settings with defaults if not found in the database.

    Returns:
        VisualizerSettings: The visualizer settings.

    Raises:
        HTTPException: If an error occurs while fetching the visualizer settings.
    """
    try:
        settings = SettingsRepository.get_visualizer_settings()
        if not settings:
            # Return default settings from the model
            return VisualizerSettings().dict()
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-test-email", response_model=StatusMessageResponse)
async def send_test_email():
    """
    Sends a test email to the email address in GMAIL_SENDER_ADDRESS environment variable.

    This endpoint sends a test email to the email address in GMAIL_SENDER_ADDRESS environment variable.
    It returns a success message if the email is sent successfully.

    Returns:
        dict: A dictionary containing the status and message of the test email operation.

    Raises:
        HTTPException: If an error occurs while sending the test email.
    """
    try:
        from ...communication import GmailCommunication
        
        # Initialize email client
        gmail_sender = os.getenv("GMAIL_SENDER_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        
        if not gmail_sender or not gmail_password:
            raise HTTPException(
                status_code=500,
                detail="Email credentials not configured"
            )
            
        comm = GmailCommunication(
            sender_address=gmail_sender,
            app_password=gmail_password,
            receiver_address=gmail_sender
        )
        
        # Send test email
        test_content = """
        This is a test email from Theseus Insight.
        
        If you're receiving this, your email configuration is working correctly.
        
        Best regards,
        Theseus Insight Team
        """
        
        from datetime import datetime
        comm.compose_message(
            content=test_content,
            start_date=datetime.now().date(),
            end_date=datetime.now().date(),
            subject="Theseus Insight - Test Email"
        )
        comm.send_email()

        return {"status": "success", "message": "Test email sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/credentials", response_model=Dict[str, dict])
async def get_credentials():
    from ...db.async_bridge import run_repo
    def read_status():
        return {key: {"configured": bool(SettingsRepository.get(key) or os.getenv(key)),
                      "value": (SettingsRepository.get(key) or os.getenv(key, "")) if key == "OLLAMA_URL" else ""}
                for key in CREDENTIAL_KEYS}
    return await run_repo(read_status)


@router.put("/credentials", response_model=StatusMessageResponse)
async def update_credentials(data: Dict[str, str]):
    from ...db.async_bridge import run_repo
    # Empty fields mean unchanged; clearing is explicit through DELETE.
    for key, value in data.items():
        if key not in CREDENTIAL_KEYS or not value:
            continue
        setter = SettingsRepository.set if key == "OLLAMA_URL" else SettingsRepository.set_secret_setting
        await run_repo(setter, key, value)
        os.environ[key] = value
        if hasattr(ti_module, key):
            setattr(ti_module, key, value)
    return {"status": "success"}


@router.delete("/credentials/{key}", response_model=StatusMessageResponse)
async def delete_credential(key: str):
    from ...db.async_bridge import run_repo
    if key not in CREDENTIAL_KEYS:
        raise HTTPException(status_code=404, detail="Unknown credential")
    await run_repo(SettingsRepository.delete, key)
    os.environ.pop(key, None)
    if hasattr(ti_module, key):
        setattr(ti_module, key, "")
    return {"status": "success"}

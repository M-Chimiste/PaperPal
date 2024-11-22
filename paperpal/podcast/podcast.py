import json
import os

from ..pdf import MarkdownParser, parse_pdf_to_markdown
from ..data_processing import ArxivData
from ..llm import AnthropicInference, GeminiInference, OpenAIInference, OllamaInference
from .tts import TTSInference
import json

MODEL_CONFIG = {
    "podcast_stage_1": {
        "model": "llama3.2",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "model_type": "ollama"
    },
    "podcast_stage_2": {
        "model": "hermes3",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "model_type": "ollama"
    },
    "podcast_stage_3": {
        "model": "hermes3",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "model_type": "ollama"
    },
    "podcast_stage_4": {
        "model": "TTS",
        "model_type": "parler"
    }
}


def get_model(model_config: dict):
    """
    Get the appropriate inference model based on the model configuration.

    Args:
        model_config (dict): A dictionary containing model configuration parameters.
            Must include 'model_type' key specifying the type of model to use
            ('ollama', 'openai', 'anthropic', or 'gemini').
            Must include 'model' key with the specific model name.
            Must include 'max_new_tokens' and 'temperature' for all types except 'parler'.

    Returns:
        An instance of the appropriate inference class (OllamaInference, OpenAIInference,
        AnthropicInference, or GeminiInference) initialized with the config parameters.

    Raises:
        ValueError: If an invalid model_type is provided.
    """
    if model_config["model_type"] == "ollama":
        return OllamaInference(model_config["model"], model_config["max_new_tokens"], model_config["temperature"])
    elif model_config["model_type"] == "openai":
        return OpenAIInference(model_config["model"], model_config["max_new_tokens"], model_config["temperature"])
    elif model_config["model_type"] == "anthropic":
        return AnthropicInference(model_config["model"], model_config["max_new_tokens"], model_config["temperature"])
    elif model_config["model_type"] == "gemini":
        return GeminiInference(model_config["model"], model_config["max_new_tokens"], model_config["temperature"])
    elif model_config["model_type"] == "parler":
        return TTSInference(model_config["model"])
    else:
        raise ValueError(f"Invalid model type: {model_config['model_type']}")


class PodcastGenerator:
    def __init__(self, 
                 pdf_urls: list[str], 
                 model_config: dict|str = MODEL_CONFIG):
        self.pdf_urls = pdf_urls
        if isinstance(model_config, str):
            try:
                with open(model_config, 'r') as f:
                    self.model_config = json.loads(f.read())
            except (json.JSONDecodeError, FileNotFoundError):
                raise ValueError("model_config must be either a valid path to a JSON file or a dictionary")
        
        self.model_config = model_config
        self.stage_1_model = get_model(self.model_config["podcast_stage_1"])
        self.stage_2_model = get_model(self.model_config["podcast_stage_2"])
        self.stage_3_model = get_model(self.model_config["podcast_stage_3"])
        self.tts_model = "TTS Model Stuff"
    
    
    
    def _run_stage_1(self, pdf_url: str) -> str:
        full_paper = ArxivData(pdf_url)
        markdown_data = parse_pdf_to_markdown(full_paper.pdf_path)
        parsed_data = MarkdownParser(markdown_data)
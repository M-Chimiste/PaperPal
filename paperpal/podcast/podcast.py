import json
import os

from ..pdf import MarkdownParser, parse_pdf_to_markdown
from ..data_processing import ArxivData
from ..llm import AnthropicInference, GeminiInference, OpenAIInference, OpenAIChatInference
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


class PodcastGenerator:
    def __init__(self, pdf_urls: list[str], model_config: dict|str = MODEL_CONFIG):
        self.pdf_urls = pdf_urls
        if isinstance(model_config, str):
            try:
                with open(model_config, 'r') as f:
                    self.model_config = json.loads(f.read())
            except (json.JSONDecodeError, FileNotFoundError):
                raise ValueError("model_config must be either a valid path to a JSON file or a dictionary")
        self.model_config = model_config
        
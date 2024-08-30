import os
import json
import datetime

import pandas as pd
from .communication import GmailCommunication
from paperpal.paperswithcode import ProcessData
from tqdm import tqdm
from dotenv import load_dotenv
from .prompts import *
from .inference import SentenceTransformerInference
from .utils import cosine_similarity, get_n_days_ago, TODAY
from .data_handling import PaperDatabase

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


class PaperPal:
    #TODO: Add the papers database
    #TODO: Add the data downloading with paperswithcode
    #TODO: Embed the abstracts
    #TODO: Apply cutoff
    #TODO: Run evaluation on papers
    #TODO: apply top_n cutoff
    #TODO: Generate newsletter
    #TODO: Send newsletter email
    def __init__(self,
                 research_interests_path="config/research_interests.txt",
                 n_days=7,
                 model_type="local",
                 model_name="NousResearch/Hermes-3-Llama-3.1-8B",
                 embedding_model_name="Alibaba-NLP/gte-base-en-v1.5",
                 trust_remote_code=True,
                 receiver_address=None,
                 max_new_tokens=1024,
                 temperature=0.1,
                 cosine_similarity_threshold=0.5,
                 data_path="data/"):
        self.research_interests_path = research_interests_path
        self.start_date = get_n_days_ago(n_days)
        self.end_date = TODAY
        self.model_type = model_type
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.communication = GmailCommunication(sender_address=os.getenv('GMAIL_SENDER_ADDRESS', None),
                                               app_password=os.getenv('GMAIL_APP_PASSWORD', None),
                                               receiver_address=receiver_address)
        self.embedding_model = SentenceTransformerInference(embedding_model_name, trust_remote_code=trust_remote_code)
        self.cosine_similarity_threshold = cosine_similarity_threshold
        # Load research interests
        try:
            with open(self.research_interests_path, 'r') as file:
                self.research_interests = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"The research interests file at {self.research_interests_path} could not be found. Please check the path and try again.")
        except IOError:
            raise IOError(f"There was an error reading the file at {self.research_interests_path}. Please check the file permissions and try again.")

        # Load model
        model_type = os.getenv("MODEL_TYPE", "local")
        if model_type == "local":
            from .inference import LocalCudaInference
            self.inference = LocalCudaInference(model_name, max_new_tokens, temperature)
        elif model_type == "anthropic":
            if ANTHROPIC_API_KEY is None:
                raise ValueError("Anthropic API key is not set. Please check your .env file and ensure ANTHROPIC_API_KEY is properly configured.")
            from .inference import AnthropicInference
            self.inference = AnthropicInference(model_name, max_new_tokens, temperature)
        elif model_type == "openai":
            if OPENAI_API_KEY is None:
                raise ValueError("OpenAI API key is not set. Please check your .env file and ensure OPENAI_API_KEY is properly configured.")
            from .inference import OpenAIInference
            self.inference = OpenAIInference(model_name, max_new_tokens, temperature)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of 'local', 'anthropic', or 'openai'.")
# Copyright 2023 M Chimiste

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, Any, Tuple

class BaseInference(ABC):
    """Base class for all inference implementations."""
    def __init__(self, 
                 model_name: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.client = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """Load and return the model/client."""
        pass

    @abstractmethod
    def invoke(self, messages: list, system_prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Generate a response using the model."""
        pass


class APIBasedInference(BaseInference):
    """Base class for API-based inference implementations."""
    @abstractmethod
    def _prepare_messages(self, messages: list, system_prompt: str) -> list:
        """Prepare messages in the format expected by the API."""
        pass


class LocalCudaInference(BaseInference):
    def __init__(self, model_name: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1,
                 repetition_penalty: float = 1.1):
        self.repetition_penalty = repetition_penalty
        super().__init__(model_name, max_new_tokens, temperature)

    def _load_model(self) -> Tuple[Any, Any]:
        import torch
        import torchao
        from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
        
        quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config
        )
        model = torch.compile(model, mode="max-autotune")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def invoke(self, messages: list, system_prompt: str, schema: Optional[BaseModel] = None) -> str:
        model, tokenizer = self.client
        messages = [{"role": "system", "content": system_prompt}] + messages
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty
        )
        return tokenizer.decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


class AnthropicInference(APIBasedInference):
    def _load_model(self):
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _prepare_messages(self, messages: list, system_prompt: str) -> list:
        return messages  # Anthropic handles messages as-is

    def invoke(self, messages: list, system_prompt: str, schema: Optional[BaseModel] = None) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.content


class OpenAIInference(APIBasedInference):
    def _load_model(self):
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _prepare_messages(self, messages: list, system_prompt: str) -> list:
        return [{"role": "system", "content": system_prompt}] + messages

    def invoke(self, messages: list, system_prompt: str, schema: Optional[BaseModel] = None) -> str:
        full_messages = self._prepare_messages(messages, system_prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class SentenceTransformerInference(BaseInference):
    """SentenceTransformerInference for generating embeddings."""
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self.trust_remote_code = trust_remote_code
        super().__init__(model_name)
    
    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code)
    
    def invoke(self, text: str, system_prompt: str = None):
        """Override to handle single text input instead of messages."""
        return self.client.encode(text, normalize_embeddings=True)


class OllamaInference(APIBasedInference):
    """OllamaInference for local Ollama API."""
    def __init__(self,
                 model_name: str = "hermes3",
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 url: str = "http://127.0.0.1:11434",
                 num_ctx: int = 131072):
        self.url = url
        self.num_ctx = num_ctx
        super().__init__(model_name, max_new_tokens, temperature)
    
    def _load_model(self):
        from ollama import Client
        return Client(host=self.url)
    
    def _prepare_messages(self, messages: list, system_prompt: str) -> list:
        return [{"role": "system", "content": system_prompt}] + messages
    
    def invoke(self, messages: list, system_prompt: str, schema: Optional[BaseModel] = None) -> str:
        full_messages = self._prepare_messages(messages, system_prompt)
        
        options = {
            "num_predict": self.max_new_tokens,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx
        }

        if schema:
            response = self.client.chat(
                model=self.model_name,
                messages=full_messages,
                format=schema.model_json_schema(),
                options=options
            )
        else:
            response = self.client.chat(
                model=self.model_name,
                messages=full_messages,
                options=options
            )
        return response['message']['content']


class GeminiInference(APIBasedInference):
    """GeminiInference for Google's Gemini API."""
    def __init__(self, 
                 model_name: str = "gemini-1.5-flash",
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1):
        self.safety = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        super().__init__(model_name, max_new_tokens, temperature)
    
    def _load_model(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return genai
    
    def _prepare_messages(self, messages: list, system_prompt: str) -> list:
        gemini_messages = [{"role": "user", "parts": [system_prompt]}]
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            gemini_messages.append({"role": role, "parts": [message["content"]]})
        return gemini_messages
    
    def invoke(self, messages: list, system_prompt: str, schema: Optional[BaseModel] = None) -> str:
        gemini_messages = self._prepare_messages(messages, system_prompt)
        
        model = self.client.GenerativeModel(model_name=self.model_name)
        response = model.generate_content(
            gemini_messages,
            safety_settings=self.safety,
            generation_config=self.client.types.GenerationConfig(
                max_output_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
        )
        return response.text
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

class LocalCudaInference:
    """
    LocalCudaInference class for generating responses using a pre-trained model on a local GPU.

    This class handles the initialization of the model and tokenizer, and provides methods
    to invoke the model for generating responses based on input messages.

    Attributes:
        model_name (str): The name of the pre-trained model to be used.
        max_new_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): The sampling temperature to use for generation.
        repetition_penalty (float): The repetition penalty to use for generation.
        model: The loaded model.
        tokenizer: The loaded tokenizer.

    Methods:
        __init__(model_name, max_new_tokens, temperature, repetition_penalty): Initializes the LocalCudaInference instance.
        _load_model(): Loads and returns the model and tokenizer.
        invoke(messages): Generates a response using the model.
    """
    
    def __init__(self, model_name,
                 max_new_tokens=1024,
                 temperature=0.1,
                 repetition_penalty=1.1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        import torch
        import torchao
        from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
        quantization_config = TorchAoConfig("int4_weight_only",
                                             group_size=128)
        model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                    device_map="auto",
                                                    quantization_config=quantization_config)
        model = torch.compile(model, mode="max-autotune")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
    

    def invoke(self, messages, system_prompt):
        """
        Invoke the model to generate a response based on the given messages.

        Args:
            messages (list): A list of dictionaries containing the conversation history.
                             Each dictionary should have 'role' and 'content' keys.

        Returns:
            str: The generated text response from the model.

        Note:
            This method applies the chat template to the messages, tokenizes the input,
            generates new tokens using the model, and then decodes the output.
        """
        messages = [{"role": "system", "content": system_prompt}] + messages
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs,
                                    max_new_tokens=self.max_new_tokens,
                                    temperature=self.temperature,
                                    repetition_penalty=self.repetition_penalty)
        generated_text = self.tokenizer.decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return generated_text
    

class AnthropicInference:
    """
    AnthropicInference class for generating responses using Anthropic's API.

    This class handles the initialization of the Anthropic client and provides methods
    to invoke the model for generating responses based on input messages and a system prompt.

    Attributes:
        model_name (str): The name of the Anthropic model to be used.
        max_new_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): The sampling temperature to use for generation.
        client: The Anthropic client instance.

    Methods:
        __init__(model_name, max_new_tokens, temperature): Initializes the AnthropicInference instance.
        _load_model(): Loads and returns the Anthropic client.
        invoke(messages, system_prompt): Generates a response using the Anthropic model.
    """
    def __init__(self, model_name,
                 max_new_tokens=1024,
                 temperature=0.1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    
    def _load_model(self):
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return client
    
    def invoke(self, messages, system_prompt):
        """
        Invoke the model to generate a response based on the given messages.

        Args:
            messages (list): A list of dictionaries containing the conversation history.
                             Each dictionary should have 'role' and 'content' keys.
            system_prompt (str): The system prompt to be used for the model.

        Returns:
            str: The generated text response from the model.
        """
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.content


class OpenAIInference:
    """
    OpenAIInference class for generating responses using OpenAI's API.

    This class handles the initialization of the OpenAI client and provides methods
    to invoke the model for generating responses based on input messages and a system prompt.

    Attributes:
        model_name (str): The name of the OpenAI model to be used.
        max_new_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): The sampling temperature to use for generation.
        client: The OpenAI client instance.

    Methods:
        __init__(model_name, max_new_tokens, temperature): Initializes the OpenAIInference instance.
        _load_model(): Loads and returns the OpenAI client.
        invoke(messages, system_prompt): Generates a response using the OpenAI model.
    """
    def __init__(self, model_name,
                 max_new_tokens=1024,
                 temperature=0.1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.client = self._load_model()
    
    def _load_model(self):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return client
    
    def invoke(self, messages, system_prompt):
        """
        Invoke the model to generate a response based on the given messages.

        Args:
            messages (list): A list of dictionaries containing the conversation history.
                             Each dictionary should have 'role' and 'content' keys.
            system_prompt (str): The system prompt to be used for the model.

        Returns:
            str: The generated text response from the model.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class SentenceTransformerInference:
    """
    SentenceTransformerInference class for generating embeddings using a pre-trained Sentence Transformer model.

    Attributes:
        model_name (str): The name of the pre-trained model to be used.
        model: The loaded Sentence Transformer model.

    Methods:
        _load_model(): Loads the Sentence Transformer model.
        invoke(text): Generates embeddings for the given text.
    """
    def __init__(self, model_name, trust_remote_code=False):
        self.model_name = model_name
        self.model = self._load_model(trust_remote_code)
    
    def _load_model(self, trust_remote_code):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name, trust_remote_code=trust_remote_code)
        return model
    
    def invoke(self, text):
        """
        Invoke the model to generate a response based on the given messages.

        Args:
            text (str): The text to be embedded.

        Returns:
            tensor: The tensor representation of the messages."""
        return self.model.encode(text, normalize_embeddings=True)
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

import torch
from transformers import (AutoModelForCausalLM,
                        AutoTokenizer,
                        BitsAndBytesConfig)


def load_model(model_name,
              device,
              num_gpus,
              load_8bit=False,
              load_4bit=False,
              debug=False):
    

    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    if load_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        kwargs["quantization_config"] = bnb_config
    
    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["quantization_config"] = bnb_config
   
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        low_cpu_mem_usage=True, **kwargs)

    if (device == "cuda" and num_gpus == 1):
        model.to(device)        

    if debug:
        print(model)

    return model, tokenizer


class Inference:
    def __init__(self, model_name,
                device="cuda",
                num_gpus=1,
                load_4bit=False,
                load_8bit=False,
                debug=False,
                apply_chat_template=False,
                system_prompt=None,
                prompt_template=None):
        self.model, self.tokenizer = load_model(model_name,
                                                device,
                                                num_gpus,
                                                load_8bit,
                                                load_4bit,
                                                debug)
        self.apply_chat_template = apply_chat_template
        self.system_prompt=system_prompt
        self.prompt_template=prompt_template


    def construct_prompt(self, text):
        """Method to constuct the appropriate LLM prompt."""
        if self.apply_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": text})
            formatted_prompt = self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        else:
            print("You are not using the apply chat template.  It's recommended you use a model with a chat template for best results.")
        if self.prompt_template:
            formatted_prompt = self.prompt_template.format(text)
        
        return formatted_prompt
    

    def construct_research_prompt(self, text, research_interests):
        """Method to create the research specific prompt.  Edit this to change behavior of PaperPal"""
        research_prompt = f"""I have the following research interests:
{research_interests}

Based on the content below delimited by <> does this paper directly relate to any of my research interests?
Respond in a json with the keys related (bool) and reasoning (str).

<{text}>
"""
        return research_prompt
    

    def generate(self, text, model_prompt, temp=0.6, top_p=0.9, max_tokens=512, terminators=None, **kwargs):
        """Method to generate LLM inference

        Args:
            text (str): Text to feed to model
            temp (int, optional): Temperature value. Defaults to 0.6.
            model_prompt (str): Type of prompt to construct.
            top_p (float, optional): Top-P Value. Defaults to 0.9.
            max_tokens (int, optional): Maximum number of token for . Defaults to 512
            **kwargs.

        Returns:
            str: Model inference
        """
        if not terminators:
            terminators = self.tokenizer.eos_token_id

        prompt = self.construct_prompt(text, model_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temp,
            top_p=top_p

        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

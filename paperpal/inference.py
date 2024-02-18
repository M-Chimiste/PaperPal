from sys import platform
from transformers import GenerationConfig
from .prompts import *
import torch


def load_model_mac(model_path):
    from llama_cpp import Llama

    model = Llama(model_path)
    return model


def load_model(model_name,
              device="cuda", 
              num_gpus="auto", 
              load_8bit=False,
              load_4bit=False,
              max_memory="16GiB"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import bitsandbytes as bnb
    
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.bfloat16}
        if load_8bit:
            kwargs["load_in_8bit"] = True
        if load_4bit:
            kwargs["quantization_config"] =  bnb.BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["device_map"] = "auto"
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: max_memory for i in range(num_gpus)},
                })

   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                low_cpu_mem_usage=True,
                                                **kwargs)

    return model, tokenizer


class Inference:
    def __init__(self, model_name, device="cuda", num_gpus="auto", load_4bit=True, load_8bit=False, llama_cpp=False):
        
        if platform() != "darwin" or not llama_cpp:
            self.platform = 'huggingface'
            self.model, self.tokenizer = load_model(model_name, device, num_gpus, load_8bit, load_4bit)
        
        else:
            self.platform = "llama-cpp"
            self.model = load_model_mac(model_name)



    def construct_prompt(self, text, model='wizard-vicuna'):
        """Method to constuct the appropriate Vicuna v1.1 prompt."""
        if model == "wizard-vicuna":
            formatted_prompt = f"""USER: {text}
            ASSISTANT:"""
        else:
            formatted_prompt = f"""SYSTEM:A chat between a curious human and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the human's questions.
HUMAN: {text}ASSISTANT:"""
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
    

    def generate(self,
                text,
                model_prompt,
                temp=0.5,
                top_k=40,
                top_p=0.75,
                num_beams=4, 
                max_tokens=512,
                repetition_penalty=1.2,
                **kwargs):
        """Method to generate LLM inference

        Args:
            text (str): Text to feed to model
            temp (int, optional): Temperature value. Defaults to 1.
            model_prompt (str): Type of prompt to construct.
            top_k (int, optional): Top-K Value. Defaults to 40.
            top_p (float, optional): Top-P Value. Defaults to 0.75.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            max_tokens (int, optional): Maximum number of token for . Defaults to 512
            **kwargs.

        Returns:
            str: Model inference
        """
        if self.platform == "huggingface":
            prompt = self.construct_prompt(text, model_prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            generation_config = GenerationConfig(
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                **kwargs,)
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_tokens)
            output = self.tokenizer.batch_decode(generation_output[:, input_ids.shape[1]:])[0]  #Need to test if this just returns the generated text
            # s = generation_output.sequences[0]
            # output = self.tokenizer.decode(s)
            # output = output.split("ASSISTANT:")[1].strip()
            return output.strip(self.tokenizer.eos_token)

        elif self.platform == "llama-cpp":
            # TODO implement llama-cpp inference.
            return

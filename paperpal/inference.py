# Lines 12 -> 153 Taken from or adapted from FastChat https://github.com/lm-sys/FastChat

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import dataclasses

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""
    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True

default_compression_config = CompressionConfig(
    num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True)


class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, weight, bias, device):
        super().__init__()

        self.weight = compress(weight.data.to(device), default_compression_config)
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        weight = decompress(self.weight, default_compression_config)
        return F.linear(input, weight, self.bias)


def compress_module(module, target_device):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(module, attr_str,
                CLinear(target_attr.weight, target_attr.bias, target_device))
    for name, child in module.named_children():
        compress_module(child, target_device)


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (original_shape[:group_dim] + (num_groups, group_size) +
                 original_shape[group_dim+1:])

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim+1:]
        tensor = torch.cat([
            tensor,
            torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim)
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim] +
            (original_shape[group_dim] + pad_len,) +
            original_shape[group_dim+1:])
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


def load_model(model_name, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
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

   
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        low_cpu_mem_usage=True, **kwargs)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1):
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


class Inference:

    def __init__(self, model_name, device="cuda", num_gpus=2, load_8bit=False, debug=False):
        self.model, self.tokenizer = load_model(model_name, device, num_gpus, load_8bit, debug)


    def construct_prompt(self, text):
        """Method to constuct the appropriate Vicuna v1.1 prompt."""
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
    

    def generate(self, text, temp=1, top_k=40, top_p=0.75, num_beams=4, max_tokens=512, **kwargs):
        """Method to generate LLM inference

        Args:
            text (str): Text to feed to model
            temp (int, optional): Temperature value. Defaults to 1.
            top_k (int, optional): Top-K Value. Defaults to 40.
            top_p (float, optional): Top-P Value. Defaults to 0.75.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            max_tokens (int, optional): Maximum number of token for . Defaults to 512
            **kwargs.

        Returns:
            str: Model inference
        """

        prompt = self.construct_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_config = GenerationConfig(
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,)
        
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_tokens)
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        output = output.split("ASSISTANT:")[1].strip()
        return output.strip("<//s>")

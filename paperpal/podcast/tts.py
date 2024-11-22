import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from pathlib import Path

class TTSInference:
    def __init__(self, output_dir="audio_output", voice_description=None, model_name="parler-tts/parler-tts-mini-v1"):
        """Initialize Parler TTS model"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Default voice description if none provided
        self.voice_description = voice_description or (
            "The voice is clear and professional, with natural pacing "
            "and a neutral tone, recorded in a quiet studio environment."
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_audio(self, text_segments):
        """
        Generate audio files from text segments using Parler TTS
        Args:
            text_segments (list): List of text strings to convert to speech
        Returns:
            list: Paths to generated audio files
        """
        audio_files = []
        
        # Convert description to input_ids once since it's the same for all segments
        input_ids = self.tokenizer(
            self.voice_description, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        for idx, text in enumerate(text_segments):
            # Prepare prompt
            prompt_input_ids = self.tokenizer(
                text, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Generate audio
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids
                )
            
            # Save the audio file
            audio_arr = generation.cpu().numpy().squeeze()
            output_path = self.output_dir / f"segment_{idx:04d}.wav"
            sf.write(str(output_path), audio_arr, self.model.config.sampling_rate)
            audio_files.append(output_path)
            
        return audio_files
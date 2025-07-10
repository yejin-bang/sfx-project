import torch
import numpy as np
import soundfile as sf
import time
import os
from pathlib import Path
import cv2

from video_processor import VideoProcessor
from walking_detector import WalkingDetector
from scene_analyzer import FinalSceneAnalyzer

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

class VideoToAudioPipeline:
    def __init__(self, output_dir="generated_audio", use_local_llm=False):
        print("Initializing Video-to-Audio Pipeline...")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.video_processor = VideoProcessor()
        self.walking_detector = WalkingDetector()
        self.scene_analyzer = FinalSceneAnalyzer(use_local_llm=use_local_llm)

        print("Loading Stable Audio model...")
        self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        print(f"✅ Audio model loaded on {self.device}")
        print(f"🎵 Sample rate: {self.model_config['sample_rate']}")
        print("✅ Pipeline initialized successfully!")
        print()


    def generate_audio_from_prompt(self, prompt, duration=5.0, steps=50):
        print(f"Generating audio: '{prompt}'")
        start_time = time.time()

        try:
            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": duration
            }]

            with torch.no_grad():
                output = generate_diffusion_cond(
                    model=self.model,
                    steps=steps,
                    cfg_scale=7,
                    conditioning=conditioning,
                    sample_rate=self.model_config["sample_rate"],
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device=self.device
                )

            audio_data = output.cpu().numpy()
            if len(audio_data.shape) == 3:
                audio_data = audio_data[0]

            if audio_data.shape[0] == 2:
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.shape[0] == 1:
                audio_data = audio_data[0]

            generation_time = time.time() - start_time
            actual_duration = len(audio_data) / self.model_config["sample_rate"]

            print(f"   ✅ Generated in {generation_time:.1f}s")
            print(f"   📊 Duration: {actual_duration:.1f}s")
            
            return audio_data, {
                'generation_time': generation_time,
                'duration': actual_duration,
                'sample_rate': self.model_config["sample_rate"]
            }

        except Exception as e:
            print(f"Audio Generateion failed: {e}")
            return None, None
        
    def save_audio(self, audio_data, filename, metadata=None):
        filepath = self.output_dir / filename
        sf.write(filepath, audio_data, self.model_config["sample_rate"])
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def process_video_segment(self, frames, timestamps, segment_id=0, audio_duration=5.0 steps=50):
        
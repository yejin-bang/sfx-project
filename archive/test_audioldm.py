# File: src/audio_generator.py
from diffusers import AudioLDMPipeline
import torch
import soundfile as sf
import os

class AudioGenerator:
    def __init__(self):
        print("Loading AudioLDM model...")
        print("This will download ~2GB model on first run...")
        
        self.pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(device)
        print(f"AudioLDM loaded on {device}")
        
    def generate_footsteps(self, prompt, duration=5.0, num_inference_steps=10):

        print(f"Generating: '{prompt}' ({duration}s)")
        
        try:
            audio = self.pipe(
                prompt, 
                num_inference_steps=num_inference_steps,
                audio_length_in_s=duration
            ).audios[0]
            
            print(f"✅ Generated {len(audio)} audio samples")
            return audio
            
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            return None
    
    def save_audio(self, audio, filename, sample_rate=16000):

        try:
            os.makedirs("generated_audio", exist_ok=True)
            filepath = f"generated_audio/{filename}"
            sf.write(filepath, audio, sample_rate)
            print(f"✅ Saved audio: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None
    
    def test_full_suite(self):
        """Test prompts matching your organized folders"""
        print("=== Full AudioLDM Test Suite ===\n")
        
        # Based on your organized folders
        test_prompts = [
            "boots on concrete",
            "heels on marble", 
            "sneakers on wood",
            "dress shoes on creaky floor",
            "boots on gravel",
            "heels on concrete",
            "sneakers on dirt",
            "footsteps on metal"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"Generating {i+1}/{len(test_prompts)}: {prompt}")
            audio = self.generate_footsteps(prompt, duration=3.0)
            
            if audio is not None:
                filename = f"test_{i+1:02d}_{prompt.replace(' ', '_')}.wav"
                self.save_audio(audio, filename)
            
            print()
        
        print("Full test suite complete!")
        print("Check generated_audio/ folder for all samples")

# Simple test
if __name__ == "__main__":
    try:
        generator = AudioGenerator()
        generator.test_full_suite()
        
    except Exception as e:
        print(f"❌ AudioLDM test failed: {e}")
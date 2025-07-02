
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
        """Generate footstep audio from text prompt"""
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
        """Save generated audio to file"""
        try:
            os.makedirs("generated_audio", exist_ok=True)
            filepath = f"generated_audio/{filename}"
            sf.write(filepath, audio, sample_rate)
            print(f"✅ Saved audio: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None
    
    def quick_test(self):
        """Generate one test file for you to listen to"""
        print("=== Quick AudioLDM Test ===")
        
        prompt = "footsteps on concrete"
        print(f"Testing: {prompt}")
        
        # Generate short test
        audio = self.generate_footsteps(prompt, duration=3.0)
        
        if audio is not None:
            filepath = self.save_audio(audio, "test_footsteps.wav")
            print(f"\n Test file saved: {filepath}")
            return True
        else:
            print("❌ Generation failed")
            return False

# Test the audio generator
if __name__ == "__main__":
    try:
        generator = AudioGenerator()
        generator.quick_test()
        
    except Exception as e:
        print(f"❌ AudioLDM test failed: {e}")
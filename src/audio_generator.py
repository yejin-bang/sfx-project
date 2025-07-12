import torch
import numpy as np
import soundfile as sf
import time
from pathlib import Path
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

class AudioGenerator:
    def __init__(self, model_name="stabilityai/stable-audio-open-1.0", device=None):
        print("Loading Stable Audio model...")

        self.model, self.model_config = get_pretrained_model(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"✅ Audio model loaded successfully!")
        print(f"🔧 Device: {self.device}")
        print(f"🎵 Sample rate: {self.model_config['sample_rate']}")
        print()

    def generate(self, prompt, duration=5.0, steps=50, cfg_scale=7,
                 sigma_min=0.3, sigma_max=500, sampler_type="dpmpp-3m-sde"):
        
        """
        Generate audio from text prompt
        
        Args:
            prompt (str): Text description of audio to generate
            duration (float): Length of audio in seconds
            steps (int): Number of diffusion steps (higher = better quality)
            cfg_scale (float): Classifier-free guidance scale
            sigma_min (float): Minimum noise level
            sigma_max (float): Maximum noise level
            sampler_type (str): Diffusion sampler type
            
        Returns:
            tuple: (audio_data, metadata) or (None, None) if failed
        """

        print(f"Generatating audio: '{prompt}', ({duration}s, {steps} steps)")
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
                    cfg_scale=cfg_scale,
                    conditioning=conditioning,
                    sample_rate=self.model_config["sample_rate"],
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    sampler_type=sampler_type,
                    device=self.device
                )
            
            audio_data = self._process_audio_output(output)

            generation_time = time.time() - start_time
            actual_duration = len(audio_data) /self.model_config["sample_rate"]
            max_amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data **2))

            metadata = {
                'prompt': prompt,
                'generation_time': generation_time,
                'duration': actual_duration,
                'sample_rate': self.model_config["sample_rate"],
                'max_amplitude': max_amplitude,
                'rms': rms,
                'steps': steps,
                'cfg_scale': cfg_scale,
                'success': True
            }

            print(f"   ✅ Generated in {generation_time:.1f}s")
            print(f"   📊 Duration: {actual_duration:.1f}s, RMS: {rms:.3f}")
            
            return audio_data, metadata
        
        except Exception as e:
            print(f" Audio generation failed: {e}")
            return None, {
                'prompt': prompt,
                'success': False,
                'error': str(e)
            }
        
    def _process_audio_output(self, output):
        """Process raw model output to clean audio array"""
        # Move to CPU and convert to numpy
        audio_data = output.cpu().numpy()

        # Remove batch dimension if present
        if len(audio_data.shape) == 3:
            audio_data = audio_data[0]

        if len(audio_data.shape) == 2:
            if audio_data.shape[0] == 2: # Stereo
                audio_data = np.mean(audio_data, axis=0)

            elif audio_data.shape[0] == 1: # Mono with channel dimension
                audio_data = audio_data[0]

        return audio_data
    
    def save_audio(self, audio_data, filepath, metadata=None):
        """
        Save audio to file
        
        Args:
            audio_data (np.array): Audio data to save
            filepath (str/Path): Where to save the file
            metadata (dict): Optional metadata to log
            
        Returns:
            str: Absolute path to saved file
        """

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        sf.write(filepath, audio_data, self.model_config["sample_rate"])
        print(f"Saved: {filepath}")
        return str(filepath.absolute())
    
    def generate_and_save(self, prompt, output_path, **generation_kwargs):

        audio_data, metadata = self.generate(prompt, **generation_kwargs)
        
        if audio_data is not None:
            filepath = self.save_audio(audio_data, output_path, metadata)
            metadata['file_path'] = filepath

        return metadata
    
    def batch_generate(self, prompts, output_dir, prefix="audio", **generation_kwargs):
        """
        Generate multiple audio files from list of prompts
        
        Args:
            prompts (list): List of text prompts
            output_dir (str/Path): Directory to save files
            prefix (str): Filename prefix
            **generation_kwargs: Additional arguments for generate()
            
        Returns:
            list: List of generation results
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, prompt in enumerate(prompts):

            clean_prompt = prompt.replace(' ', '_').replace(',','').replace('/','_')[:30]
            filename = f"{prefix}_{i:03d}_{clean_prompt}.wav"
            filepath = output_dir / filename

            result = self.generate_and_save(prompt, filepath, **generation_kwargs)
            results.append(result)
            print()

        successful = [r for r in results if r.get('success', False)]
        print(f"Batch generation complete!")

    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'device': self.device,
            'sample_rate': self.model_config['sample_rate'],
            'model_config': self.model_config
        }
    
if __name__ == "__main__":
    generator = AudioGenerator()

    prompt = "sneakers on gravel path"
    audio_data, metadata = generator.generate(prompt, duration=3.0, steps=25)

    if audio_data is not None:
        test_dir = Path("test_audio_output")
        test_file = test_dir / "test_generation.wav"
        generator.save_audio(audio_data, test_file, metadata)

        print(f"Test audio saved: {test_file}")
        print("Audio generation test successful")

    else:
        print("Audio generation test failed")

        

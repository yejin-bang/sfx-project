import torch
import numpy as np
import soundfile as sf
import time
import os
from pathlib import Path

output_dir = Path("footstep_tests")
output_dir.mkdir(exist_ok=True)

try:
    print("📥 Loading Stable Audio model...")
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✅ Model loaded successfully!")
    print(f"🔧 Device: {device}")
    print(f"🎵 Sample rate: {model_config['sample_rate']}")
    print()
    
    test_prompts = [
        "sneakers on gravel path"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"🎵 Generating audio {i+1}/{len(test_prompts)}: '{prompt}'")
        start_time = time.time()
        
        try:
            # Create conditioning
            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": 5.0  
            }]
            
            # Generate audio
            with torch.no_grad():
                output = generate_diffusion_cond(
                    model=model,
                    steps=25,  
                    cfg_scale=7,
                    conditioning=conditioning,
                    sample_rate=model_config["sample_rate"],
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device=device
                )
            
            # Process audio
            audio_data = output.cpu().numpy()
            if len(audio_data.shape) == 3:
                audio_data = audio_data[0]  # Remove batch dimension
            
            # Convert to mono if stereo
            if audio_data.shape[0] == 2:
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.shape[0] == 1:
                audio_data = audio_data[0]
            
            print(f"🔍 Raw output shape: {output.shape}")
            print(f"🔍 Audio data shape: {audio_data.shape}")
            print(f"🔍 Sample rate: {model_config['sample_rate']}")
            print(f"🔍 Expected duration: 5.0s")
            print(f"🔍 Actual duration: {len(audio_data) / model_config['sample_rate']:.1f}s")

            generation_time = time.time() - start_time
            
            # Save audio file
            filename = f"test_{i:02d}_{prompt.replace(' ', '_').replace(',', '')[:30]}.wav"
            filepath = output_dir / filename
            
            sf.write(filepath, audio_data, model_config["sample_rate"])
            
            # Calculate some basic audio stats
            duration = len(audio_data) / model_config["sample_rate"]
            max_amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            result = {
                'prompt': prompt,
                'file': str(filepath),
                'duration': duration,
                'generation_time': generation_time,
                'max_amplitude': max_amplitude,
                'rms': rms,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Generated in {generation_time:.1f}s")
            print(f"   📁 Saved: {filename}")
            print(f"   📊 Duration: {duration:.1f}s, RMS: {rms:.3f}")
            print()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'prompt': prompt,
                'success': False,
                'error': str(e)
            })
            print()
    
    # Summary
    print("🎉 TESTING COMPLETE!")
    print("=" * 60)
    print(f"📁 Audio files saved in: {output_dir.absolute()}")
    
    successful_tests = [r for r in results if r.get('success', False)]
    print(f"✅ Successful generations: {len(successful_tests)}/{len(test_prompts)}")
    
    if successful_tests:
        avg_time = sum(r['generation_time'] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  Average generation time: {avg_time:.1f}s")
        print(f"🎧 Generated: {successful_tests[0]['file']}")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're in the audio_test environment with stable-audio-tools installed")

except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete! Listen to the audio files and let me know the quality!")
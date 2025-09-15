from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import cv2
from PIL import Image

try: 
    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f'BLIP-2 loaded on {device}')

    video_path = 'data/test_videos/walk1.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        print(f"Image size: {pil_image.size}")

        prompt = "What do you see?"  # Simpler prompt
        print(f"Prompt: '{prompt}'")
        
        inputs = processor(pil_image, prompt, return_tensors="pt").to(device)
        print(f"Input shape: {inputs['pixel_values'].shape if 'pixel_values' in inputs else 'No pixel_values'}")

        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=50,  # More tokens
                do_sample=False,    # Deterministic
                num_beams=1         # Simple generation
            )

        print(f"Output shape: {output.shape}")
        print(f"Input length: {inputs['input_ids'].shape[1] if 'input_ids' in inputs else 'No input_ids'}")
        print(f"Output length: {output.shape[1]}")
        
        # Decode everything
        full_response = processor.decode(output[0], skip_special_tokens=True)
        print(f"Full response: '{full_response}'")
        
        # Try to get just new tokens
        if 'input_ids' in inputs:
            input_length = inputs['input_ids'].shape[1]
            new_tokens = output[0][input_length:]
            new_response = processor.decode(new_tokens, skip_special_tokens=True)
            print(f"New tokens only: '{new_response}'")
        else:
            print("No input_ids found")

    else:
        print("Could not read video")

except Exception as e:
    print(f"BLIP-2 failed: {e}")
    import traceback
    traceback.print_exc()
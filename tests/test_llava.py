print("Testing LLaVA model loading...")

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    import torch
    
    print("Step 1: Loading processor...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    print("✅ Processor loaded")
    
    print("Step 2: Loading model...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", 
        torch_dtype=torch.float32
    )
    print("✅ Model loaded")
    
    print("Step 3: Moving to device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Model on {device}")
    
    print("🎉 LLaVA loading test successful!")
    
except Exception as e:
    print(f"❌ LLaVA loading failed at: {e}")
    import traceback
    traceback.print_exc()
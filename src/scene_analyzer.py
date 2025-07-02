# File: src/scene_analyzer_final.py
import clip
import torch
import cv2
from PIL import Image
import random
# import openai  # Remove this line
# import os      # Keep this if needed for other things

class SceneAnalyzer:
    def __init__(self, use_openai=False):
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"✅ CLIP loaded on {self.device}")
        
        self.use_openai = use_openai
        if use_openai:
            print("⚠️ OpenAI not available - using simple context analysis")
            self.use_openai = False  # Force to False
    
    def analyze_surface_with_clip(self, frame):
        """Use CLIP to classify surface type"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            surface_options = [
                "concrete floor",
                "wooden floor", 
                "marble floor",
                "gravel ground",
                "dirt path",
                "metal surface",
                "creaky wooden floor",
                "deck boards"
            ]
            
            text_tokens = clip.tokenize(surface_options).to(self.device)
            
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            best_idx = probs.argmax()
            full_surface = surface_options[best_idx]
            surface = full_surface.split()[0]
            confidence = probs[0][best_idx]
            
            print(f"CLIP detected surface: {surface} (confidence: {confidence:.2f})")
            return surface, confidence
            
        except Exception as e:
            print(f"❌ Error with CLIP: {e}")
            return "concrete", 0.5
    
    def analyze_context_with_text_llm(self, frame):
        """Simple context analysis (no API needed)"""
        return self._simple_context_analysis()
    
    def _simple_context_analysis(self):
        """Fallback context analysis without API"""
        contexts = ["indoor office", "outdoor street", "home interior"]
        selected = random.choice(contexts)
        print(f"Context guess: {selected}")
        return selected
    
    def guess_shoe_type(self, surface, context):
        """Smart shoe guessing based on surface and context"""
        if "office" in context:
            return random.choice(['dress_shoes', 'heels'])
        elif "outdoor" in context or "street" in context:
            return random.choice(['boots', 'sneakers'])
        elif "home" in context:
            return random.choice(['sneakers'])
        elif surface == "marble":
            return random.choice(['dress_shoes', 'heels'])
        elif surface == "gravel" or surface == "dirt":
            return "boots"
        else:
            return "sneakers"
    
    def create_audio_prompt(self, frame):
        """Complete analysis to create AudioLDM prompt"""
        print("Analyzing scene with CLIP...")
        
        surface, confidence = self.analyze_surface_with_clip(frame)
        context = self.analyze_context_with_text_llm(frame)
        shoe_type = self.guess_shoe_type(surface, context)
        
        prompt = f"{shoe_type} on {surface}"
        
        print(f"Final audio prompt: '{prompt}'")
        return prompt

# Test
if __name__ == "__main__":
    try:
        analyzer = SceneAnalyzer(use_openai=False)
        
        video_path = 'data/test_videos/walk1.mp4'
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            prompt = analyzer.create_audio_prompt(frame)
            print(f"\n🎉 Scene analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
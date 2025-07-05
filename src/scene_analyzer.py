# File: src/scene_analyzer_final.py
import clip
import torch
import cv2
from PIL import Image
import ollama
import os
import random

class FinalSceneAnalyzer:
    def __init__(self, use_local_llm=False):  # Default to False now
        print("🚀 Loading Final Scene Analyzer...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"✅ CLIP loaded on {self.device}")
        

        self.use_local_llm = use_local_llm
        if use_local_llm:
            self.test_local_llm()
        
        # Environment-specific sound mapping algorithm
        self.environment_sound_map = {
            "indoor office space": {
                "dress_shoes": ["tapping", "clicking", "clacking"],
                "sneakers": ["soft padding", "muffled steps"]
            },
            "indoor home interior": {
                "sneakers": ["padding", "shuffling", "soft steps"], 
                "barefoot": ["patting", "soft padding", "quiet steps"]
            },
            "outdoor city street": {
                "sneakers": ["urban shuffling", "pavement padding", "street walking"],
                "boots": ["solid thudding", "concrete stomping", "heavy steps"]
            },
            "outdoor hiking trail": {
                "boots": ["trail crunching", "dirt thudding", "earthy stomping"]
            },
            "outdoor beach area": {
                "barefoot": ["soft patting", "gentle steps", "beach walking"],
                "sneakers": ["muffled stomping", "soft crunching", "cushioned steps"]
            }
        }
        

    
    def test_local_llm(self):
        """Test if Ollama is working"""
        try:
            response = ollama.generate(
                model='llama3.1:8b',
                prompt='Test',
                options={"num_predict": 5}
            )
            print("✅ Local LLM (Ollama) working")
            return True
        except Exception as e:
            print(f"⚠️ Local LLM failed: {e}")
            self.use_local_llm = False
            return False
    
    def analyze_environment_with_clip(self, frame):

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            

            environment_options = [
                "indoor office space",
                "indoor home interior",
                "outdoor city street", 
                "outdoor hiking trail",
                "outdoor beach area"
            ]
            
            environment, confidence = self._clip_classify(image, environment_options) # one image, 5 options
            
            print(f"🌍 Environment: {environment} (confidence: {confidence:.3f})")
            return environment, confidence
            
        except Exception as e:
            print(f"❌ Environment analysis failed: {e}")
            return "outdoor city street", 0.5
    
    def get_contextual_footwear(self, environment, use_random=True):
        """Smart contextual footwear detection based on environment"""

        
        environment_footwear_map = {
            "indoor office space": ["dress_shoes", "sneakers"],
            "indoor home interior": ["sneakers", "barefoot"], 
            "outdoor city street": ["sneakers", "boots"],
            "outdoor hiking trail": ["boots"],  
            "outdoor beach area": ["barefoot"]
        }
        
        possible_shoes = environment_footwear_map.get(environment, ["sneakers"])
        
        if use_random and len(possible_shoes) > 1:

            selected_footwear = random.choice(possible_shoes)
            print(f"👟 Randomly selected: {selected_footwear}")
            print(f"   (From options: {possible_shoes})")
        else:

            selected_footwear = possible_shoes[0]
            print(f"👟 Default footwear: {selected_footwear}")
            print(f"   (Other options: {possible_shoes[1:] if len(possible_shoes) > 1 else 'none'})")
        
        return selected_footwear
    
    def get_contextual_sound(self, environment, footwear):
        """Get environment-appropriate sound verb using smart algorithm"""
        
        # Get sound options for this environment + footwear combination
        env_sounds = self.environment_sound_map.get(environment, {})
        sound_options = env_sounds.get(footwear, [])
        
        if sound_options:
            # Randomly pick from contextual options for variety
            selected_sound = random.choice(sound_options)
            print(f"Contextual sound: '{selected_sound}' from {sound_options}")
            return selected_sound
        else:
            # Fallback to generic sound if no mapping exists
            fallback_sounds = {
                'dress_shoes': 'tapping',
                'boots': 'thudding',
                'sneakers': 'padding',
                'barefoot': 'patting'
            }
            fallback = fallback_sounds.get(footwear, 'stepping')
            print(f"🔄 Fallback sound: '{fallback}' (no contextual mapping)")
            return fallback
    
    def get_contextual_surface(self, environment):
        """Map environment to most likely surface from my dataset"""
        
        environment_surface_map = {
            "indoor office space": "marble",
            "indoor home interior": "wood",
            "outdoor city street": "concrete",
            "outdoor hiking trail": "dirt", 
            "outdoor beach area": "sand"
        }
        
        surface = environment_surface_map.get(environment, "concrete")
        
        print(f"🏗️ Contextual surface: {surface}")
        return surface
    
    def _clip_classify(self, image, options):

        text_tokens = clip.tokenize(options).to(self.device)
        
        with torch.no_grad():
            logits_per_image, _ = self.model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        best_idx = probs.argmax()
        confidence = float(probs[0][best_idx])
        
        return options[best_idx], confidence
    
    def generate_smart_audio_prompt(self, environment, surface, footwear):
        """Generate audio prompt using smart environment-sound algorithm"""
        
        # Get contextual sound verb
        sound_verb = self.get_contextual_sound(environment, footwear)
        
        # Get surface modifier from my dataset
        surface_modifiers = {
            'concrete': 'solid',
            'deck': 'hollow wooden',  
            'dirt': 'muffled',
            'gravel': 'crunching',
            'leaves': 'rustling',
            'marble': 'sharp',
            'metal': 'ringing',
            'mud': 'squelching',
            'sand': '',  # No modifier for sand (sounds already descriptive)
            'snow': 'crunching',
            'wet': 'splashing',
            'wood': 'hollow'
        }
        
        modifier = surface_modifiers.get(surface, '')
        
        # Build prompt with better grammar: [footwear] with [modifier] [sound_verb] on [surface]
        if modifier:
            prompt = f"{footwear.replace('_', ' ')} with {modifier} {sound_verb} on {surface}"
        else:
            prompt = f"{footwear.replace('_', ' ')} {sound_verb} on {surface}"
        
        print(f"🎯 Smart algorithm prompt: '{prompt}'")
        return prompt
 
        """Generate dry footstep audio prompt using local LLM"""
        if not self.use_local_llm:
            return self._rule_based_prompt(surface, footwear)
        
        try:
            prompt = f"Give me a concise and direct {footwear} on {surface} description with audio-focused point of view. Make it simple."

            response = ollama.generate(
                model='llama3.1:8b',
                prompt=prompt,
                options={
                    "num_predict": 12,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "stop": ["\n"]
                }
            )
            
            enhanced_prompt = response['response'].strip()
            
            # Clean up response
            if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
                enhanced_prompt = enhanced_prompt[1:-1]
            
            # Remove any environmental words that snuck in
            env_words = ["with", "in", "reverb", "echo", "ambiance", "acoustic"]
            for word in env_words:
                if word in enhanced_prompt.lower():
                    enhanced_prompt = enhanced_prompt.split(word)[0].strip()
                    break
            
            # Fallback to rule-based if LLM gives bad output
            if len(enhanced_prompt.split()) < 3:
                print(f"⚠️ LLM gave short output: '{enhanced_prompt}', using rules")
                return self._rule_based_prompt(surface, footwear)
            
            print(f"🤖 LLM enhanced: '{enhanced_prompt}'")
            return enhanced_prompt
            
        except Exception as e:
            print(f"❌ LLM enhancement failed: {e}")
            return self._rule_based_prompt(surface, footwear)
    
    def _rule_based_prompt(self, surface, footwear):
        """Reliable rule-based prompts as fallback"""
        
        # Sound characteristics for each footwear type
        sound_map = {
            'dress_shoes': 'tapping',
            'high_heels': 'clicking',
            'boots': 'thudding',
            'sneakers': 'padding',
            'running_shoes': 'bouncing',
            'sandals': 'slapping',
            'barefoot': 'patting'
        }
        
        # Surface-specific modifiers based on your dataset
        surface_modifiers = {
            'concrete': 'solid',
            'deck': 'hollow wooden',  
            'dirt': 'muffled',
            'gravel': 'crunching',
            'leaves': 'rustling',
            'marble': 'sharp',
            'metal': 'ringing',
            'mud': 'squelching',
            'sand': 'soft',
            'snow': 'crunching',
            'wet': 'splashing',
            'wood': 'hollow'
        }
        
        sound = sound_map.get(footwear, 'stepping')
        modifier = surface_modifiers.get(surface, '')
        
        if modifier:
            prompt = f"{modifier} {footwear.replace('_', ' ')} {sound} on {surface}"
        else:
            prompt = f"{footwear.replace('_', ' ')} {sound} on {surface}"
        
        print(f"🛠️ Rule-based: '{prompt}'")
        return prompt
    
    def analyze_scene(self, frame):
        """Main interface - complete scene analysis with clean logic"""
        print("=" * 50)
        print("🔍 Starting scene analysis...")

        
        # Step 1: Environment detection (CLIP's strength)
        environment, env_conf = self.analyze_environment_with_clip(frame)
        
        # Step 2: Contextual footwear logic (your domain knowledge)
        footwear = self.get_contextual_footwear(environment, use_random=True)
        
        # Step 3: Contextual surface mapping (your dataset categories)
        surface = self.get_contextual_surface(environment)
        
        print(f"\n📊 Final Analysis:")
        print(f"   Environment: {environment} ({env_conf:.3f} confidence)")
        print(f"   Surface: {surface}")
        print(f"   Footwear: {footwear}")
        
        # Step 4: Generate audio prompt using smart algorithm
        audio_prompt = self.generate_smart_audio_prompt(environment, surface, footwear)
        
        print(f"\nFinal audio prompt: '{audio_prompt}'")
        print("=" * 50)
        
        return {
            'environment': environment,
            'surface': surface,
            'footwear': footwear,
            'audio_prompt': audio_prompt,
            'confidence': env_conf
        }

# TEST WITH FRAME 60
if __name__ == "__main__":
    
    analyzer = FinalSceneAnalyzer(use_local_llm=False)
    video_path = 'data/test_videos/walk3.mp4'

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open video file")
        print("Current directory:", os.getcwd())
    
    else:

        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("✅ Successfully loaded frame 60")
            print(f"Frame dimensions: {frame.shape}")
            
            # Run complete analysis
            result = analyzer.analyze_scene(frame)
            
            print(f"This prompt is ready for AudioLDM!")
            
        else:
            print("❌ Could not read frame 60")
            print("Try checking if the video file exists and is readable")
import clip
import torch
import cv2
from PIL import Image
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import time

class MultiSegmentSceneAnalyzer:
    def __init__(self, seed=42):
        print("Loading Multi-Segment Scene Analyzer...")
        
        # Set seed for reproducibility
        self.set_seed(seed)
        
        # Initialize CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"CLIP loaded on {self.device}")
        
        # 15 environment categories optimized for CLIP
        self.environment_options = [
            # Indoor (7개)
            "indoor office workspace",
            "indoor home residential", 
            "indoor kitchen",
            "indoor shopping retail",
            "indoor restaurant cafe",
            "indoor gym sports facility", 
            "indoor hallway corridor",
            
            # Outdoor urban (4개)
            "outdoor city street",
            "outdoor sidewalk pavement", 
            "outdoor parking lot",
            "outdoor urban plaza",
            
            # Outdoor natural (4개)
            "outdoor beach sand",
            "outdoor forest trail",
            "outdoor park grass", 
            "outdoor dirt gravel path"
        ]
        
        # Environment to surface mapping
        self.environment_surface_map = {
            "indoor office workspace": "carpet",
            "indoor home residential": "wood", 
            "indoor kitchen": "tile",
            "indoor shopping retail": "tile",
            "indoor restaurant cafe": "wood",
            "indoor gym sports facility": "rubber",
            "indoor hallway corridor": "marble",
            
            "outdoor city street": "concrete",
            "outdoor sidewalk pavement": "concrete",
            "outdoor parking lot": "asphalt",
            "outdoor urban plaza": "stone",
            
            "outdoor beach sand": "sand",
            "outdoor forest trail": "dirt",
            "outdoor park grass": "grass",
            "outdoor dirt gravel path": "gravel"
        }
        
        # Environment to footwear mapping
        self.environment_footwear_map = {
            "indoor office workspace": ["dress_shoes", "sneakers"],
            "indoor home residential": ["sneakers", "barefoot", "slippers"], 
            "indoor kitchen": ["sneakers", "barefoot"],
            "indoor shopping retail": ["sneakers", "dress_shoes"],
            "indoor restaurant cafe": ["dress_shoes", "sneakers"],
            "indoor gym sports facility": ["running_shoes", "sneakers"],
            "indoor hallway corridor": ["sneakers", "dress_shoes"],
            
            "outdoor city street": ["sneakers", "boots", "dress_shoes"],
            "outdoor sidewalk pavement": ["sneakers", "running_shoes"],
            "outdoor parking lot": ["sneakers", "boots"],
            "outdoor urban plaza": ["sneakers", "dress_shoes"],
            
            "outdoor beach sand": ["barefoot", "sandals"],
            "outdoor forest trail": ["boots", "hiking_boots"],
            "outdoor park grass": ["sneakers", "running_shoes"],
            "outdoor dirt gravel path": ["boots", "sneakers"]
        }
        
        # Environment + footwear to sound mapping
        self.environment_sound_map = {
            "indoor office workspace": {
                "dress_shoes": ["crisp tapping", "sharp clicking"],
                "sneakers": ["soft padding", "muffled steps"]
            },
            "indoor home residential": {
                "sneakers": ["gentle padding", "soft steps"], 
                "barefoot": ["soft patting", "quiet footfalls"],
                "slippers": ["soft shuffling", "gentle sliding"]
            },
            "outdoor city street": {
                "sneakers": ["urban padding", "street walking"],
                "boots": ["solid thudding", "heavy footfalls"],
                "dress_shoes": ["pavement clicking", "urban tapping"]
            },
            "outdoor beach sand": {
                "barefoot": ["sand crunching", "soft sandy steps"],
                "sandals": ["sandy slapping", "beach walking"]
            },
            "outdoor forest trail": {
                "boots": ["trail crunching", "forest stomping"],
                "hiking_boots": ["rugged stomping", "earthy thudding"]
            }
        }
        
        # Scene change detection parameters
        self.scene_change_threshold = 0.15  # Structural similarity threshold
        
    def set_seed(self, seed):
        """Set seed for reproducible results"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def analyze_video_segments(self, frames, timestamps):
        """
        Main function: Analyze video frames and return multiple segments with prompts
        
        Args:
            frames (List): List of video frames (OpenCV format)
            timestamps (List): Corresponding timestamps for each frame
            
        Returns:
            List[Dict]: List of segment analysis results
        """
        print("Starting multi-segment scene analysis...")
        print(f"Input: {len(frames)} frames, {timestamps[0]:.1f}s - {timestamps[-1]:.1f}s")
        
        start_time = time.time()
        
        # Step 1: Detect scene segments based on transitions
        segments = self.detect_scene_segments(frames, timestamps)
        
        print(f"Detected {len(segments)} scene segments")
        
        # Step 2: Analyze each segment
        segment_results = []
        for i, segment in enumerate(segments):
            segment_result = self.analyze_single_segment(segment, segment_id=i)
            if segment_result:
                segment_results.append(segment_result)
        
        processing_time = time.time() - start_time
        print(f"Multi-segment analysis completed in {processing_time:.2f}s")
        print(f"Generated {len(segment_results)} audio prompts")
        
        return segment_results
    
    def detect_scene_segments(self, frames, timestamps):
        """
        Detect scene transition points and split into segments
        
        Args:
            frames (List): Video frames
            timestamps (List): Frame timestamps
            
        Returns:
            List[Dict]: List of segments with frames and time info
        """
        if len(frames) <= 1:
            return [{'frames': frames, 'timestamps': timestamps, 'start_time': timestamps[0], 'end_time': timestamps[-1]}]
        
        segments = []
        current_segment_start = 0
        
        print("Detecting scene transitions...")
        
        # Check each consecutive frame pair for scene changes
        for i in range(1, len(frames)):
            if self.is_scene_transition(frames[i-1], frames[i]):
                # Transition detected - end current segment
                segment = {
                    'frames': frames[current_segment_start:i],
                    'timestamps': timestamps[current_segment_start:i],
                    'start_time': timestamps[current_segment_start],
                    'end_time': timestamps[i-1]
                }
                segments.append(segment)
                print(f"  Transition at {timestamps[i]:.1f}s")
                current_segment_start = i
        
        # Add final segment
        final_segment = {
            'frames': frames[current_segment_start:],
            'timestamps': timestamps[current_segment_start:],
            'start_time': timestamps[current_segment_start],
            'end_time': timestamps[-1]
        }
        segments.append(final_segment)
        
        return segments
    
    def is_scene_transition(self, frame1, frame2):
        """
        Check if there's a significant scene transition between two frames
        
        Args:
            frame1, frame2: OpenCV frames
            
        Returns:
            bool: True if scene transition detected
        """
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate structural similarity
        # Simple approach: normalized cross-correlation
        gray1_norm = gray1.astype(np.float32) / 255.0
        gray2_norm = gray2.astype(np.float32) / 255.0
        
        # Resize for faster computation if frames are large
        if gray1.shape[0] > 240:
            gray1_norm = cv2.resize(gray1_norm, (320, 240))
            gray2_norm = cv2.resize(gray2_norm, (320, 240))
        
        # Calculate mean squared difference
        mse = np.mean((gray1_norm - gray2_norm) ** 2)
        
        # Scene change if MSE exceeds threshold
        is_change = mse > self.scene_change_threshold
        
        return is_change
    
    def analyze_single_segment(self, segment, segment_id=0):
        """
        Analyze a single video segment to determine environment and generate audio prompt
        
        Args:
            segment (Dict): Segment with frames, timestamps, and time range
            segment_id (int): Segment identifier
            
        Returns:
            Dict: Analysis results for this segment
        """
        frames = segment['frames']
        start_time = segment['start_time']
        end_time = segment['end_time']
        duration = end_time - start_time
        
        print(f"Analyzing segment {segment_id}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
        
        if not frames:
            print(f"  Warning: Empty segment {segment_id}")
            return None
        
        # Use multiple frames for more stable classification
        sample_frames = self.sample_frames_from_segment(frames)
        
        # Batch CLIP analysis
        environment, confidence = self.classify_environment_batch(sample_frames)
        
        # Get contextual mappings
        surface = self.get_contextual_surface(environment)
        footwear = self.get_contextual_footwear(environment)
        
        # Generate audio prompt
        audio_prompt = self.generate_audio_prompt(environment, surface, footwear)
        
        result = {
            'segment_id': segment_id,
            'time_range': (start_time, end_time),
            'duration': duration,
            'environment': environment,
            'surface': surface,
            'footwear': footwear,
            'audio_prompt': audio_prompt,
            'confidence': confidence,
            'frame_count': len(frames)
        }
        
        print(f"  Environment: {environment} ({confidence:.3f})")
        print(f"  Prompt: '{audio_prompt}'")
        
        return result
    
    def sample_frames_from_segment(self, frames, max_frames=5):
        """
        Sample representative frames from a segment for stable classification
        
        Args:
            frames (List): All frames in segment
            max_frames (int): Maximum frames to sample
            
        Returns:
            List: Sampled frames
        """
        if len(frames) <= max_frames:
            return frames
        
        # Evenly sample frames across the segment
        indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
        return [frames[i] for i in indices]
    
    def classify_environment_batch(self, frames):
        """
        Classify environment using batch of frames for stability
        
        Args:
            frames (List): Sample frames from segment
            
        Returns:
            Tuple: (environment, average_confidence)
        """
        if not frames:
            return "outdoor city street", 0.5
        
        # Preprocess all frames
        processed_images = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            processed_image = self.preprocess(pil_image)
            processed_images.append(processed_image)
        
        # Batch process
        image_batch = torch.stack(processed_images).to(self.device)
        text_tokens = clip.tokenize(self.environment_options).to(self.device)
        
        with torch.no_grad():
            logits_per_image, _ = self.model(image_batch, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Majority vote: count most frequent prediction
        predictions = np.argmax(probs, axis=1)
        prediction_counts = np.bincount(predictions, minlength=len(self.environment_options))
        final_prediction = np.argmax(prediction_counts)
        
        # Average confidence for the winning prediction
        confidences_for_winner = probs[:, final_prediction]
        average_confidence = np.mean(confidences_for_winner)
        
        environment = self.environment_options[final_prediction]
        
        return environment, average_confidence
    
    def get_contextual_surface(self, environment):
        """Get surface type based on environment"""
        return self.environment_surface_map.get(environment, "concrete")
    
    def get_contextual_footwear(self, environment):
        """Get appropriate footwear based on environment"""
        possible_shoes = self.environment_footwear_map.get(environment, ["sneakers"])
        return random.choice(possible_shoes)
    
    def get_contextual_sound(self, environment, footwear):
        """Get sound description based on environment and footwear"""
        env_sounds = self.environment_sound_map.get(environment, {})
        sound_options = env_sounds.get(footwear, [])
        
        if sound_options:
            return random.choice(sound_options)
        else:
            # Fallback sounds
            fallback_sounds = {
                'dress_shoes': 'crisp tapping',
                'boots': 'heavy thudding',
                'hiking_boots': 'rugged stomping',
                'sneakers': 'soft padding',
                'running_shoes': 'rhythmic bouncing',
                'sandals': 'light slapping',
                'slippers': 'gentle shuffling',
                'barefoot': 'soft patting'
            }
            return fallback_sounds.get(footwear, 'steady stepping')
    
    def generate_audio_prompt(self, environment, surface, footwear):
        """
        Generate final audio prompt for stable audio generation
        
        Args:
            environment (str): Detected environment
            surface (str): Surface type
            footwear (str): Footwear type
            
        Returns:
            str: Audio generation prompt
        """
        sound_verb = self.get_contextual_sound(environment, footwear)
        
        # Surface modifiers for better audio quality
        surface_modifiers = {
            'concrete': 'solid',
            'asphalt': 'firm',
            'wood': 'hollow',
            'tile': 'sharp',
            'marble': 'crisp echoing',
            'carpet': 'muffled soft',
            'rubber': 'bouncing',
            'dirt': 'crunchy',
            'gravel': 'crunching',
            'grass': 'soft',
            'sand': 'shifting',
            'stone': 'solid'
        }
        
        modifier = surface_modifiers.get(surface, 'solid')
        footwear_clean = footwear.replace('_', ' ')
        
        if modifier:
            prompt = f"{footwear_clean} with {modifier} {sound_verb} on {surface}"
        else:
            prompt = f"{footwear_clean} {sound_verb} on {surface}"
        
        return prompt
    
    def get_analyzer_info(self):
        """Get information about the analyzer configuration"""
        return {
            'device': self.device,
            'num_environments': len(self.environment_options),
            'environments': self.environment_options,
            'scene_change_threshold': self.scene_change_threshold
        }


# Test the analyzer
if __name__ == "__main__":
    import os
    
    analyzer = MultiSegmentSceneAnalyzer(seed=42)
    video_path = './data/test_videos/walk4.mp4'
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Could not open video file")
        print("Current directory:", os.getcwd())
    else:
        # Extract test frames (simulate 2-second video)
        frames = []
        timestamps = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for frame_num in range(int(2 * fps)):  # 2 seconds worth
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                timestamps.append(frame_num / fps)
            else:
                break
        
        cap.release()
        
        if frames:
            print(f"Testing with {len(frames)} frames")
            
            # Test multi-segment analysis
            results = analyzer.analyze_video_segments(frames, timestamps)
            
            print(f"\nFinal Results:")
            for result in results:
                print(f"Segment {result['segment_id']}: {result['time_range'][0]:.1f}s-{result['time_range'][1]:.1f}s")
                print(f"  Prompt: '{result['audio_prompt']}'")
            
        else:
            print("Could not read frames from video")
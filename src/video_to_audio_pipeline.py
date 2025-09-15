
import os
import time
from pathlib import Path
import cv2


from video_processor import VideoProcessor
from src.footstep_detector_heel import WalkingDetector
from scene_analyzer import FinalSceneAnalyzer
from audio_generator import AudioGenerator


class VideoToAudioPipeline:
    def __init__(self, output_dir="generated_audio", use_local_llm=False, 
                 audio_model="stabilityai/stable-audio-open-1.0"):
        """
        Initialize the complete pipeline
        
        Args:
            output_dir (str): Directory to save generated audio files
            use_local_llm (bool): Whether to use local LLM in scene analyzer
            audio_model (str): Audio model identifier for generation
        """
        print("🚀 Initializing Professional Video-to-Audio Pipeline...")
        

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("📋 Loading pipeline components...")
        self.video_processor = VideoProcessor()
        self.walking_detector = WalkingDetector()
        self.scene_analyzer = FinalSceneAnalyzer(use_local_llm=use_local_llm)
        self.audio_generator = AudioGenerator(model_name=audio_model)
        
        print("✅ Pipeline initialized successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print()
    
    def process_video_segment(self, frames, timestamps, segment_id=0, 
                            audio_duration=3.0, audio_steps=50):
        """
        Process a single video segment to generate contextual audio
        
        Args:
            frames (list): List of video frames for this segment
            timestamps (list): Corresponding timestamps for each frame
            segment_id (int): Unique identifier for this segment
            audio_duration (float): Length of audio to generate
            audio_steps (int): Quality steps for audio generation
            
        Returns:
            dict: Processing results or None if failed
        """
        print(f"\n{'='*60}")
        print(f"🎬 Processing segment {segment_id}")
        print(f"📍 Frames: {len(frames)} | Timespan: {timestamps[0]:.1f}s - {timestamps[-1]:.1f}s")
        
        # Step 1: Detect people walking
        print("\n👥 Detecting people...")
        detections = self.walking_detector.analyze_walking_video(frames, timestamps)
        
        # Check if people are detected
        frames_with_people = sum(1 for d in detections if d['people_count'] > 0)
        if frames_with_people == 0:
            print("⚠️ No people detected in this segment, skipping audio generation")
            return None
        
        print(f"✅ People detected in {frames_with_people}/{len(frames)} frames")
        
        # Step 2: Analyze scene (use middle frame for best representation)
        middle_frame_idx = len(frames) // 2
        middle_frame = frames[middle_frame_idx]
        middle_timestamp = timestamps[middle_frame_idx]
        
        print(f"\n🔍 Analyzing scene at {middle_timestamp:.1f}s...")
        scene_analysis = self.scene_analyzer.analyze_scene(middle_frame)
        
        # Step 3: Generate audio using dedicated audio generator
        print(f"\n🎵 Generating footstep audio...")
        audio_prompt = scene_analysis['audio_prompt']
        
        # Generate filename
        base_name = f"segment_{segment_id:03d}_{middle_timestamp:.1f}s_{scene_analysis['footwear']}_{scene_analysis['surface']}"
        clean_name = base_name.replace(' ', '_').replace(',', '').replace('/', '_')
        audio_filename = f"{clean_name}.wav"
        audio_filepath = self.output_dir / audio_filename
        
        # Use the professional audio generator
        generation_result = self.audio_generator.generate_and_save(
            prompt=audio_prompt,
            output_path=audio_filepath,
            duration=audio_duration,
            steps=audio_steps
        )
        
        if not generation_result.get('success', False):
            print("❌ Audio generation failed for this segment")
            return None
        
        # Compile comprehensive results
        return {
            'segment_id': segment_id,
            'timestamp_range': (timestamps[0], timestamps[-1]),
            'middle_timestamp': middle_timestamp,
            'people_detected': frames_with_people,
            'total_frames': len(frames),
            'scene_analysis': scene_analysis,
            'audio_prompt': audio_prompt,
            'audio_file': generation_result['file_path'],
            'audio_metadata': generation_result
        }
    
    def process_full_video(self, video_path, segment_duration=5.0, audio_duration=3.0, 
                          extraction_fps=10, audio_steps=50):
        """
        Process entire video and generate audio for all walking segments
        
        Args:
            video_path (str): Path to input video file
            segment_duration (float): Duration of each video segment in seconds
            audio_duration (float): Duration of generated audio per segment
            extraction_fps (int): FPS for frame extraction from video
            audio_steps (int): Quality steps for audio generation
            
        Returns:
            dict: Complete processing results
        """
        print("🎬 Starting Professional Video Processing...")
        print(f"📹 Video: {video_path}")
        print(f"⏱️ Segment duration: {segment_duration}s")
        print(f"🎵 Audio duration: {audio_duration}s per segment")
        print(f"🖼️ Extraction FPS: {extraction_fps}")
        print(f"🔧 Audio quality steps: {audio_steps}")
        
        start_time = time.time()
        
        # Step 1: Video validation and frame extraction
        print(f"\n{'='*60}")
        print("📋 Video validation and frame extraction...")
        
        try:
            video_info = self.video_processor.validate_video(video_path)
            frames, timestamps = self.video_processor.extract_frames(video_path, extraction_fps)
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            return {'success': False, 'error': str(e)}
        
        # Step 2: Create video segments
        segment_frame_count = int(segment_duration * extraction_fps)
        segments = []
        
        for i in range(0, len(frames), segment_frame_count):
            segment_frames = frames[i:i + segment_frame_count]
            segment_timestamps = timestamps[i:i + segment_frame_count]
            
            # Only process segments with sufficient frames
            if len(segment_frames) >= segment_frame_count // 2:
                segments.append((segment_frames, segment_timestamps))
        
        print(f"📊 Created {len(segments)} segments of ~{segment_duration}s each")
        
        # Step 3: Process each segment through the complete pipeline
        results = []
        successful_generations = 0
        
        for segment_id, (segment_frames, segment_timestamps) in enumerate(segments):
            result = self.process_video_segment(
                frames=segment_frames, 
                timestamps=segment_timestamps, 
                segment_id=segment_id,
                audio_duration=audio_duration,
                audio_steps=audio_steps
            )
            
            if result is not None:
                results.append(result)
                successful_generations += 1
            
            print()  # Add spacing between segments
        
        # Step 4: Generate comprehensive summary
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("🎉 PROFESSIONAL VIDEO PROCESSING COMPLETE!")
        print(f"⏱️ Total processing time: {total_time:.1f}s")
        print(f"📊 Successfully processed: {successful_generations}/{len(segments)} segments")
        print(f"📁 Audio files saved in: {self.output_dir.absolute()}")
        
        if successful_generations > 0:
            print(f"\n🎧 Generated Audio Files:")
            for result in results:
                filename = Path(result['audio_file']).name
                print(f"   • {filename}")
                print(f"     Time: {result['middle_timestamp']:.1f}s | Environment: {result['scene_analysis']['environment']}")
                print(f"     Prompt: '{result['audio_prompt']}'")
        else:
            print("❌ No audio files generated. Check video content and pipeline settings.")
        
        return {
            'success': True,
            'video_info': video_info,
            'processing_time': total_time,
            'total_segments': len(segments),
            'successful_generations': successful_generations,
            'results': results,
            'output_directory': str(self.output_dir.absolute()),
            'pipeline_settings': {
                'segment_duration': segment_duration,
                'audio_duration': audio_duration,
                'extraction_fps': extraction_fps,
                'audio_steps': audio_steps
            }
        }
    
    def get_pipeline_info(self):
        """Get information about all pipeline components"""
        return {
            'audio_generator': self.audio_generator.get_model_info(),
            'output_directory': str(self.output_dir.absolute()),
            'components': [
                'VideoProcessor',
                'WalkingDetector', 
                'FinalSceneAnalyzer',
                'AudioGenerator'
            ]
        }


# Professional main execution
if __name__ == "__main__":
    # Configuration - easily adjustable for different use cases
    CONFIG = {
        'video_path': "../data/test_videos/walk3.mp4",  # Update this path
        'output_dir': "professional_footsteps",
        'segment_duration': 5.0,      # Seconds per video segment
        'audio_duration': 3.0,        # Seconds of audio per segment
        'extraction_fps': 10,         # Frame extraction rate
        'audio_steps': 50,           # Audio quality (25=fast, 50=good, 100=high)
        'use_local_llm': False,      # Enable if you have Ollama setup
        'audio_model': "stabilityai/stable-audio-open-1.0"
    }
    
    try:
        # Initialize professional pipeline
        pipeline = VideoToAudioPipeline(
            output_dir=CONFIG['output_dir'],
            use_local_llm=CONFIG['use_local_llm'],
            audio_model=CONFIG['audio_model']
        )
        
        # Process video with professional pipeline
        results = pipeline.process_full_video(
            video_path=CONFIG['video_path'],
            segment_duration=CONFIG['segment_duration'],
            audio_duration=CONFIG['audio_duration'],
            extraction_fps=CONFIG['extraction_fps'],
            audio_steps=CONFIG['audio_steps']
        )
        
        if results['success']:
            print(f"\n🎯 Professional pipeline completed successfully!")
            print(f"📊 Generated {results['successful_generations']} audio files")
            print(f"📁 Check: {results['output_directory']}")
        else:
            print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        
    except FileNotFoundError:
        print(f"❌ Video file not found: {CONFIG['video_path']}")
        print("Please update the video_path in CONFIG with the correct path")
        
    except Exception as e:
        print(f"❌ Professional pipeline failed: {e}")
        import traceback
        traceback.print_exc()
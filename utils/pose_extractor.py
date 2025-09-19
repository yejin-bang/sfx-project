import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path

# Add src directory to path to import VideoValidator
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from video_validator import VideoValidator


class PoseExtractor:
    """
    Extract pose landmarks from video at 10fps for footstep detection
    
    Extracts specific landmarks needed for gait analysis:
    - Hip: LEFT_HIP(23), RIGHT_HIP(24) 
    - Heel: LEFT_HEEL(29), RIGHT_HEEL(30)
    - Ankle: LEFT_ANKLE(27), RIGHT_ANKLE(28)
    - Foot Index: LEFT_FOOT_INDEX(31), RIGHT_FOOT_INDEX(32)
    """
    
    def __init__(self, target_fps: int = 10, confidence_threshold: float = 0.7):
        """
        Initialize pose extractor
        
        Args:
            target_fps: Process video at this fps (default: 10)
            confidence_threshold: Minimum confidence for valid landmarks (default: 0.7)
        """
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold
        
        # Initialize VideoValidator
        self.video_validator = VideoValidator()
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between accuracy and speed
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices we need for gait analysis
        self.landmark_indices = {
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29,
            'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31,
            'RIGHT_FOOT_INDEX': 32
        }
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Extract pose landmarks from a single frame
        
        Returns:
            Dictionary with landmark coordinates and confidence, or None if pose not detected
            Format: {'LEFT_HIP': (x, y, confidence), ...}
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = {}
        
        for name, idx in self.landmark_indices.items():
            landmark = results.pose_landmarks.landmark[idx]
            
            # Check confidence threshold
            if landmark.visibility < self.confidence_threshold:
                landmarks[name] = (None, None, landmark.visibility)
            else:
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * frame.shape[1]  # width
                y = landmark.y * frame.shape[0]  # height
                landmarks[name] = (x, y, landmark.visibility)
        
        return landmarks
    
    def process_video(self, video_path: str, verbose: bool = True) -> Dict[str, any]:
        """
        Process entire video and extract pose landmarks at target fps
        
        Returns:
            Dictionary containing:
            - landmarks_data: List of landmark dictionaries for each processed frame (signal_processor compatible)
            - timestamps: List of timestamps for each frame (signal_processor compatible)
            - video_info: Video metadata from VideoValidator
            - processing_stats: Processing statistics
        """
        # Use VideoValidator to validate and get video info
        video_info = self.video_validator.validate_video(video_path)
        
        # Calculate frame skip interval for target fps
        original_fps = video_info['fps']
        frame_skip_interval = max(1, int(original_fps / self.target_fps))
        
        if verbose:
            print(f"Processing video: {video_info['filename']}")
            print(f"Original FPS: {original_fps:.2f}")
            print(f"Target FPS: {self.target_fps}")
            print(f"Frame skip interval: {frame_skip_interval}")
            print(f"Total frames: {video_info['frame_count']}")
        
        # Get video reader from validator
        cap = self.video_validator.get_video_reader(video_path)
        
        landmarks_list = []
        timestamps = []
        frame_count = 0
        processed_count = 0
        frames_with_pose = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to achieve target fps
                if frame_count % frame_skip_interval == 0:
                    # Extract landmarks
                    landmarks = self.extract_landmarks_from_frame(frame)
                    
                    # Calculate timestamp
                    timestamp = frame_count / original_fps
                    
                    landmarks_list.append(landmarks)
                    timestamps.append(timestamp)
                    processed_count += 1
                    
                    if landmarks is not None:
                        frames_with_pose += 1
                
                frame_count += 1
        
        finally:
            cap.release()
        
        # Calculate processing statistics
        pose_detection_rate = (frames_with_pose / processed_count * 100) if processed_count > 0 else 0
        actual_fps = processed_count / video_info['duration'] if video_info['duration'] > 0 else 0
        
        processing_stats = {
            'total_frames_read': frame_count,
            'frames_processed': processed_count,
            'frames_with_pose': frames_with_pose,
            'pose_detection_rate': pose_detection_rate,
            'actual_processing_fps': actual_fps
        }
        
        if verbose:
            print(f"\nProcessing complete!")
            print(f"Frames processed: {processed_count}")
            print(f"Pose detection rate: {pose_detection_rate:.1f}%")
            print(f"Actual processing FPS: {actual_fps:.2f}")
        
        return {
            'landmarks_data': landmarks_list,  # ✅ signal_processor 호환 형태
            'timestamps': timestamps,          # ✅ signal_processor 호환 형태
            'video_info': video_info,
            'processing_stats': processing_stats
        }
    
    def get_signal_processor_input(self, video_path: str, verbose: bool = True) -> Tuple[List[Dict], List[float]]:
        """
        Process video and return data in signal_processor.py compatible format
        
        Returns:
            Tuple of (landmarks_data, timestamps) ready for signal_processor.process_coordinates()
        """
        result = self.process_video(video_path, verbose)
        return result['landmarks_data'], result['timestamps']
    
    def get_landmark_quality_report(self, landmarks_data: List[Dict]) -> Dict[str, Dict]:
        """
        Generate quality report for extracted landmarks
        
        Returns:
            Dictionary with statistics for each landmark type
        """
        quality_report = {}
        
        for landmark_name in self.landmark_indices.keys():
            valid_count = 0
            confidence_sum = 0
            total_frames = len(landmarks_data)
            
            for frame_landmarks in landmarks_data:
                if frame_landmarks is not None and landmark_name in frame_landmarks:
                    x, y, confidence = frame_landmarks[landmark_name]
                    if x is not None and y is not None:
                        valid_count += 1
                        confidence_sum += confidence
            
            avg_confidence = (confidence_sum / valid_count) if valid_count > 0 else 0
            detection_rate = (valid_count / total_frames * 100) if total_frames > 0 else 0
            
            quality_report[landmark_name] = {
                'detection_rate': detection_rate,
                'average_confidence': avg_confidence,
                'valid_frames': valid_count,
                'total_frames': total_frames
            }
        
        return quality_report
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if self.pose:
            self.pose.close()


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    extractor = PoseExtractor(target_fps=10, confidence_threshold=0.7)
    
    # Process a video (replace with your video path)
    video_path = "./test_videos/walk4.mp4" 
    
    try:
        # Method 1: Get full result (backward compatibility)
        result = extractor.process_video(video_path, verbose=True)
        
        # Method 2: Get signal_processor compatible data directly
        landmarks_data, timestamps = extractor.get_signal_processor_input(video_path, verbose=False)
        
        print(f"\n=== Signal Processor Compatible Output ===")
        print(f"landmarks_data type: {type(landmarks_data)}")
        print(f"timestamps type: {type(timestamps)}")
        print(f"Total frames: {len(landmarks_data)}")
        print(f"Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")
        
        # Show sample data structure
        if landmarks_data[0] is not None:
            print(f"\nSample frame data structure:")
            for name, coords in landmarks_data[0].items():
                x, y, conf = coords
                status = "✓" if x is not None else "✗"
                print(f"  {name}: {status} ({conf:.3f})")
        
        # Generate quality report
        quality_report = extractor.get_landmark_quality_report(landmarks_data)
        
        print("\n=== Landmark Quality Report ===")
        for landmark_name, stats in quality_report.items():
            print(f"{landmark_name}:")
            print(f"  Detection rate: {stats['detection_rate']:.1f}%")
            print(f"  Average confidence: {stats['average_confidence']:.3f}")
            print(f"  Valid frames: {stats['valid_frames']}/{stats['total_frames']}")
        
        # Test with signal_processor (if available)
        print(f"\n=== Testing with Signal Processor ===")
        try:
            from signal_processor import SignalProcessor, SignalProcessorConfig
            
            config = SignalProcessorConfig()
            processor = SignalProcessor(config)
            
            # This should work now!
            processed_result = processor.process_coordinates(landmarks_data, timestamps)
            print(f"✅ Signal processor integration successful!")
            print(f"   Sampling rate: {processed_result['metadata']['sampling_rate']:.2f} fps")
            print(f"   Data coverage: {processed_result['metadata']['data_coverage']:.1%}")
            
        except ImportError:
            print("⚠️ signal_processor.py not available, skipping integration test")
        except Exception as e:
            print(f"❌ Signal processor integration failed: {e}")
    
    except FileNotFoundError:
        print(f"Please provide a valid video file path")
        print(f"Expected path: {video_path}")
        
        # Create sample data for testing without video
        print(f"\n=== Creating Sample Data for Testing ===")
        
        # Generate mock landmarks data
        sample_landmarks = []
        sample_timestamps = []
        
        for i in range(100):  # 100 frames
            timestamp = i * 0.033  # 30fps
            
            # Simulate walking pattern
            landmarks = {
                'LEFT_HIP': (320 + i*0.5, 240 + 5*np.sin(i*0.1), 0.9),
                'RIGHT_HIP': (340 + i*0.5, 240 + 5*np.sin(i*0.1), 0.9),
                'LEFT_HEEL': (315 + i*0.5, 380 + 10*np.sin(i*0.2), 0.8),
                'RIGHT_HEEL': (345 + i*0.5, 380 + 10*np.sin(i*0.2 + np.pi), 0.8),
                'LEFT_ANKLE': (316 + i*0.5, 360 + 8*np.sin(i*0.2), 0.85),
                'RIGHT_ANKLE': (344 + i*0.5, 360 + 8*np.sin(i*0.2 + np.pi), 0.85),
                'LEFT_FOOT_INDEX': (318 + i*0.5, 385 + 12*np.sin(i*0.2), 0.75),
                'RIGHT_FOOT_INDEX': (342 + i*0.5, 385 + 12*np.sin(i*0.2 + np.pi), 0.75)
            }
            
            # Simulate some missing data
            if i % 15 == 0:  # Every 15th frame, lose some landmarks
                landmarks['LEFT_HEEL'] = (None, None, 0.3)
                landmarks['RIGHT_HEEL'] = (None, None, 0.3)
            
            sample_landmarks.append(landmarks)
            sample_timestamps.append(timestamp)
        
        print(f"Generated {len(sample_landmarks)} sample frames")
        
        # Test quality report with sample data
        quality_report = extractor.get_landmark_quality_report(sample_landmarks)
        print(f"\nSample data quality:")
        for name, stats in quality_report.items():
            print(f"  {name}: {stats['detection_rate']:.1f}% detection rate")
    
    except Exception as e:
        print(f"Error processing video: {e}")
    
    finally:
        # Clean up
        extractor.cleanup()
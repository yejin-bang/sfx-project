import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import pipeline modules
from utils.pose_extractor import PoseExtractor
from utils.signal_processor import SignalProcessor, SignalProcessorConfig
from utils.gait_detector import GaitDetector, GaitDetectorConfig
from video_validator import VideoValidator


@dataclass
class FootstepDetectorConfig:
    """Configuration for footstep detection pipeline"""
    # Pipeline configurations
    pose_extractor_fps: int = 10
    pose_confidence_threshold: float = 0.7
    signal_cutoff_frequency: float = 0.1752
    gait_peak_distance: int = 5
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.pose_extractor_fps <= 0:
            raise ValueError("pose_extractor_fps must be positive")
        if not 0 < self.pose_confidence_threshold <= 1:
            raise ValueError("pose_confidence_threshold must be between 0 and 1")


class FootstepDetector:
    """
    Pure footstep detection pipeline without ground truth comparison
    
    Pipeline:
    1. PoseExtractor: Extract landmarks from video at 10fps
    2. SignalProcessor: Clean and filter coordinate time series
    3. GaitDetector: Detect heel strikes and return frame indices
    4. FootstepDetector: Convert frame indices to timestamps
    
    Output: Pure detection results for audio generation pipeline
    """
    
    def __init__(self, config: Optional[FootstepDetectorConfig] = None):
        """
        Initialize footstep detector with pipeline components
        
        Args:
            config: FootstepDetectorConfig object. If None, uses default values.
        """
        self.config = config if config is not None else FootstepDetectorConfig()
        
        # Initialize pipeline components
        self.pose_extractor = PoseExtractor(
            target_fps=self.config.pose_extractor_fps,
            confidence_threshold=self.config.pose_confidence_threshold
        )
        
        signal_config = SignalProcessorConfig(
            cutoff_frequency=self.config.signal_cutoff_frequency,
            confidence_threshold=self.config.pose_confidence_threshold
        )
        self.signal_processor = SignalProcessor(signal_config)
        
        gait_config = GaitDetectorConfig(
            peak_distance=self.config.gait_peak_distance
        )
        self.gait_detector = GaitDetector(gait_config)
    
    def convert_frame_indices_to_timestamps(self, frame_indices: List[int], 
                                          timestamps: np.ndarray, 
                                          original_fps: float) -> List[float]:
        """
        Convert frame indices to timestamps
        
        Args:
            frame_indices: List of frame indices from gait_detector
            timestamps: Timestamp array from signal_processor
            original_fps: Original video FPS
            
        Returns:
            List of timestamps corresponding to frame indices
        """
        if len(frame_indices) == 0:
            return []
        
        # Method 1: Use processed timestamps directly (more accurate for actual processed frames)
        frame_timestamps = []
        for frame_idx in frame_indices:
            if 0 <= frame_idx < len(timestamps):
                frame_timestamps.append(float(timestamps[frame_idx]))
            else:
                # Fallback: calculate from original FPS
                timestamp = frame_idx / original_fps
                frame_timestamps.append(timestamp)
                warnings.warn(f"Frame index {frame_idx} out of bounds, using FPS calculation")
        
        return sorted(frame_timestamps)
    
    def process_video(self, video_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run complete footstep detection pipeline on video
        
        Args:
            video_path: Path to video file
            verbose: Print processing information
            
        Returns:
            Dictionary containing detection results and metadata
        """
        if verbose:
            print("=" * 60)
            print("FOOTSTEP DETECTION PIPELINE")
            print("=" * 60)
        
        # Step 1: Extract pose landmarks
        if verbose:
            print("\n🏃 Step 1: Extracting pose landmarks...")
        
        landmarks_data, timestamps = self.pose_extractor.get_signal_processor_input(
            video_path, verbose=verbose
        )
        
        # Get original video FPS for timestamp conversion
        video_info = self.pose_extractor.video_validator.validate_video(video_path)
        original_fps = video_info['fps']
        
        # Step 2: Process coordinate signals
        if verbose:
            print("\n📊 Step 2: Processing coordinate signals...")
        
        processed_data = self.signal_processor.process_coordinates(
            landmarks_data, timestamps, verbose=verbose
        )
        
        # Step 3: Detect gait events (frame indices only)
        if verbose:
            print("\n👣 Step 3: Detecting gait events...")
        
        gait_results = self.gait_detector.process_gait_events(
            processed_data, verbose=verbose
        )
        
        # Step 4: Convert frame indices to timestamps
        if verbose:
            print("\n⏰ Step 4: Converting frame indices to timestamps...")
        
        frame_indices = gait_results['heel_strike_frame_indices']
        detected_timestamps = self.convert_frame_indices_to_timestamps(
            frame_indices, processed_data['timestamps'], original_fps
        )
        
        if verbose:
            print(f"   Converted {len(frame_indices)} frame indices to timestamps")
            if len(detected_timestamps) > 0:
                print(f"   Timestamp range: {min(detected_timestamps):.3f}s to {max(detected_timestamps):.3f}s")
        
        # Compile final results
        detection_results = {
            'detected_timestamps': detected_timestamps,  # 🎯 Final timestamps for audio
            'detected_frame_indices': frame_indices,     # 🎯 Original frame indices
            'total_detections': len(detected_timestamps),
            'gait_results': gait_results,
            'processed_data': processed_data,
            'landmarks_data': landmarks_data,
            'video_info': video_info,
            'pipeline_metadata': {
                'video_path': video_path,
                'original_fps': original_fps,
                'processed_fps': processed_data['metadata']['sampling_rate'],
                'pose_extractor_config': {
                    'target_fps': self.config.pose_extractor_fps,
                    'confidence_threshold': self.config.pose_confidence_threshold
                },
                'signal_processor_config': processed_data['metadata']['config'],
                'gait_detector_config': gait_results['metadata']['processing_config']
            }
        }
        
        if verbose:
            print(f"\n✅ Pipeline complete!")
            print(f"   Detected {detection_results['total_detections']} footsteps")
            print(f"   Original video FPS: {original_fps:.2f}")
            print(f"   Processing FPS: {processed_data['metadata']['sampling_rate']:.2f}")
        
        return detection_results
    
    def get_footstep_timestamps(self, video_path: str, verbose: bool = False) -> List[float]:
        """
        Convenience method to get just the footstep timestamps
        
        Args:
            video_path: Path to video file
            verbose: Print processing information
            
        Returns:
            List of footstep timestamps in seconds
        """
        detection_results = self.process_video(video_path, verbose=verbose)
        return detection_results['detected_timestamps']
    
    def print_detection_summary(self, detection_results: Dict[str, Any]) -> None:
        """Print detection summary to console"""
        print("\n" + "=" * 60)
        print("FOOTSTEP DETECTION SUMMARY")
        print("=" * 60)
        
        timestamps = detection_results['detected_timestamps']
        video_info = detection_results['video_info']
        metadata = detection_results['pipeline_metadata']
        
        print(f"\n📹 VIDEO INFO:")
        print(f"   File: {Path(metadata['video_path']).name}")
        print(f"   Duration: {video_info['duration']:.1f}s")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   Original FPS: {metadata['original_fps']:.2f}")
        print(f"   Processing FPS: {metadata['processed_fps']:.2f}")
        
        print(f"\n👣 DETECTION RESULTS:")
        print(f"   Total footsteps detected: {detection_results['total_detections']}")
        
        if len(timestamps) > 0:
            print(f"   First footstep: {min(timestamps):.3f}s")
            print(f"   Last footstep: {max(timestamps):.3f}s")
            
            # Calculate intervals
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                avg_interval = np.mean(intervals)
                estimated_bpm = 30.0 / avg_interval if avg_interval > 0 else 0
                
                print(f"   Average interval: {avg_interval:.3f}s")
                print(f"   Estimated pace: {estimated_bpm:.1f} BPM")
        
        print(f"\n⚙️ PROCESSING CONFIG:")
        print(f"   Pose confidence threshold: {metadata['pose_extractor_config']['confidence_threshold']}")
        print(f"   Signal cutoff frequency: {metadata['signal_processor_config'].cutoff_frequency}")
        print(f"   Peak detection distance: {metadata['gait_detector_config'].peak_distance} frames")
        
        if len(timestamps) > 0:
            print(f"\n🎯 FOOTSTEP TIMESTAMPS:")
            for i, timestamp in enumerate(timestamps):
                print(f"   {i+1:2d}. {timestamp:.3f}s")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pose_extractor:
            self.pose_extractor.cleanup()


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = FootstepDetectorConfig(
        pose_extractor_fps=10,
        pose_confidence_threshold=0.7,
        signal_cutoff_frequency=0.1752
    )
    
    print("Pure Footstep Detector Configuration:")
    print(f"  Pose extractor FPS: {config.pose_extractor_fps}")
    print(f"  Confidence threshold: {config.pose_confidence_threshold}")
    print(f"  Signal cutoff frequency: {config.signal_cutoff_frequency}")
    
    # Initialize detector
    detector = FootstepDetector(config)
    
    print(f"\n✅ Pure Detection Pipeline:")
    print(f"   🎯 No ground truth dependency")
    print(f"   🎯 Pure detection results only")
    print(f"   🎯 Ready for audio generation pipeline")
    
    # File path (update to your actual file path)
    video_path = "./test_videos/walk4.mp4"
    
    try:
        # Method 1: Get full detection results
        detection_results = detector.process_video(video_path, verbose=True)
        
        # Print summary
        detector.print_detection_summary(detection_results)
        
        # Method 2: Get just timestamps (for audio pipeline)
        timestamps_only = detector.get_footstep_timestamps(video_path, verbose=False)
        
        print(f"\n🎵 FOR AUDIO PIPELINE:")
        print(f"   Timestamps: {timestamps_only}")
        print(f"   Ready to pass to audio generator!")
        
        # Show pipeline flow
        print(f"\n📄 Pipeline Flow:")
        print(f"   🏃 PoseExtractor: Extracted landmarks from video")
        print(f"   📊 SignalProcessor: Cleaned coordinate signals")
        print(f"   👣 GaitDetector: Detected {len(detection_results['detected_frame_indices'])} frame indices")
        print(f"   ⏰ FootstepDetector: Converted to {len(detection_results['detected_timestamps'])} timestamps")
        print(f"   ✅ Ready for audio generation!")
        
    except FileNotFoundError:
        print(f"File not found: {video_path}")
        print("Please update the video path in the script")
        
        # Show expected usage
        print(f"\nExpected usage:")
        print(f"1. detector = FootstepDetector()")
        print(f"2. timestamps = detector.get_footstep_timestamps('video.mp4')")
        print(f"3. audio_generator.place_footsteps(timestamps)")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Clean up
        detector.cleanup()
        print("\nFootstep detector cleanup complete!")
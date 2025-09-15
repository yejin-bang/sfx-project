import cv2
import os
from pathlib import Path

class VideoValidator:
    def __init__(self, max_duration=300):  # 5 minutes max
        self.max_duration = max_duration

    def validate_video(self, video_path):
        """
        Validate video file and extract basic information
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Video information and validation results
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened or is invalid
        """
        video_path = Path(video_path)
        
        # Check if file exists
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if video_path.suffix.lower() not in valid_extensions:
            print(f"Warning: Unusual file extension '{video_path.suffix}'. Supported: {valid_extensions}")
        
        print('=' * 60)
        print("Video Validation Starting...")
        print(f"File: {video_path.name}")
        print(f"Size: {video_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        if fps > 0:
            duration = frame_count / fps
        else:
            raise ValueError("Invalid video: FPS is 0 or unreadable")
        
        # Test if we can actually read frames
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            raise ValueError("Cannot read frames from video")
        
        cap.release()
        
        # Validation checks
        warnings = []
        
        if duration > self.max_duration:
            warnings.append(f"Long video: {duration:.1f}s (max recommended: {self.max_duration}s)")
        
        if fps < 15:
            warnings.append(f"Low FPS: {fps:.1f} (may affect detection accuracy)")
        
        if width < 480 or height < 360:
            warnings.append(f"Low resolution: {width}x{height} (may affect detection accuracy)")
        
        # Print results
        print(f"Duration: {duration:.1f}s ({frame_count:,} frames)")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.1f}")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ! {warning}")
        
        print("Video validation successful!")
        print('=' * 60)
        
        return {
            'file_path': str(video_path.absolute()),
            'filename': video_path.name,
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'file_size_mb': video_path.stat().st_size / (1024*1024),
            'warnings': warnings,
            'valid': True
        }
    
    def get_video_reader(self, video_path):
        """
        Get a video capture object for frame-by-frame reading
        
        Args:
            video_path (str): Path to validated video file
            
        Returns:
            cv2.VideoCapture: Video capture object ready for reading
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        return cap
    
    def get_frame_at_timestamp(self, video_path, timestamp):
        """
        Extract a specific frame at given timestamp
        
        Args:
            video_path (str): Path to video file
            timestamp (float): Time in seconds
            
        Returns:
            numpy.ndarray: Frame at specified timestamp, or None if failed
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None


if __name__ == "__main__":
    # Test the validator
    validator = VideoValidator()
    
    # Update this path to your actual test video
    test_video = "/Users/yejinbang/Documents/GitHub/sfx-project/data/test_videos/walk3.mp4"
    
    try:
        # Validate video
        video_info = validator.validate_video(test_video)
        print(f"\nVideo Info Summary:")
        print(f"  Duration: {video_info['duration']:.1f}s")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"  FPS: {video_info['fps']:.1f}")
        
        # Test frame extraction
        print(f"\nTesting frame extraction...")
        test_frame = validator.get_frame_at_timestamp(test_video, 5.0)
        if test_frame is not None:
            print(f"Successfully extracted frame at 5.0s: {test_frame.shape}")
        else:
            print("Failed to extract test frame")
        
        print(f"\nVideo validator test completed successfully!")
        
    except Exception as e:
        print(f"Validation failed: {e}")
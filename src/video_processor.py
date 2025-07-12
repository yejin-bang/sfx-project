import cv2
import os

class VideoProcessor:
    def __init__(self, max_duration=120):
        self.max_duration = max_duration

    def validate_video(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print('='*70)
        print("Starting validation...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        print()
        print(f"Video info: {duration:.1f}s, {width}x{height}, {fps:.1f} FPS")
        if duration > self.max_duration:
            print(f"Warning: Video is {duration:.1f}s, longer than max {self.max_duration}s")
        
        return {
            'duration': duration,
            'fps': fps,
            'width': width,
            'heigh': height,
            'frame_count': frame_count
        }
    
    def extract_frames(self, video_path, fps_target=10):
        
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        skip_frames = max(1, int(original_fps/fps_target))

        frames = []
        frame_timestamps = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                frames.append(frame)
                timestamp = frame_count / original_fps
                frame_timestamps.append(timestamp)
            
            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames at ~{fps_target} FPS")

        return frames, frame_timestamps

if __name__ == "__main__":
    processor = VideoProcessor()

    video_path = '../data/test_videos/walk1.mp4'

    try:
        info = processor.validate_video(video_path)
        frames, timestamps = processor.extract_frames(video_path)
        print(f'Video processing test successful!')

    except Exception as e:
        print(f"Error: {e}")
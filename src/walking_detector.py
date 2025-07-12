from ultralytics import YOLO
import cv2

class WalkingDetector:
    def __init__(self):
        print('='*70)
        print("Loading YOLO model...")
        self.yolo = YOLO('yolov8n.pt')
        print("YOLO loaded")
        print()

    def detect_people(self, frame):
        results = self.yolo(frame)

        people = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.yolo.names[class_id]

            if class_id == 0 and confidence > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                people.append({
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center_x': int((x1 + x2) /2),
                    'center_y': int((y1 + y2) /2)
                })
        return people
    
    def analyze_walking_video(self, frames, timestamps):
        print(f"Analysing {len(frames)} frames for walking people...")

        frame_detections = []

        for i, frame in enumerate(frames):
            people = self.detect_people(frame)
            frame_detections.append({
                'timestamp': timestamps[i],
                'people_count': len(people),
                'people': people
            })

        total_people_detected = sum(len(detection['people']) for detection in frame_detections)
        frames_with_people = sum(1 for detection in frame_detections if detection['people_count'] > 0)

        print(f"\n=== Detection Results ===")
        print(f"Frames with people: {frames_with_people}/{len(frames)}")
        print(f"Total people detections: {total_people_detected}")
        print(f"Average people per frame: {total_people_detected/len(frames):.1f}")

        return frame_detections
    
if __name__ == "__main__":
    from video_processor import VideoProcessor

    processor = VideoProcessor()
    video_path = '../data/test_videos/walk1.mp4'

    print('=== Processing Video ===')
    frames, timestamps = processor.extract_frames(video_path)

    print('\n=== Detecting People ===')
    detector = WalkingDetector()
    detections = detector.analyze_walking_video(frames, timestamps)

    print('\nWalking detection test complete!')
    print()
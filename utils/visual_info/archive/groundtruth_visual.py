import cv2
import json
import numpy as np
from pathlib import Path

class GroundTruthVideoChecker:
    def __init__(self, video_path, json_path):
        self.video_path = video_path
        self.json_path = json_path
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load ground truth
        with open(json_path, 'r') as f:
            self.gt_data = json.load(f)
        
        self.annotations = self.gt_data['annotations']
        self.current_frame = 0
        
        print(f"Video: {video_path}")
        print(f"FPS: {self.fps}, Total frames: {self.total_frames}")
        print(f"Ground truth steps: {len(self.annotations)}")
        print()
        print("CONTROLS:")
        print("  A/D      - Previous/Next frame")
        print("  1-8      - Jump to ground truth step 1-8")
        print("  G        - Go to specific frame (type frame number)")
        print("  SPACE    - Play/Pause")
        print("  S        - Show step list")
        print("  Q/ESC    - Quit")
        print()
        
        self.show_step_list()
    
    def show_step_list(self):
        """Display all ground truth steps with numbers"""
        print("GROUND TRUTH STEPS:")
        for i, step in enumerate(self.annotations, 1):
            frame = step['frame']
            timestamp = step['timestamp']
            foot = step['foot']
            print(f"  {i}: Frame {frame:3d} | {timestamp:6.3f}s | {foot.upper()} foot")
        print()
    
    def jump_to_step(self, step_number):
        """Jump to a specific ground truth step"""
        if 1 <= step_number <= len(self.annotations):
            step = self.annotations[step_number - 1]
            target_frame = max(0, step['frame'] - 10)  # Go 10 frames before the step
            self.current_frame = target_frame
            print(f"Jumped to step {step_number}: Frame {step['frame']} ({step['timestamp']:.3f}s) - {step['foot']} foot")
            print(f"Currently at frame {self.current_frame} (10 frames before step)")
        else:
            print(f"Invalid step number. Use 1-{len(self.annotations)}")
    
    def go_to_frame(self):
        """Prompt user to input a specific frame number"""
        print(f"\nCurrent frame: {self.current_frame}")
        print(f"Total frames: {self.total_frames - 1} (0-indexed)")
        
        try:
            frame_input = input(f"Enter frame number (0-{self.total_frames - 1}): ").strip()
            
            if not frame_input:
                print("No input provided, staying at current frame.")
                return
            
            target_frame = int(frame_input)
            
            if 0 <= target_frame < self.total_frames:
                self.current_frame = target_frame
                timestamp = self.current_frame / self.fps
                print(f"Jumped to frame {self.current_frame} ({timestamp:.3f}s)")
                
                # Check if there are any nearby annotations
                nearby = self.get_nearby_annotations(self.current_frame, window=15)
                if nearby:
                    print("Nearby ground truth steps:")
                    for item in nearby:
                        ann = item['annotation']
                        distance = item['distance']
                        if distance == 0:
                            print(f"  >>> EXACT MATCH: Step {item['index']} - {ann['foot']} foot <<<")
                        else:
                            print(f"  Step {item['index']}: {ann['foot']} foot ({distance:+d} frames)")
            else:
                print(f"Invalid frame number. Must be between 0 and {self.total_frames - 1}")
                
        except ValueError:
            print("Invalid input. Please enter a valid frame number.")
        except KeyboardInterrupt:
            print("\nInput cancelled.")
    
    def get_nearby_annotations(self, frame_number, window=30):
        """Get annotations within window frames of current frame"""
        nearby = []
        for i, ann in enumerate(self.annotations):
            if abs(ann['frame'] - frame_number) <= window:
                distance = ann['frame'] - frame_number
                nearby.append({
                    'index': i + 1,
                    'annotation': ann,
                    'distance': distance
                })
        return nearby
    
    def draw_frame_info(self, frame):
        """Draw frame information and nearby annotations"""
        h, w = frame.shape[:2]
        timestamp = self.current_frame / self.fps
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Current frame info
        info_text = f"Frame: {self.current_frame:4d} | Time: {timestamp:7.3f}s"
        cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Check for nearby annotations
        nearby = self.get_nearby_annotations(self.current_frame, window=15)
        
        if nearby:
            y_pos = 70
            cv2.putText(overlay, "NEARBY STEPS:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            for item in nearby:
                y_pos += 35
                ann = item['annotation']
                distance = item['distance']
                
                if distance == 0:
                    # Exact match - highlight in red
                    text = f">>> STEP {item['index']}: {ann['foot'].upper()} FOOT <<<"
                    color = (0, 0, 255)
                    thickness = 3
                elif abs(distance) <= 5:
                    # Very close - highlight in yellow
                    text = f"Step {item['index']}: {ann['foot']} foot ({distance:+d} frames)"
                    color = (0, 255, 255)
                    thickness = 2
                else:
                    # Nearby - normal text
                    text = f"Step {item['index']}: {ann['foot']} foot ({distance:+d} frames)"
                    color = (255, 255, 255)
                    thickness = 1
                
                cv2.putText(overlay, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
        
        # Controls reminder
        controls = "A/D=Frame | 1-8=Jump | G=Go to frame | SPACE=Play | S=Steps | Q=Quit"
        cv2.putText(overlay, controls, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run(self):
        """Main checking loop"""
        paused = True
        
        try:
            while True:
                # Set frame position
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                
                if not ret:
                    print("End of video")
                    break
                
                # Draw information overlay
                display_frame = self.draw_frame_info(frame)
                
                # Show frame
                cv2.imshow('Ground Truth Checker', display_frame)
                
                # Handle input
                key = cv2.waitKey(30 if not paused else 0) & 0xFF
                
                if key == ord(' '):  # Space - play/pause
                    paused = not paused
                    print("PAUSED" if paused else "PLAYING")
                
                elif key == ord('a') and paused:  # A - previous frame
                    self.current_frame = max(0, self.current_frame - 1)
                
                elif key == ord('d') and paused:  # D - next frame
                    self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                
                elif key == ord('g'):  # G - go to specific frame
                    paused = True  # Pause video when getting input
                    print(f"\n{'='*50}")
                    print("GO TO SPECIFIC FRAME")
                    print(f"{'='*50}")
                    self.go_to_frame()
                    print(f"{'='*50}")
                
                elif key == ord('s'):  # S - show step list
                    self.show_step_list()
                
                elif ord('1') <= key <= ord('8'):  # Number keys - jump to step
                    step_num = key - ord('0')
                    self.jump_to_step(step_num)
                    paused = True
                
                elif key == ord('q') or key == 27:  # Q or ESC - quit
                    break
                
                # Auto-advance if playing
                if not paused:
                    self.current_frame += 1
                    if self.current_frame >= self.total_frames:
                        self.current_frame = self.total_frames - 1
                        paused = True
                        print("End of video - PAUSED")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def check_ground_truth(video_path, json_path):
    """Main function to check ground truth against video"""
    try:
        checker = GroundTruthVideoChecker(video_path, json_path)
        checker.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Get file paths
    video_file = "./data/test_videos/walk4.mp4"
    json_file = "./data/test_videos/walk4_ground_truth.json"
    
    if video_file and json_file:
        check_ground_truth(video_file, json_file)
    else:
        print("Please provide both video and JSON file paths.")
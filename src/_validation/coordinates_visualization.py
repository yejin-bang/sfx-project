import cv2
import numpy as np
import mediapipe as mp
import json
from collections import deque

class VideoCoordinateOverlay:
    def __init__(self, video_path, ground_truth_path=None):
        self.video_path = video_path
        self.ground_truth_path = ground_truth_path
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
        
        # Local minimum detection parameters
        self.window_size = 5
        self.min_confidence = 0.5
        
        # Pre-calculated data
        self.all_heel_data = {}  # Store heel positions for all frames
        self.detected_minimums = {}  # Store detected minimums by frame number
        
        # Load ground truth if provided
        self.ground_truth = None
        if ground_truth_path:
            with open(ground_truth_path, 'r') as f:
                gt_data = json.load(f)
                self.ground_truth = gt_data['annotations']
                print(f"Loaded {len(self.ground_truth)} ground truth annotations")
        
        # Pre-process the entire video to detect minimums
        print("Pre-processing video to detect all minimums...")
        self.preprocess_video()
        
        print("COLORS:")
        print("  RED dot = Left heel")
        print("  BLUE dot = Right heel") 
        print("  YELLOW circle = Detected local minimum")
        print("  GREEN circle = Ground truth annotation")
        print()
        print("CONTROLS:")
        print("  SPACE = Play/Pause")
        print("  A/D = Previous/Next frame (when paused)")
        print("  Q/ESC = Quit")
        print()
    
    def detect_pose(self, frame):
        """Extract pose landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks if results.pose_landmarks else None
    
    def extract_heel_positions(self, landmarks):
        """Extract heel positions and confidence"""
        if not landmarks:
            return None
        
        left_heel = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        
        heel_data = {
            'left': {
                'x': left_heel.x,
                'y': left_heel.y,
                'confidence': left_heel.visibility,
                'usable': left_heel.visibility >= self.min_confidence
            },
            'right': {
                'x': right_heel.x,
                'y': right_heel.y,
                'confidence': right_heel.visibility,
                'usable': right_heel.visibility >= self.min_confidence
            }
        }
        
        return heel_data
    
    def detect_local_minimum_simple(self, y_values):
        """Same local minimum detection as your detector"""
        if len(y_values) < self.window_size:
            return False
        
        recent_y = y_values[-self.window_size:]
        min_idx = np.argmin(recent_y)
        
        # Must be in middle of window
        if min_idx <= 1 or min_idx >= len(recent_y) - 2:
            return False
        
        # Check for valley pattern
        before_min = recent_y[:min_idx]
        after_min = recent_y[min_idx+1:]
        min_value = recent_y[min_idx]
        
        # Simple checks
        going_down = len(before_min) >= 2 and before_min[-1] < before_min[-2]
        going_up = len(after_min) >= 1 and after_min[0] > min_value
        
        # Check for significant motion
        if len(before_min) > 0 and len(after_min) > 0:
            height_drop = before_min[0] - min_value
            height_rise = after_min[-1] - min_value
            significant_motion = height_drop > 0.01 and height_rise > 0.01
        else:
            significant_motion = False
        
        return going_down and going_up and significant_motion
    
    def preprocess_video(self):
        """Pre-process entire video to detect all heel positions and minimums"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tracking data for minimum detection
        left_y_history = deque(maxlen=self.window_size * 2)
        right_y_history = deque(maxlen=self.window_size * 2)
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / fps
            landmarks = self.detect_pose(frame)
            heel_data = self.extract_heel_positions(landmarks) if landmarks else None
            
            # Store heel data for this frame
            self.all_heel_data[frame_number] = heel_data
            
            # Detect minimums
            if heel_data:
                for foot_name in ['left', 'right']:
                    heel = heel_data[foot_name]
                    if not heel['usable']:
                        continue
                    
                    # Add to history
                    history = left_y_history if foot_name == 'left' else right_y_history
                    history.append(heel['y'])
                    
                    # Check for local minimum
                    if len(history) >= self.window_size:
                        if self.detect_local_minimum_simple(list(history)):
                            # Store the detection for this frame
                            self.detected_minimums[frame_number] = {
                                'foot': foot_name,
                                'timestamp': timestamp,
                                'frame': frame_number,
                                'y_value': heel['y']
                            }
                            print(f"Pre-detected minimum: {foot_name.upper()} at frame {frame_number} ({timestamp:.3f}s)")
            
            frame_number += 1
            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")
        
        cap.release()
        print(f"Pre-processing complete. Found {len(self.detected_minimums)} minimums.")
    
    def get_exact_ground_truth(self, frame_number):
        """Get ground truth annotations for exact frame number"""
        if not self.ground_truth:
            return []
        
        exact_matches = []
        for gt in self.ground_truth:
            if gt['frame'] == frame_number:
                exact_matches.append(gt)
        return exact_matches
    
    def draw_overlays(self, frame, frame_number, timestamp):
        """Draw heel positions and annotations on frame using pre-calculated data"""
        heel_data = self.all_heel_data.get(frame_number)
        if not heel_data:
            return frame
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw heel positions
        for foot_name in ['left', 'right']:
            heel = heel_data[foot_name]
            if heel['usable']:
                # Convert normalized coordinates to pixel coordinates
                x = int(heel['x'] * w)
                y = int(heel['y'] * h)
                
                # Choose color: RED for left, BLUE for right
                color = (0, 0, 255) if foot_name == 'left' else (255, 0, 0)
                
                # Draw heel position
                cv2.circle(overlay, (x, y), 8, color, -1)
                
                # Draw confidence text
                conf_text = f"{heel['confidence']:.2f}"
                cv2.putText(overlay, conf_text, (x + 12, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw detected minimum as YELLOW circle (if this frame has one)
        if frame_number in self.detected_minimums:
            detection = self.detected_minimums[frame_number]
            foot_name = detection['foot']
            
            if heel_data[foot_name]['usable']:
                heel = heel_data[foot_name]
                x = int(heel['x'] * w)
                y = int(heel['y'] * h)
                
                # Draw YELLOW circle for detected minimum
                cv2.circle(overlay, (x, y), 20, (0, 255, 255), 3)
                cv2.putText(overlay, "MIN", (x - 15, y - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw ground truth annotations as GREEN circles (only on exact frame)
        exact_gt = self.get_exact_ground_truth(frame_number)
        for gt in exact_gt:
            # We don't have pixel coordinates for ground truth, so estimate from current heel positions
            if heel_data and heel_data[gt['foot']]['usable']:
                heel = heel_data[gt['foot']]
                x = int(heel['x'] * w)
                y = int(heel['y'] * h)
                
                # Draw GREEN circle for ground truth
                cv2.circle(overlay, (x, y), 25, (0, 255, 0), 3)
                cv2.putText(overlay, "GT", (x - 10, y + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw frame information
        info_text = f"Frame: {frame_number:4d} | Time: {timestamp:6.3f}s"
        cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw detection info if this frame has a minimum
        if frame_number in self.detected_minimums:
            detection = self.detected_minimums[frame_number]
            det_text = f"DETECTED: {detection['foot'].upper()} minimum"
            cv2.putText(overlay, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw legend
        legend_y = 90
        cv2.circle(overlay, (30, legend_y), 6, (0, 0, 255), -1)  # Red
        cv2.putText(overlay, "Left Heel", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 25
        cv2.circle(overlay, (30, legend_y), 6, (255, 0, 0), -1)  # Blue
        cv2.putText(overlay, "Right Heel", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 25
        cv2.circle(overlay, (30, legend_y), 8, (0, 255, 255), 2)  # Yellow
        cv2.putText(overlay, "Detected Min", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 25
        cv2.circle(overlay, (30, legend_y), 8, (0, 255, 0), 2)  # Green
        cv2.putText(overlay, "Ground Truth", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw controls
        controls = "SPACE=Play/Pause | A/D=Frame step | Q=Quit"
        cv2.putText(overlay, controls, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run(self):
        """Main visualization loop"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        current_frame = 0
        paused = True
        
        print(f"Video loaded: {self.video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        print("Starting visualization...")
        print()
        
        try:
            while True:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video")
                    break
                
                timestamp = current_frame / fps
                
                # Get frame from video
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video")
                    break
                
                # Draw overlays using pre-calculated data
                display_frame = self.draw_overlays(frame, current_frame, timestamp)
                
                # Show frame
                cv2.imshow('Heel Coordinate Visualization', display_frame)
                
                # Handle input
                key = cv2.waitKey(30 if not paused else 0) & 0xFF
                
                if key == ord(' '):  # Space - play/pause
                    paused = not paused
                    print("PAUSED" if paused else "PLAYING")
                
                elif key == ord('a') and paused:  # A - previous frame
                    current_frame = max(0, current_frame - 1)
                
                elif key == ord('d') and paused:  # D - next frame
                    current_frame = min(total_frames - 1, current_frame + 1)
                
                elif key == ord('q') or key == 27:  # Q or ESC - quit
                    break
                
                # Auto-advance if playing
                if not paused:
                    current_frame += 1
                    if current_frame >= total_frames:
                        current_frame = total_frames - 1
                        paused = True
                        print("End of video - PAUSED")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print summary
            print(f"\nSUMMARY:")
            print(f"Total local minimums detected: {len(self.detected_minimums)}")
            if self.ground_truth:
                print(f"Ground truth annotations: {len(self.ground_truth)}")
            
            print(f"\nDetected minimums:")
            for frame_num, detection in self.detected_minimums.items():
                print(f"  Frame {frame_num:3d} ({detection['timestamp']:6.3f}s): {detection['foot'].upper()} foot")

def visualize_video_coordinates(video_path, ground_truth_path=None):
    """Main function to visualize coordinates on video"""
    try:
        visualizer = VideoCoordinateOverlay(video_path, ground_truth_path)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Pre-set file paths for your specific files
    video_file = "./data/test_videos/walk5.mp4"
    json_file = "./data/test_videos/walk5_ground_truth.json"
    
    print(f"Using video: {video_file}")
    print(f"Using ground truth: {json_file}")
    print()
    
    visualize_video_coordinates(video_file, json_file)
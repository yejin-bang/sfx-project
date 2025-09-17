import cv2
import numpy as np
import mediapipe as mp
import json
import csv
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DetectedStep:
    timestamp: float
    foot: str
    position_x: float
    position_y: float
    confidence: float
    frame_number: int

class PureHeelDetector:
    def __init__(self, min_confidence=0.5, window_size=7, min_step_interval=0.3):
        """
        Pure heel detection using local minimum approach
        
        Args:
            min_confidence: Minimum MediaPipe confidence to use heel landmark
            window_size: Number of frames to look for local minimum
            min_step_interval: Minimum seconds between steps (same foot)
        """
        print("Initializing Pure Heel Detector...")
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
        
        # Detection parameters
        self.min_confidence = min_confidence
        self.window_size = window_size
        self.min_step_interval = min_step_interval
        
        # Tracking data
        self.left_heel_history = deque(maxlen=window_size * 2)
        self.right_heel_history = deque(maxlen=window_size * 2)
        self.last_step_times = {'left': -999, 'right': -999}
        
        # Frame-by-frame data for CSV export
        self.frame_data = []
        
        print(f"Configuration:")
        print(f"  Min confidence: {min_confidence}")
        print(f"  Window size: {window_size} frames")
        print(f"  Min step interval: {min_step_interval}s")
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
    
    def detect_local_minimum(self, y_history, current_frame):
        """
        Detect if there's a local minimum in recent Y positions
        
        A local minimum means:
        - Y was decreasing (foot going down)
        - Y reached a bottom point
        - Y started increasing (foot going up)
        """
        if len(y_history) < self.window_size:
            return False
        
        # Get recent positions
        recent_y = list(y_history)[-self.window_size:]
        
        # Find the minimum point in the window
        min_idx = np.argmin(recent_y)
        
        # Local minimum must be in the middle of window (not at edges)
        if min_idx <= 1 or min_idx >= len(recent_y) - 2:
            return False
        
        # Check if it's a true local minimum:
        # Values before minimum should be smaller (foot going down)
        # Values after minimum should be larger (foot going up)
        
        before_min = recent_y[:min_idx]
        after_min = recent_y[min_idx+1:]
        min_value = recent_y[min_idx]
        
        # Check downward trend before minimum
        going_down = len(before_min) >= 2 and before_min[-1] < before_min[-2]
        
        # Check upward trend after minimum  
        going_up = len(after_min) >= 1 and after_min[0] > min_value
        
        # Additional check: minimum should be "significant" (not just noise)
        if len(before_min) > 0 and len(after_min) > 0:
            height_drop = before_min[0] - min_value
            height_rise = after_min[-1] - min_value
            significant_motion = height_drop > 0.01 and height_rise > 0.01  # 1% of screen height
        else:
            significant_motion = False
        
        return going_down and going_up and significant_motion
    
    def analyze_frame(self, frame, timestamp, frame_number):
        """Analyze single frame for heel-based footstep detection"""
        landmarks = self.detect_pose(frame)
        
        # Initialize frame record
        frame_record = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'left_heel_x': None,
            'left_heel_y': None,
            'left_heel_confidence': None,
            'left_heel_usable': False,
            'right_heel_x': None,
            'right_heel_y': None,
            'right_heel_confidence': None,
            'right_heel_usable': False,
            'left_step_detected': False,
            'right_step_detected': False
        }
        
        if not landmarks:
            self.frame_data.append(frame_record)
            return []
        
        heel_data = self.extract_heel_positions(landmarks)
        if not heel_data:
            self.frame_data.append(frame_record)
            return []
        
        # Update frame record with heel data
        for foot_name in ['left', 'right']:
            heel = heel_data[foot_name]
            frame_record[f'{foot_name}_heel_x'] = heel['x']
            frame_record[f'{foot_name}_heel_y'] = heel['y']
            frame_record[f'{foot_name}_heel_confidence'] = heel['confidence']
            frame_record[f'{foot_name}_heel_usable'] = heel['usable']
        
        detected_steps = []
        
        # Process each foot
        for foot_name in ['left', 'right']:
            heel = heel_data[foot_name]
            
            # Skip if heel confidence is too low
            if not heel['usable']:
                continue
            
            # Add to history
            history = self.left_heel_history if foot_name == 'left' else self.right_heel_history
            history.append({
                'timestamp': timestamp,
                'frame': frame_number,
                'x': heel['x'],
                'y': heel['y'],
                'confidence': heel['confidence']
            })
            
            # Check for local minimum
            y_values = [h['y'] for h in history]
            
            if self.detect_local_minimum(y_values, frame_number):
                # Check minimum time between steps
                last_step_time = self.last_step_times[foot_name]
                if timestamp - last_step_time >= self.min_step_interval:
                    
                    # Find the exact minimum point in recent history
                    recent_history = list(history)[-self.window_size:]
                    min_point = min(recent_history, key=lambda h: h['y'])
                    
                    step = DetectedStep(
                        timestamp=min_point['timestamp'],
                        foot=foot_name,
                        position_x=min_point['x'],
                        position_y=min_point['y'],
                        confidence=min_point['confidence'],
                        frame_number=min_point['frame']
                    )
                    
                    detected_steps.append(step)
                    self.last_step_times[foot_name] = min_point['timestamp']
                    
                    # Mark detection in frame record
                    frame_record[f'{foot_name}_step_detected'] = True
                    
                    print(f"HEEL STEP: {foot_name} at {min_point['timestamp']:.3f}s "
                          f"(frame {min_point['frame']}, y={min_point['y']:.3f}, conf={min_point['confidence']:.3f})")
        
        self.frame_data.append(frame_record)
        return detected_steps
    
    def save_frame_data_csv(self, output_path):
        """Save frame-by-frame data to CSV file"""
        if not self.frame_data:
            print("No frame data to save")
            return
        
        fieldnames = [
            'frame_number', 'timestamp',
            'left_heel_x', 'left_heel_y', 'left_heel_confidence', 'left_heel_usable',
            'right_heel_x', 'right_heel_y', 'right_heel_confidence', 'right_heel_usable',
            'left_step_detected', 'right_step_detected'
        ]
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.frame_data)
        
        print(f"Frame data saved to: {output_path}")
        print(f"Total frames: {len(self.frame_data)}")
    
    def analyze_video(self, video_path, progress_callback=None):
        """Analyze entire video for heel-based footstep detection"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Analyzing video: {video_path}")
        print(f"FPS: {fps}, Duration: {duration:.2f}s, Total frames: {total_frames}")
        print(f"Starting heel detection...")
        print()
        
        # Clear previous data
        self.frame_data = []
        
        all_steps = []
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / fps
            steps = self.analyze_frame(frame, timestamp, frame_number)
            all_steps.extend(steps)
            
            frame_number += 1
            
            # Progress callback
            if progress_callback and frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                progress_callback(progress, len(all_steps))
        
        cap.release()
        
        print(f"\nHeel detection completed!")
        print(f"Total steps detected: {len(all_steps)}")
        print(f"Left steps: {len([s for s in all_steps if s.foot == 'left'])}")
        print(f"Right steps: {len([s for s in all_steps if s.foot == 'right'])}")
        
        return all_steps

def compare_with_ground_truth(detected_steps, ground_truth_file, tolerance=0.2):
    """Compare detected steps with manual ground truth annotations"""
    
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    gt_steps = gt_data['annotations']
    
    print(f"\n{'='*50}")
    print(f"COMPARISON WITH GROUND TRUTH")
    print(f"{'='*50}")
    print(f"Ground truth steps: {len(gt_steps)}")
    print(f"Detected steps: {len(detected_steps)}")
    print(f"Tolerance: ±{tolerance}s")
    print()
    
    # Find matches
    matches = []
    false_positives = []
    missed_detections = list(gt_steps)  # Start with all GT steps as "missed"
    
    for detected in detected_steps:
        best_match = None
        best_distance = float('inf')
        
        for gt_step in gt_steps:
            if gt_step['foot'] == detected.foot:  # Same foot
                time_diff = abs(detected.timestamp - gt_step['timestamp'])
                if time_diff <= tolerance and time_diff < best_distance:
                    best_match = gt_step
                    best_distance = time_diff
        
        if best_match:
            matches.append({
                'detected': detected,
                'ground_truth': best_match,
                'time_diff': best_distance
            })
            # Remove from missed detections
            if best_match in missed_detections:
                missed_detections.remove(best_match)
        else:
            false_positives.append(detected)
    
    # Calculate metrics
    precision = len(matches) / len(detected_steps) if detected_steps else 0
    recall = len(matches) / len(gt_steps) if gt_steps else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"RESULTS:")
    print(f"  Correct matches: {len(matches)}")
    print(f"  False positives: {len(false_positives)}")
    print(f"  Missed detections: {len(missed_detections)}")
    print()
    print(f"METRICS:")
    print(f"  Precision: {precision:.2%} ({len(matches)}/{len(detected_steps)})")
    print(f"  Recall: {recall:.2%} ({len(matches)}/{len(gt_steps)})")
    print(f"  F1-Score: {f1_score:.2%}")
    
    # Show timing accuracy for matches
    if matches:
        time_errors = [m['time_diff'] for m in matches]
        print(f"\nTIMING ACCURACY:")
        print(f"  Average error: {np.mean(time_errors):.3f}s")
        print(f"  Max error: {np.max(time_errors):.3f}s")
        print(f"  Std deviation: {np.std(time_errors):.3f}s")
    
    return {
        'matches': matches,
        'false_positives': false_positives,
        'missed_detections': missed_detections,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def test_heel_detection(video_path, ground_truth_file):
    """Test pure heel detection on one video"""
    
    print(f"TESTING PURE HEEL DETECTION")
    print(f"Video: {video_path}")
    print(f"Ground truth: {ground_truth_file}")
    print()
    
    # Create detector
    detector = PureHeelDetector(
        min_confidence=0.5,  # Start with moderate confidence requirement
        window_size=7,       # Updated to 7 frames as discussed
        min_step_interval=0.3  # Minimum 0.3s between steps of same foot
    )
    
    # Detect steps
    def progress_callback(progress, step_count):
        if progress % 20 == 0:  # Print every 20%
            print(f"Progress: {progress:.0f}% | Steps found: {step_count}")
    
    detected_steps = detector.analyze_video(video_path, progress_callback)
    
    # Save frame-by-frame data to CSV
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"{video_name}_frame_data.csv"
    detector.save_frame_data_csv(csv_path)
    
    # Compare with ground truth
    comparison = compare_with_ground_truth(detected_steps, ground_truth_file)
    
    return detected_steps, comparison

if __name__ == "__main__":
    # Updated paths for your 30 FPS video
    video_file = "/Users/yejinbang/Documents/GitHub/sfx-project/data/test_videos/walk4.mp4"
    gt_file = "/Users/yejinbang/Documents/GitHub/sfx-project/data/test_videos/walk4_ground_truth.json"
    
    if os.path.exists(video_file) and os.path.exists(gt_file):
        detected_steps, comparison = test_heel_detection(video_file, gt_file)
    else:
        print(f"Files not found:")
        print(f"  Video: {video_file} (exists: {os.path.exists(video_file)})")
        print(f"  Ground truth: {gt_file} (exists: {os.path.exists(gt_file)})")
        print("Please check file paths.")
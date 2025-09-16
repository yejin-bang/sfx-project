import cv2
import numpy as np
import mediapipe as mp
import json
from collections import deque
from dataclasses import dataclass
from typing import List
import csv

@dataclass
class DetectedStep:
    timestamp: float
    foot: str
    position_x: float
    position_y: float
    confidence: float
    frame_number: int

class PureAnkleDetector:
    def __init__(self, min_confidence=0.5, window_size=5, min_step_interval=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
        self.min_confidence = min_confidence
        self.window_size = window_size
        self.min_step_interval = min_step_interval
        self.left_ankle_history = deque(maxlen=window_size*2)
        self.right_ankle_history = deque(maxlen=window_size*2)
        self.last_step_times = {'left': -999, 'right': -999}

    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks if results.pose_landmarks else None

    def extract_ankle_positions(self, landmarks):
        if not landmarks:
            return None
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        return {
            'left': {'x': left_ankle.x, 'y': left_ankle.y, 'confidence': left_ankle.visibility, 'usable': left_ankle.visibility >= self.min_confidence},
            'right': {'x': right_ankle.x, 'y': right_ankle.y, 'confidence': right_ankle.visibility, 'usable': right_ankle.visibility >= self.min_confidence}
        }

    def detect_local_minimum(self, y_history):
        if len(y_history) < self.window_size:
            return False
        recent_y = list(y_history)[-self.window_size:]
        min_idx = np.argmin(recent_y)
        if min_idx <= 1 or min_idx >= len(recent_y)-2:
            return False
        before_min = recent_y[:min_idx]
        after_min = recent_y[min_idx+1:]
        min_value = recent_y[min_idx]
        going_down = len(before_min) >= 2 and before_min[-1] < before_min[-2]
        going_up = len(after_min) >= 1 and after_min[0] > min_value
        if len(before_min) > 0 and len(after_min) > 0:
            height_drop = before_min[0] - min_value
            height_rise = after_min[-1] - min_value
            significant_motion = height_drop > 0.01 and height_rise > 0.01
        else:
            significant_motion = False
        return going_down and going_up and significant_motion

    def analyze_frame(self, frame, timestamp, frame_number):
        landmarks = self.detect_pose(frame)
        if not landmarks:
            return [], None
        ankle_data = self.extract_ankle_positions(landmarks)
        if not ankle_data:
            return [], None
        detected_steps = []
        local_minimum_flags = {'left':0, 'right':0}

        for foot_name in ['left','right']:
            ankle = ankle_data[foot_name]
            history = self.left_ankle_history if foot_name=='left' else self.right_ankle_history
            if ankle['usable']:
                history.append({'timestamp':timestamp,'frame':frame_number,'x':ankle['x'],'y':ankle['y'],'confidence':ankle['confidence']})
                y_values = [h['y'] for h in history]
                if self.detect_local_minimum(y_values):
                    if timestamp - self.last_step_times[foot_name] >= self.min_step_interval:
                        recent_history = list(history)[-self.window_size:]
                        min_point = min(recent_history, key=lambda h:h['y'])
                        step = DetectedStep(timestamp=min_point['timestamp'], foot=foot_name,
                                            position_x=min_point['x'], position_y=min_point['y'],
                                            confidence=min_point['confidence'], frame_number=min_point['frame'])
                        detected_steps.append(step)
                        self.last_step_times[foot_name] = min_point['timestamp']
                        local_minimum_flags[foot_name] = 1
        return detected_steps, local_minimum_flags

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_steps = []
        csv_rows = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_number / fps
            steps, min_flags = self.analyze_frame(frame, timestamp, frame_number)
            all_steps.extend(steps)
            # Save every frame for CSV (Left and Right separately)
            landmarks = self.detect_pose(frame)
            ankle_data = self.extract_ankle_positions(landmarks) if landmarks else None
            if ankle_data:
                for foot_name in ['left','right']:
                    ankle = ankle_data[foot_name]
                    usable = ankle['usable'] if ankle else 0
                    y = ankle['y'] if ankle else 0
                    conf = ankle['confidence'] if ankle else 0
                    is_min = min_flags[foot_name] if min_flags else 0
                    csv_rows.append([frame_number, foot_name, f"{y:.5f}", f"{conf:.3f}", is_min])
            frame_number += 1

        cap.release()
        # Save CSV
        with open("ankle_steps_full.csv","w",newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_number","foot","y","confidence","is_minimum"])
            writer.writerows(csv_rows)
        return all_steps

def compare_with_ground_truth(detected_steps, ground_truth_file, tolerance=0.2):
    with open(ground_truth_file,'r') as f:
        gt_data = json.load(f)
    gt_steps = gt_data['annotations']
    matches, false_positives, missed_detections = [], [], list(gt_steps)
    for detected in detected_steps:
        best_match = None
        best_distance = float('inf')
        for gt_step in gt_steps:
            if gt_step['foot']==detected.foot:
                diff = abs(detected.timestamp - gt_step['timestamp'])
                if diff <= tolerance and diff < best_distance:
                    best_match = gt_step
                    best_distance = diff
        if best_match:
            matches.append({'detected':detected,'ground_truth':best_match,'time_diff':best_distance})
            if best_match in missed_detections:
                missed_detections.remove(best_match)
        else:
            false_positives.append(detected)
    precision = len(matches)/len(detected_steps) if detected_steps else 0
    recall = len(matches)/len(gt_steps) if gt_steps else 0
    f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    return {
        'matches':matches,
        'false_positives':false_positives,
        'missed_detections':missed_detections,
        'precision':precision,
        'recall':recall,
        'f1_score':f1_score
    }

def test_ankle_detection(video_path, ground_truth_file):
    detector = PureAnkleDetector(min_confidence=0.5, window_size=5, min_step_interval=0.3)
    detected_steps = detector.analyze_video(video_path)
    comparison = compare_with_ground_truth(detected_steps, ground_truth_file)
    return detected_steps, comparison

if __name__=="__main__":
    video_file = "./data/test_videos/walk4.mp4"
    gt_file = "./data/test_videos/walk4_ground_truth.json"
    detected_steps, comparison = test_ankle_detection(video_file, gt_file)
    print(f"CSV saved as 'ankle_steps_full.csv', total detected steps: {len(detected_steps)}")

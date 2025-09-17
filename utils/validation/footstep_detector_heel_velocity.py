import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from dataclasses import dataclass
from typing import List

@dataclass
class FootstepEvent:
    timestamp: float
    foot: str  # 'left' or 'right'
    person_id: int
    position_x: float  # normalized 0-1
    position_y: float  # normalized 0-1
    distance_estimate: float  # 0-1
    confidence: float  # visibility
    velocity: float  # y-direction velocity at impact

class FootstepDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 velocity_threshold=0.01, smoothing_window=5):
        print("Initializing MediaPipe Heel-based Footstep Detector...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True
        )
        self.velocity_threshold = velocity_threshold
        self.smoothing_window = smoothing_window
        self.person_trackers = {}
        self.frame_count = 0
        self.fps = 30
        print("Initialized.")

    def set_video_fps(self, fps):
        self.fps = fps

    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks if results.pose_landmarks else None

    def extract_foot_positions(self, landmarks, frame_shape):
        if not landmarks:
            return None
        height, width = frame_shape[:2]

        left_heel = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        
        # 몸 높이 기반 거리 추정
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # 머리에서 발목까지의 높이
        body_height = abs(nose.y - (left_ankle.y + right_ankle.y) / 2)
        distance_estimate = min(1.0, max(0.1, 1.0 / max(0.1, body_height)))
        
        return {
            'left_foot': {'x': left_heel.x, 'y': left_heel.y, 'visibility': left_heel.visibility},
            'right_foot': {'x': right_heel.x, 'y': right_heel.y, 'visibility': right_heel.visibility},
            'distance_estimate': distance_estimate,
            'person_center_x': (left_heel.x + right_heel.x) / 2
        }

    def calculate_foot_velocity(self, current_pos, previous_positions):
        if len(previous_positions) < 2:
            return 0.0
        velocities = []
        for i in range(1, min(len(previous_positions), self.smoothing_window)):
            dy = current_pos[1] - previous_positions[-i-1][1]
            velocities.append(dy)
        return np.mean(velocities) if velocities else 0.0

    def detect_footstep_impact(self, foot_data, person_tracker, timestamp):
        footsteps = []
        
        # 좌우 발 교대 추적 초기화
        if 'last_foot' not in person_tracker:
            person_tracker['last_foot'] = None
            
        for foot_name in ['left_foot', 'right_foot']:
            foot_pos = foot_data[foot_name]
            if foot_pos['visibility'] < 0.5:
                continue

            current_position = (foot_pos['x'], foot_pos['y'])
            if foot_name not in person_tracker['positions']:
                person_tracker['positions'][foot_name] = deque(maxlen=self.smoothing_window*2)

            position_history = person_tracker['positions'][foot_name]
            position_history.append(current_position)
            if len(position_history) < 3:
                continue

            current_velocity = self.calculate_foot_velocity(current_position, position_history)
            if foot_name not in person_tracker['velocities']:
                person_tracker['velocities'][foot_name] = deque(maxlen=self.smoothing_window)
            velocity_history = person_tracker['velocities'][foot_name]
            velocity_history.append(current_velocity)

            # 원래 30% 규칙 유지
            if len(velocity_history) >= 3:
                recent = list(velocity_history)[-3:]
                if recent[1] < -self.velocity_threshold and abs(recent[2]) < abs(recent[1]) * 0.3:
                    last_step_key = f"{person_tracker['person_id']}_{foot_name}_last_step"
                    
                    # 같은 발이 연속으로 나오는 것 방지
                    current_foot = foot_name.split('_')[0]
                    if person_tracker['last_foot'] == current_foot:
                        if timestamp - person_tracker[last_step_key] < 0.8:  # 더 긴 간격 요구
                            continue
                    
                    if last_step_key not in person_tracker or timestamp - person_tracker[last_step_key] > 0.3:
                        footsteps.append(FootstepEvent(
                            timestamp=timestamp,
                            foot=foot_name.split('_')[0],
                            person_id=person_tracker['person_id'],
                            position_x=foot_pos['x'],
                            position_y=foot_pos['y'],
                            distance_estimate=foot_data['distance_estimate'],
                            confidence=foot_pos['visibility'],
                            velocity=recent[1]
                        ))
                        person_tracker[last_step_key] = timestamp
                        person_tracker['last_foot'] = current_foot
                        print(f"Footstep detected: {foot_name.split('_')[0]} at {timestamp:.2f}s")
        return footsteps

    def analyze_frame(self, frame, timestamp):
        self.frame_count += 1
        landmarks = self.detect_pose(frame)
        if not landmarks:
            return []
        foot_data = self.extract_foot_positions(landmarks, frame.shape)
        if not foot_data:
            return []

        person_id = 0
        if person_id not in self.person_trackers:
            self.person_trackers[person_id] = {'person_id': person_id, 'positions': {}, 'velocities': {}, 'last_seen': timestamp}
        person_tracker = self.person_trackers[person_id]
        person_tracker['last_seen'] = timestamp
        return self.detect_footstep_impact(foot_data, person_tracker, timestamp)

    def analyze_video(self, video_path, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.set_video_fps(fps)

        all_footsteps = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_number / fps
            footsteps = self.analyze_frame(frame, timestamp)
            all_footsteps.extend(footsteps)
            frame_number += 1
            if progress_callback and frame_number % 30 == 0:
                progress_callback((frame_number / total_frames)*100, len(all_footsteps))

        cap.release()
        print(f"Total footsteps detected: {len(all_footsteps)}")
        return all_footsteps
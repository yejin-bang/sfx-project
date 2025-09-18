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
                 velocity_threshold=0.005, smoothing_window=5):
        print("Initializing MediaPipe Ankle-based Footstep Detector...")
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
        print("Initialized with ankle tracking and relaxed thresholds.")

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

        # Heel 대신 Ankle 사용
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # 몸 높이 기반 거리 추정
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        # 머리에서 발목까지의 높이
        body_height = abs(nose.y - (left_ankle.y + right_ankle.y) / 2)
        distance_estimate = min(1.0, max(0.1, 1.0 / max(0.1, body_height)))
        
        return {
            'left_foot': {'x': left_ankle.x, 'y': left_ankle.y, 'visibility': left_ankle.visibility},
            'right_foot': {'x': right_ankle.x, 'y': right_ankle.y, 'visibility': right_ankle.visibility},
            'distance_estimate': distance_estimate,
            'person_center_x': (left_ankle.x + right_ankle.x) / 2
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

            # 발걸음 검출 로직
            if len(velocity_history) >= 3:
                recent = list(velocity_history)[-3:]
                # 하향 움직임 후 속도가 50% 이상 감소하거나 거의 멈춤
                speed_dropped = abs(recent[2]) < abs(recent[1]) * 0.5
                nearly_stopped = abs(recent[2]) < self.velocity_threshold * 2
                
                # 발걸음 조건 확인
                footstep_detected = (recent[1] < -self.velocity_threshold and 
                                   (speed_dropped or nearly_stopped))
                
                if footstep_detected:
                    last_step_key = f"{person_tracker['person_id']}_{foot_name}_last_step"
                    
                    # 같은 발이 연속으로 나오는 것 방지
                    current_foot = foot_name.split('_')[0]
                    if person_tracker['last_foot'] == current_foot:
                        if (last_step_key in person_tracker and 
                            timestamp - person_tracker[last_step_key] < 0.8):
                            continue
                    
                    # 최소 시간 간격 확인
                    if (last_step_key not in person_tracker or 
                        timestamp - person_tracker[last_step_key] > 0.3):
                        
                        # 속도 계산 (이전 속도 사용)
                        velocity_magnitude = abs(recent[1]) if len(recent) > 1 else abs(current_velocity)
                        
                        footsteps.append(FootstepEvent(
                            timestamp=timestamp,
                            foot=current_foot,
                            person_id=person_tracker['person_id'],
                            position_x=foot_pos['x'],
                            position_y=foot_pos['y'],
                            distance_estimate=foot_data['distance_estimate'],
                            confidence=foot_pos['visibility'],
                            velocity=velocity_magnitude
                        ))
                        
                        person_tracker[last_step_key] = timestamp
                        person_tracker['last_foot'] = current_foot
                        print(f"Footstep detected: {current_foot} at {timestamp:.2f}s "
                              f"(velocity={velocity_magnitude:.4f})")
                        
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
            self.person_trackers[person_id] = {
                'person_id': person_id, 
                'positions': {}, 
                'velocities': {}, 
                'last_seen': timestamp
            }
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

# 테스트용 함수들
def test_footstep_detector(video_path, show_visualization=True, save_results=True):
    """
    발걸음 검출 테스트 함수
    """
    detector = FootstepDetector(
        velocity_threshold=0.01,
        smoothing_window=5
    )
    
    print(f"Testing video: {video_path}")
    start_time = time.time()
    
    def progress_callback(progress, step_count):
        print(f"Progress: {progress:.1f}% | Steps detected: {step_count}")
    
    footsteps = detector.analyze_video(video_path, progress_callback)
    
    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
    print(f"Total footsteps: {len(footsteps)}")
    
    # 발걸음 통계 출력
    if footsteps:
        left_steps = [f for f in footsteps if f.foot == 'left']
        right_steps = [f for f in footsteps if f.foot == 'right']
        
        print(f"Left foot steps: {len(left_steps)}")
        print(f"Right foot steps: {len(right_steps)}")
        
        # 시간 간격 분석
        if len(footsteps) > 1:
            intervals = []
            for i in range(1, len(footsteps)):
                interval = footsteps[i].timestamp - footsteps[i-1].timestamp
                intervals.append(interval)
            
            avg_interval = np.mean(intervals)
            estimated_pace = 60 / avg_interval if avg_interval > 0 else 0
            
            print(f"Average step interval: {avg_interval:.2f} seconds")
            print(f"Estimated pace: {estimated_pace:.1f} steps/minute")
    
    return footsteps

def visualize_footsteps_on_video(video_path, footsteps, output_path=None):
    """
    비디오에 발걸음 검출 결과를 시각화
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 출력 비디오 설정
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 발걸음 이벤트를 시간별로 정리
    footstep_dict = {}
    for step in footsteps:
        frame_num = int(step.timestamp * fps)
        if frame_num not in footstep_dict:
            footstep_dict[frame_num] = []
        footstep_dict[frame_num].append(step)
    
    frame_count = 0
    recent_steps = deque(maxlen=30)  # 최근 30프레임 동안의 스텝 표시
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 현재 프레임의 발걸음 체크
        if frame_count in footstep_dict:
            for step in footstep_dict[frame_count]:
                recent_steps.append((step, frame_count))
                print(f"Frame {frame_count}: {step.foot} step detected")
        
        # 최근 발걸음들 시각화
        for step, step_frame in recent_steps:
            age = frame_count - step_frame
            alpha = max(0, 1.0 - age / 30)  # 점점 희미해짐
            
            # 발 위치에 원 그리기
            x = int(step.position_x * width)
            y = int(step.position_y * height)
            
            color = (0, 255, 0) if step.foot == 'left' else (0, 0, 255)  # 왼발=녹색, 오른발=빨강
            radius = int(20 * alpha)
            
            if radius > 0:
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.putText(frame, step.foot.upper(), (x-15, y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 정보 텍스트 오버레이
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Steps: {len(footsteps)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if output_path:
            out.write(frame)
        else:
            cv2.imshow('Footstep Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    if output_path:
        out.release()
        print(f"Visualization saved to: {output_path}")
    else:
        cv2.destroyAllWindows()

# 사용 예시
if __name__ == "__main__":
    # 테스트 실행
    video_file = "./data/test_videos/walk1.mp4"  # 여기에 실제 비디오 파일 경로 입력
    
    try:
        # 1. 발걸음 검출 테스트
        footsteps = test_footstep_detector(video_file)
        
        # 2. 시각화 (선택사항)
        # visualize_footsteps_on_video(video_file, footsteps, "output_with_footsteps.mp4")
        
        # 또는 실시간 시각화
        visualize_footsteps_on_video(video_file, footsteps)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the video file exists and the path is correct.")
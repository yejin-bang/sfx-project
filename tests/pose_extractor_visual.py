import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.pose_extractor import PoseExtractor

class PoseVideoOverlay:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # Extract pose data using our PoseExtractor
        print("Extracting pose data...")
        self.extractor = PoseExtractor(target_fps=10, confidence_threshold=0.7)
        self.pose_result = self.extractor.process_video(video_path, verbose=True)
        
        # Create frame-to-pose mapping
        self.frame_to_pose = {}
        self._create_frame_mapping()
        
        print("\nVIDEO OVERLAY INFO:")
        print("  RED dot = Left Hip")
        print("  BLUE dot = Right Hip") 
        print("  GREEN dot = Left Heel")
        print("  YELLOW dot = Right Heel")
        print("  CYAN dot = Left Ankle")
        print("  MAGENTA dot = Right Ankle")
        print("  ORANGE dot = Left Foot Index")
        print("  PURPLE dot = Right Foot Index")
        print()
        print("CONTROLS:")
        print("  SPACE = Play/Pause")
        print("  A/D = Previous/Next frame (when paused)")
        print("  R = Reset to beginning")
        print("  Q/ESC = Quit")
        print()
    
    def _create_frame_mapping(self):
        """Create mapping from video frame numbers to pose data"""
        # Get video info
        cap = cv2.VideoCapture(self.video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Map timestamps to frame numbers
        for i, timestamp in enumerate(self.pose_result['timestamps']):
            original_frame_number = int(timestamp * original_fps)
            self.frame_to_pose[original_frame_number] = i
        
        print(f"Created mapping for {len(self.frame_to_pose)} frames")
    
    def get_landmark_colors(self):
        """Define colors for each landmark"""
        return {
            'LEFT_HIP': (0, 0, 255),        # RED
            'RIGHT_HIP': (255, 0, 0),       # BLUE
            'LEFT_HEEL': (0, 255, 0),       # GREEN
            'RIGHT_HEEL': (0, 255, 255),    # YELLOW
            'LEFT_ANKLE': (255, 255, 0),    # CYAN
            'RIGHT_ANKLE': (255, 0, 255),   # MAGENTA
            'LEFT_FOOT_INDEX': (0, 165, 255),   # ORANGE
            'RIGHT_FOOT_INDEX': (128, 0, 128)   # PURPLE
        }
    
    def draw_pose_overlays(self, frame, frame_number):
        """Draw pose landmarks on frame"""
        # Check if we have pose data for this frame
        if frame_number not in self.frame_to_pose:
            return frame
        
        pose_index = self.frame_to_pose[frame_number]
        landmarks = self.pose_result['landmarks'][pose_index]
        
        if not landmarks:
            return frame
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        colors = self.get_landmark_colors()
        
        # Draw each landmark
        for landmark_name, (x, y, confidence) in landmarks.items():
            if x is not None and y is not None:  # Valid detection
                # Convert to pixel coordinates (already converted in pose_extractor)
                pixel_x = int(x)
                pixel_y = int(y)
                
                # Get color for this landmark
                color = colors.get(landmark_name, (255, 255, 255))  # Default white
                
                # Draw landmark dot
                cv2.circle(overlay, (pixel_x, pixel_y), 8, color, -1)
                
                # Draw confidence text
                conf_text = f"{confidence:.2f}"
                cv2.putText(overlay, conf_text, (pixel_x + 12, pixel_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw connections between related landmarks
        self.draw_connections(overlay, landmarks, colors)
        
        return overlay
    
    def draw_connections(self, overlay, landmarks, colors):
        """Draw lines connecting related landmarks"""
        connections = [
            ('LEFT_HIP', 'LEFT_ANKLE'),
            ('LEFT_ANKLE', 'LEFT_HEEL'),
            ('LEFT_ANKLE', 'LEFT_FOOT_INDEX'),
            ('RIGHT_HIP', 'RIGHT_ANKLE'),
            ('RIGHT_ANKLE', 'RIGHT_HEEL'),
            ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),
            ('LEFT_HIP', 'RIGHT_HIP')  # Hip connection
        ]
        
        for landmark1, landmark2 in connections:
            if (landmark1 in landmarks and landmark2 in landmarks and
                landmarks[landmark1][0] is not None and landmarks[landmark2][0] is not None):
                
                x1, y1, _ = landmarks[landmark1]
                x2, y2, _ = landmarks[landmark2]
                
                # Draw line
                cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                        (200, 200, 200), 2)
    
    def draw_info_panel(self, overlay, frame_number, timestamp, total_frames):
        """Draw information panel on the frame"""
        h, w = overlay.shape[:2]
        
        # Frame information
        info_text = f"Frame: {frame_number:4d}/{total_frames} | Time: {timestamp:.3f}s"
        cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Processing information
        if frame_number in self.frame_to_pose:
            pose_text = "Pose: DETECTED"
            cv2.putText(overlay, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            pose_text = "Pose: NOT PROCESSED"
            cv2.putText(overlay, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw legend
        self.draw_legend(overlay)
        
        # Draw controls
        controls = "SPACE=Play/Pause | A/D=Frame step | R=Reset | Q=Quit"
        cv2.putText(overlay, controls, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def draw_legend(self, overlay):
        """Draw color legend for landmarks"""
        colors = self.get_landmark_colors()
        legend_items = [
            ('LEFT_HIP', 'L Hip'),
            ('RIGHT_HIP', 'R Hip'),
            ('LEFT_HEEL', 'L Heel'),
            ('RIGHT_HEEL', 'R Heel'),
            ('LEFT_ANKLE', 'L Ankle'),
            ('RIGHT_ANKLE', 'R Ankle'),
            ('LEFT_FOOT_INDEX', 'L Foot'),
            ('RIGHT_FOOT_INDEX', 'R Foot')
        ]
        
        legend_y = 90
        for landmark_name, display_name in legend_items:
            color = colors[landmark_name]
            
            # Draw colored circle
            cv2.circle(overlay, (30, legend_y), 6, color, -1)
            
            # Draw text
            cv2.putText(overlay, display_name, (50, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            legend_y += 20
    
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
                
                # Draw pose overlays
                display_frame = self.draw_pose_overlays(frame, current_frame)
                
                # Draw information panel
                self.draw_info_panel(display_frame, current_frame, timestamp, total_frames)
                
                # Show frame
                cv2.imshow('Pose Landmarks Visualization', display_frame)
                
                # Handle input
                key = cv2.waitKey(30 if not paused else 0) & 0xFF
                
                if key == ord(' '):  # Space - play/pause
                    paused = not paused
                    print("PAUSED" if paused else "PLAYING")
                
                elif key == ord('a') and paused:  # A - previous frame
                    current_frame = max(0, current_frame - 1)
                
                elif key == ord('d') and paused:  # D - next frame
                    current_frame = min(total_frames - 1, current_frame + 1)
                
                elif key == ord('r'):  # R - reset to beginning
                    current_frame = 0
                    print("Reset to beginning")
                
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
            self.extractor.cleanup()
            
            # Print summary
            print(f"\nSUMMARY:")
            print(f"Total frames: {total_frames}")
            print(f"Frames with pose data: {len(self.frame_to_pose)}")
            print(f"Pose detection coverage: {len(self.frame_to_pose)/total_frames*100:.1f}%")

def visualize_pose_on_video(video_path):
    """Main function to visualize pose landmarks on video"""
    try:
        visualizer = PoseVideoOverlay(video_path)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Video path
    video_file = "/Users/yejinbang/Documents/GitHub/sfx-project/data/test_videos/walk4.mp4"
    
    print(f"Using video: {video_file}")
    print()
    
    visualize_pose_on_video(video_file)
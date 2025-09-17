import cv2
import json
import os
from datetime import datetime

class FootstepAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        # Annotation data
        self.annotations = []
        self.current_frame = 0
        self.paused = True
        
        # Create output filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_file = f"{video_name}_ground_truth.json"
        
        print(f"Video loaded: {video_path}")
        print(f"FPS: {self.fps:.2f}, Duration: {self.duration:.2f}s, Total frames: {self.total_frames}")
        print(f"Annotations will be saved to: {self.output_file}")
        print()
        print("CONTROLS:")
        print("  SPACE      - Play/Pause")
        print("  A/D        - Previous/Next frame (when paused)")
        print("  LEFT Arrow - Mark LEFT foot step")
        print("  RIGHT Arrow- Mark RIGHT foot step") 
        print("  U          - Undo last annotation")
        print("  S          - Save annotations")
        print("  Q/ESC      - Quit")
        print("  H          - Show help")
        print()
    
    def get_timestamp(self):
        """Get current timestamp in seconds"""
        return self.current_frame / self.fps
    
    def add_annotation(self, foot):
        """Add footstep annotation at current frame"""
        timestamp = self.get_timestamp()
        annotation = {
            'frame': self.current_frame,
            'timestamp': round(timestamp, 3),
            'foot': foot,
            'datetime': datetime.now().isoformat()
        }
        self.annotations.append(annotation)
        
        print(f"Added {foot.upper()} foot step at frame {self.current_frame} (t={timestamp:.3f}s)")
        return annotation
    
    def undo_last_annotation(self):
        """Remove the last annotation"""
        if self.annotations:
            removed = self.annotations.pop()
            print(f"Removed {removed['foot'].upper()} foot step at frame {removed['frame']}")
        else:
            print("No annotations to undo")
    
    def save_annotations(self):
        """Save annotations to JSON file"""
        data = {
            'video_path': self.video_path,
            'video_info': {
                'fps': self.fps,
                'total_frames': self.total_frames,
                'duration': self.duration
            },
            'annotations': sorted(self.annotations, key=lambda x: x['timestamp']),
            'summary': {
                'total_steps': len(self.annotations),
                'left_steps': len([a for a in self.annotations if a['foot'] == 'left']),
                'right_steps': len([a for a in self.annotations if a['foot'] == 'right']),
                'created': datetime.now().isoformat()
            }
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.annotations)} annotations to {self.output_file}")
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("FOOTSTEP ANNOTATION HELP")
        print("="*50)
        print("PLAYBACK:")
        print("  SPACE      - Toggle play/pause")
        print("  A          - Previous frame (when paused)")
        print("  D          - Next frame (when paused)")
        print()
        print("ANNOTATION:")
        print("  LEFT Arrow - Mark LEFT foot touching ground")
        print("  RIGHT Arrow- Mark RIGHT foot touching ground")
        print("  U          - Undo last annotation")
        print()
        print("FILE:")
        print("  S          - Save annotations to file")
        print("  Q/ESC      - Quit (will prompt to save)")
        print("  H          - Show this help")
        print()
        print("TIPS:")
        print("- Look for the EXACT moment heel/foot first touches ground")
        print("- Use frame-by-frame mode (A/D keys) for precision")
        print("- Mark the INITIAL CONTACT, not when foot is flat")
        print("- Save frequently!")
        print("="*50 + "\n")
    
    def show_current_info(self):
        """Show current frame and annotation info"""
        timestamp = self.get_timestamp()
        status = "PLAYING" if not self.paused else "PAUSED"
        
        # Recent annotations
        recent = [a for a in self.annotations if abs(a['timestamp'] - timestamp) < 2.0]
        recent_str = ", ".join([f"{a['foot'][0].upper()}@{a['timestamp']:.2f}s" for a in recent[-3:]])
        
        info = f"Frame: {self.current_frame:4d}/{self.total_frames} | Time: {timestamp:6.3f}s | {status} | Recent: {recent_str}"
        return info
    
    def draw_annotations_overlay(self, frame):
        """Draw annotation info on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Info text
        info = self.show_current_info()
        cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Annotation count
        count_text = f"Annotations: {len(self.annotations)} (L: {len([a for a in self.annotations if a['foot'] == 'left'])}, R: {len([a for a in self.annotations if a['foot'] == 'right'])})"
        cv2.putText(overlay, count_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls reminder
        controls = "LEFT=Left step | RIGHT=Right step | SPACE=Play/Pause | A/D=Frame step | S=Save | Q=Quit | H=Help"
        cv2.putText(overlay, controls, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Timeline with annotations
        timeline_y = h - 100
        timeline_start = 50
        timeline_width = w - 100
        
        # Draw timeline
        cv2.rectangle(overlay, (timeline_start, timeline_y - 5), (timeline_start + timeline_width, timeline_y + 5), (100, 100, 100), -1)
        
        # Current position
        current_pos = int(timeline_start + (self.current_frame / self.total_frames) * timeline_width)
        cv2.line(overlay, (current_pos, timeline_y - 10), (current_pos, timeline_y + 10), (0, 255, 255), 2)
        
        # Annotation markers
        for ann in self.annotations:
            pos = int(timeline_start + (ann['frame'] / self.total_frames) * timeline_width)
            color = (0, 255, 0) if ann['foot'] == 'left' else (0, 0, 255)
            cv2.circle(overlay, (pos, timeline_y), 3, color, -1)
        
        return overlay
    
    def run(self):
        """Main annotation loop"""
        self.show_help()
        
        try:
            while True:
                # Read frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                
                if not ret:
                    print("End of video reached")
                    break
                
                # Add overlay
                display_frame = self.draw_annotations_overlay(frame)
                
                # Show frame
                cv2.imshow('Footstep Annotator', display_frame)
                
                # Handle input
                key = cv2.waitKey(30 if not self.paused else 0) & 0xFF
                
                if key == ord(' '):  # Space - play/pause
                    self.paused = not self.paused
                    print("PAUSED" if self.paused else "PLAYING")
                
                elif key == ord('a') and self.paused:  # A - previous frame
                    self.current_frame = max(0, self.current_frame - 1)
                
                elif key == ord('d') and self.paused:  # D - next frame
                    self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                
                elif key == 81 or key == 2:  # LEFT Arrow - left foot
                    self.add_annotation('left')
                
                elif key == 83 or key == 3:  # RIGHT Arrow - right foot
                    self.add_annotation('right')
                
                elif key == ord('u'):  # U - undo
                    self.undo_last_annotation()
                
                elif key == ord('s'):  # S - save
                    self.save_annotations()
                
                elif key == ord('h'):  # H - help
                    self.show_help()
                
                elif key == ord('q') or key == 27:  # Q or ESC - quit
                    break
                
                # Auto-advance frame if playing
                if not self.paused:
                    self.current_frame += 1
                    if self.current_frame >= self.total_frames:
                        self.current_frame = self.total_frames - 1
                        self.paused = True
                        print("End of video - PAUSED")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Offer to save before exit
            if self.annotations:
                print(f"\nYou have {len(self.annotations)} unsaved annotations.")
                save = input("Save before exit? (y/n): ").lower().strip()
                if save == 'y':
                    self.save_annotations()
            
            print("Annotation session completed.")

# Usage function
def annotate_video(video_path):
    """Start annotation for a video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    try:
        annotator = FootstepAnnotator(video_path)
        annotator.run()
    except Exception as e:
        print(f"Error during annotation: {e}")

if __name__ == "__main__":
    video_file = "./data/test_videos/walk5.mp4"

    if video_file:
        annotate_video(video_file)
    else:
        print("No video file specified")
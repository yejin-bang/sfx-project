from footstep_detector_ankle import FootstepDetector

def test_footstep_detection():
    # 비디오 파일 경로 (본인의 경로로 변경)
    video_path = "./data/test_videos/walk1.mp4" 
    
    print("=== Footstep Detection Test ===")
    print(f"Testing video: {video_path}")
    
    try:
        # FootstepDetector 초기화
        detector = FootstepDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            velocity_threshold=0.01,
            smoothing_window=5
        )
        
        # 비디오 분석
        print("\n>>> Starting video analysis...")
        footsteps = detector.analyze_video(video_path)
        
        print(f"\n=== Results ===")
        print(f"Total footsteps detected: {len(footsteps)}")
        
        if footsteps:
            print("\n>>> Detected footsteps:")
            for i, step in enumerate(footsteps):
                print(f"  {i+1}. {step.foot} foot at {step.timestamp:.2f}s")
                print(f"     Position: ({step.position_x:.3f}, {step.position_y:.3f})")
                print(f"     Distance: {step.distance_estimate:.3f}, Confidence: {step.confidence:.3f}")
                print(f"     Velocity: {step.velocity:.4f}")
                print()
            
            # 간단한 통계
            left_steps = [s for s in footsteps if s.foot == 'left']
            right_steps = [s for s in footsteps if s.foot == 'right']
            
            print(f">>> Statistics:")
            print(f"  Left steps: {len(left_steps)}")
            print(f"  Right steps: {len(right_steps)}")
            
            if len(footsteps) > 1:
                avg_interval = sum(footsteps[i+1].timestamp - footsteps[i].timestamp 
                                 for i in range(len(footsteps)-1)) / (len(footsteps)-1)
                print(f"  Average step interval: {avg_interval:.2f}s")
                print(f"  Estimated pace: {60/avg_interval:.1f} steps/min")
        else:
            print("No footsteps detected. Check:")
            print("  - Video file path")
            print("  - Person visibility in video")
            print("  - Walking motion clarity")
            
    except FileNotFoundError:
        print(f"Error: Video file not found: {video_path}")
        print("Please update the video_path variable with correct path")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_footstep_detection()
import cv2
import os

def convert_video_fps_robust(input_path, target_fps=30):
    """
    More robust video conversion with different codec options
    """
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(directory, f"{name}_{target_fps}fps{ext}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Original FPS: {original_fps}")
    print(f"Resolution: {width}x{height}")
    
    # Try different codecs
    codecs_to_try = [
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'))
    ]
    
    for codec_name, fourcc in codecs_to_try:
        print(f"Trying codec: {codec_name}")
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        if not writer.isOpened():
            print(f"  {codec_name} failed to open writer")
            writer.release()
            continue
        
        frame_skip = original_fps / target_fps
        frame_count = 0
        frames_written = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % int(frame_skip) == 0:
                writer.write(frame)
                frames_written += 1
            
            frame_count += 1
        
        writer.release()
        
        # Check if file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"Success with {codec_name}!")
            print(f"Output: {output_path}")
            cap.release()
            return
        else:
            print(f"  {codec_name} failed")
            if os.path.exists(output_path):
                os.remove(output_path)  # Clean up failed file
    
    cap.release()
    print("All codecs failed. Try saving as .avi instead:")
    
    # Try .avi format as fallback
    avi_output = os.path.join(directory, f"{name}_{target_fps}fps.avi")
    print(f"Trying AVI format: {avi_output}")
    
    cap = cv2.VideoCapture(input_path)
    writer = cv2.VideoWriter(avi_output, cv2.VideoWriter_fourcc(*'XVID'), target_fps, (width, height))
    
    frame_skip = original_fps / target_fps
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % int(frame_skip) == 0:
            writer.write(frame)
        frame_count += 1
    
    cap.release()
    writer.release()
    
    if os.path.exists(avi_output) and os.path.getsize(avi_output) > 1000:
        print(f"Success with AVI format: {avi_output}")
    else:
        print("All formats failed")

# Try the robust version
convert_video_fps_robust("/Users/yejinbang/Documents/GitHub/sfx-project/data/test_videos/walk5.mp4", 30)
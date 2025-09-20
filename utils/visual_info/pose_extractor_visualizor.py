"""
Gait Pattern Analysis Script
Gait pattern visualization and analysis - Executable simple version
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings

# Hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pose_extractor import PoseExtractor

def extract_landmark_coordinates(landmarks_data, timestamps, landmark_name):
    """Extract time series data for specific landmark coordinates"""
    x_coords = []
    y_coords = []
    confidences = []
    valid_timestamps = []
    
    for i, frame_landmarks in enumerate(landmarks_data):
        if frame_landmarks and landmark_name in frame_landmarks:
            x, y, conf = frame_landmarks[landmark_name]
            if x is not None and y is not None:
                x_coords.append(x)
                y_coords.append(y)
                confidences.append(conf)
                valid_timestamps.append(timestamps[i])
            else:
                x_coords.append(np.nan)
                y_coords.append(np.nan)
                confidences.append(conf)
                valid_timestamps.append(timestamps[i])
        else:
            x_coords.append(np.nan)
            y_coords.append(np.nan)
            confidences.append(0.0)
            valid_timestamps.append(timestamps[i])
    
    return np.array(valid_timestamps), np.array(x_coords), np.array(y_coords), np.array(confidences)

def calculate_euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def plot_basic_trajectories(landmark_data, save_path=None):
    """Visualize basic gait trajectories"""
    plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Y-Coordinate Changes of Key Landmarks During Walking', fontsize=16)
    
    # Hip trajectory
    axes[0,0].plot(landmark_data['LEFT_HIP']['timestamps'], landmark_data['LEFT_HIP']['y'], 
                   'b-', label='Left Hip', linewidth=2)
    axes[0,0].plot(landmark_data['RIGHT_HIP']['timestamps'], landmark_data['RIGHT_HIP']['y'], 
                   'r-', label='Right Hip', linewidth=2)
    axes[0,0].set_title('Hip Height Changes')
    axes[0,0].set_ylabel('Y Coordinate (pixels)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Heel trajectory
    axes[0,1].plot(landmark_data['LEFT_HEEL']['timestamps'], landmark_data['LEFT_HEEL']['y'], 
                   'b-', label='Left Heel', linewidth=2)
    axes[0,1].plot(landmark_data['RIGHT_HEEL']['timestamps'], landmark_data['RIGHT_HEEL']['y'], 
                   'r-', label='Right Heel', linewidth=2)
    axes[0,1].set_title('Heel Height Changes')
    axes[0,1].set_ylabel('Y Coordinate (pixels)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Ankle trajectory
    axes[1,0].plot(landmark_data['LEFT_ANKLE']['timestamps'], landmark_data['LEFT_ANKLE']['y'], 
                   'b-', label='Left Ankle', linewidth=2)
    axes[1,0].plot(landmark_data['RIGHT_ANKLE']['timestamps'], landmark_data['RIGHT_ANKLE']['y'], 
                   'r-', label='Right Ankle', linewidth=2)
    axes[1,0].set_title('Ankle Height Changes')
    axes[1,0].set_xlabel('Time (seconds)')
    axes[1,0].set_ylabel('Y Coordinate (pixels)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Foot Index trajectory
    axes[1,1].plot(landmark_data['LEFT_FOOT_INDEX']['timestamps'], landmark_data['LEFT_FOOT_INDEX']['y'], 
                   'b-', label='Left Foot Index', linewidth=2)
    axes[1,1].plot(landmark_data['RIGHT_FOOT_INDEX']['timestamps'], landmark_data['RIGHT_FOOT_INDEX']['y'], 
                   'r-', label='Right Foot Index', linewidth=2)
    axes[1,1].set_title('Foot Index Height Changes')
    axes[1,1].set_xlabel('Time (seconds)')
    axes[1,1].set_ylabel('Y Coordinate (pixels)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Basic trajectory graph saved: {save_path}")
    
    plt.close()

def plot_hip_foot_distances(landmark_data, save_path=None):
    """Visualize Hip-Foot distances"""
    # Calculate Hip-Heel distances
    left_hip_heel_distance = calculate_euclidean_distance(
        landmark_data['LEFT_HIP']['x'], landmark_data['LEFT_HIP']['y'],
        landmark_data['LEFT_HEEL']['x'], landmark_data['LEFT_HEEL']['y']
    )
    
    right_hip_heel_distance = calculate_euclidean_distance(
        landmark_data['RIGHT_HIP']['x'], landmark_data['RIGHT_HIP']['y'],
        landmark_data['RIGHT_HEEL']['x'], landmark_data['RIGHT_HEEL']['y']
    )
    
    # Calculate Hip-Foot Index distances
    left_hip_foot_distance = calculate_euclidean_distance(
        landmark_data['LEFT_HIP']['x'], landmark_data['LEFT_HIP']['y'],
        landmark_data['LEFT_FOOT_INDEX']['x'], landmark_data['LEFT_FOOT_INDEX']['y']
    )
    
    right_hip_foot_distance = calculate_euclidean_distance(
        landmark_data['RIGHT_HIP']['x'], landmark_data['RIGHT_HIP']['y'],
        landmark_data['RIGHT_FOOT_INDEX']['x'], landmark_data['RIGHT_FOOT_INDEX']['y']
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Hip-Heel distances
    axes[0].plot(landmark_data['LEFT_HIP']['timestamps'], left_hip_heel_distance, 
                 'b-', label='Left Hip-Heel Distance', linewidth=2)
    axes[0].plot(landmark_data['RIGHT_HIP']['timestamps'], right_hip_heel_distance, 
                 'r-', label='Right Hip-Heel Distance', linewidth=2)
    axes[0].set_title('Hip-Heel Distance Changes (Basic Signal for Footstep Detection)', fontsize=14)
    axes[0].set_ylabel('Distance (pixels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Hip-Foot Index distances
    axes[1].plot(landmark_data['LEFT_HIP']['timestamps'], left_hip_foot_distance, 
                 'b-', label='Left Hip-Foot Distance', linewidth=2)
    axes[1].plot(landmark_data['RIGHT_HIP']['timestamps'], right_hip_foot_distance, 
                 'r-', label='Right Hip-Foot Distance', linewidth=2)
    axes[1].set_title('Hip-Foot Index Distance Changes', fontsize=14)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Distance (pixels)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hip-Foot distance graph saved: {save_path}")
    
    plt.close()
    
    return left_hip_heel_distance, right_hip_heel_distance, left_hip_foot_distance, right_hip_foot_distance

def plot_gait_pattern_comparison(landmark_data, left_hip_heel, right_hip_heel, save_path=None):
    """Visualize gait pattern comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    ax.plot(landmark_data['LEFT_HIP']['timestamps'], left_hip_heel, 
            'b-', label='Left Foot (Left Hip-Heel)', linewidth=3, alpha=0.8)
    ax.plot(landmark_data['RIGHT_HIP']['timestamps'], right_hip_heel, 
            'r-', label='Right Foot (Right Hip-Heel)', linewidth=3, alpha=0.8)
    
    ax.set_title('Left vs Right Foot Hip-Heel Distance Pattern Comparison\n(Peak = Heel Strike, Valley = Toe-off)', fontsize=14)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Hip-Heel Distance (pixels)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add pattern explanation text
    ax.text(0.02, 0.98, 'Peak (Maximum) = Heel Strike\nValley (Minimum) = Toe-off', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gait pattern comparison graph saved: {save_path}")
    
    plt.close()

def analyze_gait_statistics(left_hip_heel, right_hip_heel):
    """Analyze gait statistics"""
    print("=== Gait Pattern Statistics Analysis ===\n")
    
    # Remove NaN values for valid data analysis
    left_valid = left_hip_heel[~np.isnan(left_hip_heel)]
    right_valid = right_hip_heel[~np.isnan(right_hip_heel)]
    
    print(f"Left Foot Hip-Heel Distance:")
    print(f"  Mean: {np.mean(left_valid):.1f} pixels")
    print(f"  Standard Deviation: {np.std(left_valid):.1f} pixels")
    print(f"  Minimum: {np.min(left_valid):.1f} pixels")
    print(f"  Maximum: {np.max(left_valid):.1f} pixels")
    
    print(f"\nRight Foot Hip-Heel Distance:")
    print(f"  Mean: {np.mean(right_valid):.1f} pixels")
    print(f"  Standard Deviation: {np.std(right_valid):.1f} pixels")
    print(f"  Minimum: {np.min(right_valid):.1f} pixels")
    print(f"  Maximum: {np.max(right_valid):.1f} pixels")
    
    # Analyze left-right symmetry
    left_range = np.max(left_valid) - np.min(left_valid)
    right_range = np.max(right_valid) - np.min(right_valid)
    symmetry_ratio = min(left_range, right_range) / max(left_range, right_range)
    
    print(f"\nGait Symmetry:")
    print(f"  Left foot change range: {left_range:.1f} pixels")
    print(f"  Right foot change range: {right_range:.1f} pixels")
    print(f"  Symmetry ratio: {symmetry_ratio:.3f} (closer to 1.0 = more symmetric)")
    
    return symmetry_ratio

def get_next_version_dir(base_dir):
    """Generate next version directory path"""
    if not os.path.exists(base_dir):
        # First run, create v1
        version_dir = os.path.join(base_dir, "v1")
        return version_dir
    
    # Find existing versions
    existing_versions = []
    for item in os.listdir(base_dir):
        if item.startswith('v') and item[1:].isdigit():
            version_num = int(item[1:])
            existing_versions.append(version_num)
    
    if not existing_versions:
        # No v folders exist, create v1
        next_version = 1
    else:
        # Highest version + 1
        next_version = max(existing_versions) + 1
    
    version_dir = os.path.join(base_dir, f"v{next_version}")
    return version_dir

def main():
    """Main execution function"""
    print("=== Gait Visualization Analysis Started ===\n")
    
    # Set video path
    video_path = "/Users/yejinbang/Documents/GitHub/sfx-project/data/test_videos/walk4.mp4"
    base_output_dir = "/Users/yejinbang/Documents/GitHub/sfx-project/visual_info/output/gait_analysis"
    
    # Automatically create next version directory
    output_dir = get_next_version_dir(base_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results save path: {output_dir}")
    print()
    
    try:
        # 1. Extract pose data
        print("1. Extracting pose data...")
        extractor = PoseExtractor(target_fps=10, confidence_threshold=0.7)
        result = extractor.process_video(video_path, verbose=True)
        
        print(f"\n=== Extracted Data Information ===")
        print(f"Total frames: {len(result['landmarks'])}")
        print(f"Video duration: {result['timestamps'][-1]:.1f} seconds")
        print(f"Pose detection rate: {result['processing_stats']['pose_detection_rate']:.1f}%")
        
        # 2. Structure landmark data
        print("\n2. Structuring landmark data...")
        landmarks_to_analyze = ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_HEEL', 'RIGHT_HEEL', 
                               'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        
        landmark_data = {}
        for landmark in landmarks_to_analyze:
            t, x, y, conf = extract_landmark_coordinates(
                result['landmarks'], result['timestamps'], landmark
            )
            landmark_data[landmark] = {
                'timestamps': t,
                'x': x,
                'y': y,
                'confidence': conf
            }
        
        # 3. Basic trajectory visualization
        print("\n3. Creating basic trajectory visualization...")
        plot_basic_trajectories(landmark_data, os.path.join(output_dir, "basic_trajectories.png"))
        
        # 4. Hip-Foot distance visualization
        print("\n4. Creating Hip-Foot distance visualization...")
        left_hip_heel, right_hip_heel, left_hip_foot, right_hip_foot = plot_hip_foot_distances(
            landmark_data, os.path.join(output_dir, "hip_foot_distances.png")
        )
        
        # 5. Gait pattern comparison
        print("\n5. Creating gait pattern comparison visualization...")
        plot_gait_pattern_comparison(
            landmark_data, left_hip_heel, right_hip_heel, 
            os.path.join(output_dir, "gait_pattern_comparison.png")
        )
        
        # 6. Statistical analysis
        print("\n6. Performing statistical analysis...")
        symmetry_ratio = analyze_gait_statistics(left_hip_heel, right_hip_heel)
        
        # 7. Summary
        print("\n=== Visualization Analysis Complete ===\n")
        print("✅ Completed analyses:")
        print("  - Key landmark trajectory visualization")
        print("  - Hip-Foot distance calculation")
        print("  - Left vs right foot gait pattern comparison")
        print("  - Gait statistics analysis")
        
        print(f"\n📊 Key Results:")
        print(f"  - Pose detection success rate: {result['processing_stats']['pose_detection_rate']:.1f}%")
        print(f"  - Analyzed gait duration: {result['timestamps'][-1]:.1f} seconds")
        print(f"  - Gait symmetry: {symmetry_ratio:.3f}")
        
        print(f"\n📁 Saved Files:")
        print(f"  - {output_dir}/basic_trajectories.png")
        print(f"  - {output_dir}/hip_foot_distances.png")
        print(f"  - {output_dir}/gait_pattern_comparison.png")

        # Cleanup
        extractor.cleanup()
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
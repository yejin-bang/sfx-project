import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import seaborn as sns

# Add parent directory (utils) to Python path to import modules
current_dir = Path(__file__).parent  # visual_info/
parent_dir = current_dir.parent      # utils/
sys.path.insert(0, str(parent_dir))

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SignalVisualizer:
    """
    Visualize signal_processor.py results for footstep detection analysis
    
    Creates comprehensive plots showing:
    - Raw vs processed coordinates
    - Hip-foot distance signals
    - Signal quality metrics
    - Processing pipeline overview
    """
    
    def __init__(self, output_dir: str = "visualizations", dpi: int = 300):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualization files
            dpi: Resolution for saved images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        
        # Color scheme for consistent plots
        self.colors = {
            'raw': '#FF6B6B',           # Red for raw data
            'processed': '#4ECDC4',     # Teal for processed data
            'left': '#45B7D1',          # Blue for left side
            'right': '#96CEB4',         # Green for right side
            'missing': '#FFA07A',       # Light salmon for missing data
            'confidence': '#DDA0DD',    # Plum for confidence
            'hip': '#FFD93D',           # Yellow for hip
            'heel': '#6BCF7F'           # Light green for heel
        }
    
    def calculate_hip_foot_distances(self, processed_data: Dict) -> Dict[str, np.ndarray]:
        """
        Calculate hip-foot distances from processed coordinate data
        
        Args:
            processed_data: Output from signal_processor.process_coordinates()
            
        Returns:
            Dictionary with left and right hip-foot distances
        """
        distances = {'left': [], 'right': []}
        
        # Extract coordinates
        left_hip = processed_data['clean_coordinates']['primary']['LEFT_HIP']
        left_heel = processed_data['clean_coordinates']['primary']['LEFT_HEEL']
        right_hip = processed_data['clean_coordinates']['primary']['RIGHT_HIP']
        right_heel = processed_data['clean_coordinates']['primary']['RIGHT_HEEL']
        
        # Calculate distances frame by frame
        for i in range(len(left_hip['x'])):
            # Left side distance
            left_dist = np.sqrt(
                (left_hip['x'][i] - left_heel['x'][i])**2 + 
                (left_hip['y'][i] - left_heel['y'][i])**2
            )
            distances['left'].append(left_dist)
            
            # Right side distance
            right_dist = np.sqrt(
                (right_hip['x'][i] - right_heel['x'][i])**2 + 
                (right_hip['y'][i] - right_heel['y'][i])**2
            )
            distances['right'].append(right_dist)
        
        return {
            'left': np.array(distances['left']),
            'right': np.array(distances['right'])
        }
    
    def plot_coordinate_comparison(self, processed_data: Dict, 
                                 landmark_name: str = 'LEFT_HEEL',
                                 save_name: str = "coordinate_comparison.png") -> str:
        """
        Plot raw vs processed coordinates for a specific landmark
        
        Args:
            processed_data: Output from signal_processor.process_coordinates()
            landmark_name: Which landmark to visualize
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{landmark_name} Coordinate Processing Pipeline', fontsize=16, fontweight='bold')
        
        timestamps = processed_data['timestamps']
        
        # Get raw and processed data
        raw_coords = processed_data['raw_coordinates'][landmark_name]
        clean_coords = processed_data['clean_coordinates']['primary'][landmark_name]
        
        # Plot X coordinates
        axes[0, 0].plot(timestamps, raw_coords['x'], 'o-', color=self.colors['raw'], 
                       alpha=0.7, markersize=3, label='Raw X')
        axes[0, 0].plot(timestamps, clean_coords['x'], '-', color=self.colors['processed'], 
                       linewidth=2, label='Processed X')
        axes[0, 0].set_title('X Coordinate Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('X Position (pixels)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Y coordinates
        axes[0, 1].plot(timestamps, raw_coords['y'], 'o-', color=self.colors['raw'], 
                       alpha=0.7, markersize=3, label='Raw Y')
        axes[0, 1].plot(timestamps, clean_coords['y'], '-', color=self.colors['processed'], 
                       linewidth=2, label='Processed Y')
        axes[0, 1].set_title('Y Coordinate Over Time')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Y Position (pixels)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot confidence scores
        axes[1, 0].plot(timestamps, raw_coords['confidence'], '-', color=self.colors['confidence'], 
                       linewidth=2, label='Confidence')
        axes[1, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Threshold (0.7)')
        axes[1, 0].set_title('Detection Confidence Over Time')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot trajectory (X vs Y)
        axes[1, 1].plot(raw_coords['x'], raw_coords['y'], 'o-', color=self.colors['raw'], 
                       alpha=0.7, markersize=3, label='Raw Trajectory')
        axes[1, 1].plot(clean_coords['x'], clean_coords['y'], '-', color=self.colors['processed'], 
                       linewidth=2, label='Processed Trajectory')
        axes[1, 1].set_title('2D Trajectory (X vs Y)')
        axes[1, 1].set_xlabel('X Position (pixels)')
        axes[1, 1].set_ylabel('Y Position (pixels)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_yaxis()  # Invert Y axis for image coordinates
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_hip_foot_distances(self, processed_data: Dict, 
                              save_name: str = "hip_foot_distances.png") -> str:
        """
        Plot hip-foot distance signals for both legs
        
        Args:
            processed_data: Output from signal_processor.process_coordinates()
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Hip-Foot Distance Analysis', fontsize=16, fontweight='bold')
        
        timestamps = processed_data['timestamps']
        distances = self.calculate_hip_foot_distances(processed_data)
        
        # Plot individual leg distances
        axes[0].plot(timestamps, distances['left'], '-', color=self.colors['left'], 
                    linewidth=2, label='Left Leg')
        axes[0].set_title('Left Hip-Heel Distance Over Time')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Distance (pixels)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(timestamps, distances['right'], '-', color=self.colors['right'], 
                    linewidth=2, label='Right Leg')
        axes[1].set_title('Right Hip-Heel Distance Over Time')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Distance (pixels)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot both legs together
        axes[2].plot(timestamps, distances['left'], '-', color=self.colors['left'], 
                    linewidth=2, label='Left Leg', alpha=0.8)
        axes[2].plot(timestamps, distances['right'], '-', color=self.colors['right'], 
                    linewidth=2, label='Right Leg', alpha=0.8)
        axes[2].set_title('Both Legs Hip-Heel Distance Comparison')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Distance (pixels)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_signal_quality_metrics(self, processed_data: Dict, 
                                   save_name: str = "signal_quality.png") -> str:
        """
        Plot signal quality metrics and processing statistics
        
        Args:
            processed_data: Output from signal_processor.process_coordinates()
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Signal Quality and Processing Metrics', fontsize=16, fontweight='bold')
        
        metadata = processed_data['metadata']
        
        # Detection rate by landmark
        landmarks = ['LEFT_HIP', 'LEFT_HEEL', 'RIGHT_HIP', 'RIGHT_HEEL']
        detection_rates = []
        avg_confidences = []
        
        for landmark in landmarks:
            confidence_scores = processed_data['raw_coordinates'][landmark]['confidence']
            valid_detections = np.sum(confidence_scores >= 0.7)
            total_frames = len(confidence_scores)
            
            detection_rate = (valid_detections / total_frames * 100) if total_frames > 0 else 0
            avg_confidence = np.mean(confidence_scores[confidence_scores > 0]) if np.any(confidence_scores > 0) else 0
            
            detection_rates.append(detection_rate)
            avg_confidences.append(avg_confidence)
        
        # Bar plot of detection rates
        bars1 = axes[0, 0].bar(landmarks, detection_rates, color=[self.colors['hip'], self.colors['heel'], 
                                                                 self.colors['hip'], self.colors['heel']], alpha=0.7)
        axes[0, 0].set_title('Detection Rate by Landmark')
        axes[0, 0].set_ylabel('Detection Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, detection_rates):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # Bar plot of average confidence
        bars2 = axes[0, 1].bar(landmarks, avg_confidences, color=[self.colors['hip'], self.colors['heel'], 
                                                                 self.colors['hip'], self.colors['heel']], alpha=0.7)
        axes[0, 1].set_title('Average Confidence by Landmark')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, conf in zip(bars2, avg_confidences):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{conf:.3f}', ha='center', va='bottom')
        
        # Processing statistics pie chart
        stats_labels = ['Frames with Data', 'Missing Frames']
        stats_values = [metadata['frames_with_data'], 
                       metadata['total_frames'] - metadata['frames_with_data']]
        colors_pie = [self.colors['processed'], self.colors['missing']]
        
        axes[1, 0].pie(stats_values, labels=stats_labels, colors=colors_pie, autopct='%1.1f%%')
        axes[1, 0].set_title('Data Coverage Overview')
        
        # Processing metadata table
        axes[1, 1].axis('off')
        metadata_text = f"""
Processing Metadata:
━━━━━━━━━━━━━━━━━━━━
Sampling Rate: {metadata['sampling_rate']:.2f} fps
Data Coverage: {metadata['data_coverage']:.1%}
Total Frames: {metadata['total_frames']:,}
Frames with Data: {metadata['frames_with_data']:,}

Filter Configuration:
━━━━━━━━━━━━━━━━━━━━
Cutoff Frequency: {metadata['config'].cutoff_frequency}
Filter Order: {metadata['config'].filter_order}
Confidence Threshold: {metadata['config'].confidence_threshold}
Min Data Threshold: {metadata['config'].min_data_threshold:.1%}
        """
        
        axes[1, 1].text(0.1, 0.9, metadata_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        axes[1, 1].set_title('Processing Configuration')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_processing_pipeline_overview(self, landmarks_data: List[Dict], 
                                        processed_data: Dict,
                                        save_name: str = "pipeline_overview.png") -> str:
        """
        Create comprehensive overview of the entire processing pipeline
        
        Args:
            landmarks_data: Raw landmark data from pose_extractor
            processed_data: Processed data from signal_processor
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Complete Signal Processing Pipeline Overview', fontsize=18, fontweight='bold')
        
        timestamps = processed_data['timestamps']
        
        # 1. Raw landmark trajectory (LEFT_HEEL)
        raw_coords = processed_data['raw_coordinates']['LEFT_HEEL']
        clean_coords = processed_data['clean_coordinates']['primary']['LEFT_HEEL']
        
        axes[0, 0].plot(raw_coords['x'], raw_coords['y'], 'o-', color=self.colors['raw'], 
                       alpha=0.6, markersize=2, label='Raw')
        axes[0, 0].plot(clean_coords['x'], clean_coords['y'], '-', color=self.colors['processed'], 
                       linewidth=2, label='Processed')
        axes[0, 0].set_title('LEFT_HEEL Trajectory\n(Raw vs Processed)')
        axes[0, 0].set_xlabel('X Position (pixels)')
        axes[0, 0].set_ylabel('Y Position (pixels)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_yaxis()
        
        # 2. Hip-foot distances
        distances = self.calculate_hip_foot_distances(processed_data)
        axes[0, 1].plot(timestamps, distances['left'], '-', color=self.colors['left'], 
                       linewidth=2, label='Left')
        axes[0, 1].plot(timestamps, distances['right'], '-', color=self.colors['right'], 
                       linewidth=2, label='Right')
        axes[0, 1].set_title('Hip-Foot Distances\n(Footstep Signal)')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Distance (pixels)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence over time
        axes[0, 2].plot(timestamps, raw_coords['confidence'], '-', color=self.colors['confidence'], 
                       linewidth=2)
        axes[0, 2].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Threshold')
        axes[0, 2].fill_between(timestamps, 0, raw_coords['confidence'], 
                               where=(raw_coords['confidence'] >= 0.7), 
                               color=self.colors['processed'], alpha=0.3, label='Valid')
        axes[0, 2].fill_between(timestamps, 0, raw_coords['confidence'], 
                               where=(raw_coords['confidence'] < 0.7), 
                               color=self.colors['missing'], alpha=0.3, label='Low Confidence')
        axes[0, 2].set_title('Detection Confidence\n(LEFT_HEEL)')
        axes[0, 2].set_xlabel('Time (seconds)')
        axes[0, 2].set_ylabel('Confidence Score')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. All landmarks detection rates
        landmarks = ['LEFT_HIP', 'LEFT_HEEL', 'RIGHT_HIP', 'RIGHT_HEEL']
        detection_rates = []
        
        for landmark in landmarks:
            confidence_scores = processed_data['raw_coordinates'][landmark]['confidence']
            valid_detections = np.sum(confidence_scores >= 0.7)
            total_frames = len(confidence_scores)
            detection_rate = (valid_detections / total_frames * 100) if total_frames > 0 else 0
            detection_rates.append(detection_rate)
        
        colors_bar = [self.colors['hip'] if 'HIP' in lm else self.colors['heel'] for lm in landmarks]
        bars = axes[1, 0].bar(landmarks, detection_rates, color=colors_bar, alpha=0.7)
        axes[1, 0].set_title('Detection Success Rate\nby Landmark')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, rate in zip(bars, detection_rates):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # 5. Signal smoothing effect (before/after filtering)
        sample_start = len(timestamps) // 4
        sample_end = sample_start + min(100, len(timestamps) // 2)
        sample_times = timestamps[sample_start:sample_end]
        sample_raw = distances['left'][sample_start:sample_end]
        
        # Simulate raw noisy signal (add some noise to show filtering effect)
        np.random.seed(42)
        noisy_signal = sample_raw + np.random.normal(0, 2, len(sample_raw))
        
        axes[1, 1].plot(sample_times, noisy_signal, '-', color=self.colors['raw'], 
                       alpha=0.7, linewidth=1, label='Before Filtering')
        axes[1, 1].plot(sample_times, sample_raw, '-', color=self.colors['processed'], 
                       linewidth=2, label='After Filtering')
        axes[1, 1].set_title('Signal Smoothing Effect\n(Sample Section)')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Distance (pixels)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Processing summary
        axes[1, 2].axis('off')
        metadata = processed_data['metadata']
        
        summary_text = f"""
🎯 PROCESSING SUMMARY
{'='*40}

📊 Data Quality:
   • Data Coverage: {metadata['data_coverage']:.1%}
   • Sampling Rate: {metadata['sampling_rate']:.2f} fps
   • Total Duration: {timestamps[-1]:.1f} seconds

🔧 Filter Settings:
   • Cutoff Frequency: {metadata['config'].cutoff_frequency}
   • Filter Order: {metadata['config'].filter_order}
   • Confidence Threshold: {metadata['config'].confidence_threshold}

📈 Results:
   • Processed Frames: {metadata['frames_with_data']:,}
   • Missing Frames: {metadata['total_frames'] - metadata['frames_with_data']:,}
   • Success Rate: {metadata['data_coverage']*100:.1f}%

🚶 Gait Analysis Ready:
   ✓ Coordinates interpolated
   ✓ Noise filtered out  
   ✓ Hip-foot distances calculated
   ✓ Ready for footstep detection
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.2))
        axes[1, 2].set_title('Processing Summary')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_all_visualizations(self, landmarks_data: List[Dict], 
                                processed_data: Dict,
                                video_name: str = "video") -> Dict[str, str]:
        """
        Create all visualization plots and save them
        
        Args:
            landmarks_data: Raw landmark data from pose_extractor
            processed_data: Processed data from signal_processor
            video_name: Name for the video (used in filenames)
            
        Returns:
            Dictionary with plot names and their file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🎨 Creating visualizations for {video_name}...")
        print(f"📁 Output directory: {self.output_dir}")
        
        plots = {}
        
        # 1. Coordinate comparison
        print("   📊 Creating coordinate comparison plots...")
        for landmark in ['LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_HIP', 'RIGHT_HIP']:
            plot_name = f"coordinates_{landmark.lower()}_{video_name}_{timestamp}.png"
            plots[f'coordinates_{landmark}'] = self.plot_coordinate_comparison(
                processed_data, landmark, plot_name
            )
        
        # 2. Hip-foot distances
        print("   📈 Creating hip-foot distance plots...")
        distance_name = f"distances_{video_name}_{timestamp}.png"
        plots['distances'] = self.plot_hip_foot_distances(processed_data, distance_name)
        
        # 3. Signal quality metrics
        print("   📋 Creating signal quality plots...")
        quality_name = f"quality_{video_name}_{timestamp}.png"
        plots['quality'] = self.plot_signal_quality_metrics(processed_data, quality_name)
        
        # 4. Pipeline overview
        print("   🔄 Creating pipeline overview...")
        overview_name = f"overview_{video_name}_{timestamp}.png"
        plots['overview'] = self.plot_processing_pipeline_overview(
            landmarks_data, processed_data, overview_name
        )
        
        print(f"✅ All visualizations created successfully!")
        print(f"📁 Files saved in: {self.output_dir.absolute()}")
        
        return plots


# Example usage and testing
if __name__ == "__main__":
    print("🎨 Signal Processor Visualizer")
    print("=" * 50)
    
    try:
        from pose_extractor import PoseExtractor
        from signal_processor import SignalProcessor, SignalProcessorConfig
        
        print("📹 Testing with real video data...")
        
        # Process a test video
        video_path = "./test_videos/walk4.mp4"
        
        if not os.path.exists(video_path):
            print(f"❌ ERROR: Video file not found!")
            print(f"   Expected path: {video_path}")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Please check if the video file exists or update the path.")
            exit(1)
        
        print(f"✅ Found video file: {video_path}")
        
        # Extract landmarks
        print("🔍 Extracting pose landmarks...")
        extractor = PoseExtractor()
        landmarks_data, timestamps = extractor.get_signal_processor_input(video_path)
        
        # Process signals
        print("⚙️ Processing signals...")
        config = SignalProcessorConfig()
        processor = SignalProcessor(config)
        processed_data = processor.process_coordinates(landmarks_data, timestamps)
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        visualizer = SignalVisualizer()
        plots = visualizer.create_all_visualizations(
            landmarks_data, processed_data, "walk4"
        )
        
        print(f"\n📊 Successfully generated plots:")
        for plot_type, file_path in plots.items():
            print(f"   {plot_type}: {file_path}")
        
        print(f"\n🎯 Visualization completed successfully!")
        print(f"   All plots saved in: {Path('visualizations').absolute()}")
            
    except ImportError as e:
        print(f"❌ ERROR: Required modules not found!")
        print(f"   Missing module: {e}")
        print(f"   Please ensure pose_extractor.py and signal_processor.py are available.")
        exit(1)
        
    except Exception as e:
        print(f"❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
                
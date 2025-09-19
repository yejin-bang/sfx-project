import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import pipeline modules
from utils.pose_extractor import PoseExtractor
from utils.signal_processor import SignalProcessor, SignalProcessorConfig
from utils.gait_detector import GaitDetector, GaitDetectorConfig
from video_validator import VideoValidator


@dataclass
class FootstepDetectorConfig:
    """Configuration for footstep detection and evaluation"""
    # Matching parameters
    matching_tolerance: float = 0.3  # ±0.3 seconds for ground truth matching
    
    # Pipeline configurations
    pose_extractor_fps: int = 10
    pose_confidence_threshold: float = 0.7
    signal_cutoff_frequency: float = 0.1752
    gait_peak_distance: int = 5
    
    # Visualization parameters
    figure_size: Tuple[int, int] = (8, 6)
    dpi: int = 80
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.matching_tolerance <= 0:
            raise ValueError("matching_tolerance must be positive")


class FootstepDetector:
    """
    Complete footstep detection pipeline with ground truth comparison
    
    Updated Pipeline:
    1. PoseExtractor: Extract landmarks from video at 10fps
    2. SignalProcessor: Clean and filter coordinate time series
    3. GaitDetector: Detect heel strikes and return FRAME INDICES
    4. FootstepDetector: Convert frame indices to timestamps + evaluation
    5. Visualization: Create comparison graphs and save results
    
    Clear Separation of Concerns:
    - gait_detector.py: Pure detection (frame indices only)
    - footstep_detector.py: Timestamp conversion + evaluation + visualization
    """
    
    def __init__(self, config: Optional[FootstepDetectorConfig] = None):
        """
        Initialize footstep detector with pipeline components
        
        Args:
            config: FootstepDetectorConfig object. If None, uses default values.
        """
        self.config = config if config is not None else FootstepDetectorConfig()
        
        # Initialize pipeline components
        self.pose_extractor = PoseExtractor(
            target_fps=self.config.pose_extractor_fps,
            confidence_threshold=self.config.pose_confidence_threshold
        )
        
        signal_config = SignalProcessorConfig(
            cutoff_frequency=self.config.signal_cutoff_frequency,
            confidence_threshold=self.config.pose_confidence_threshold
        )
        self.signal_processor = SignalProcessor(signal_config)
        
        gait_config = GaitDetectorConfig(
            peak_distance=self.config.gait_peak_distance
        )
        self.gait_detector = GaitDetector(gait_config)
        
        # Storage for results
        self.last_detection_results = None
        self.last_ground_truth = None
        self.last_comparison_results = None
    
    def convert_frame_indices_to_timestamps(self, frame_indices: List[int], 
                                          timestamps: np.ndarray, 
                                          original_fps: float) -> List[float]:
        """
        Convert frame indices to timestamps
        
        Args:
            frame_indices: List of frame indices from gait_detector
            timestamps: Timestamp array from signal_processor
            original_fps: Original video FPS
            
        Returns:
            List of timestamps corresponding to frame indices
        """
        if len(frame_indices) == 0:
            return []
        
        # Method 1: Use processed timestamps directly (more accurate for actual processed frames)
        frame_timestamps = []
        for frame_idx in frame_indices:
            if 0 <= frame_idx < len(timestamps):
                frame_timestamps.append(float(timestamps[frame_idx]))
            else:
                # Fallback: calculate from original FPS
                timestamp = frame_idx / original_fps
                frame_timestamps.append(timestamp)
                warnings.warn(f"Frame index {frame_idx} out of bounds, using FPS calculation")
        
        return sorted(frame_timestamps)
    
    def load_ground_truth(self, json_path: str) -> Dict[str, Any]:
        """
        Load ground truth annotations from JSON file
        
        Args:
            json_path: Path to ground truth JSON file
            
        Returns:
            Dictionary containing ground truth data
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            ground_truth_data = json.load(f)
        
        # Extract timestamps and foot information
        gt_timestamps = []
        gt_feet = []
        gt_frames = []
        
        for annotation in ground_truth_data['annotations']:
            gt_timestamps.append(annotation['timestamp'])
            gt_feet.append(annotation['foot'])
            gt_frames.append(annotation['frame'])
        
        processed_gt = {
            'timestamps': np.array(gt_timestamps),
            'feet': gt_feet,
            'frames': np.array(gt_frames),
            'total_steps': ground_truth_data['summary']['total_steps'],
            'left_steps': ground_truth_data['summary']['left_steps'],
            'right_steps': ground_truth_data['summary']['right_steps'],
            'video_info': ground_truth_data['video_info'],
            'video_path': ground_truth_data['video_path']
        }
        
        return processed_gt
    
    def process_video(self, video_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run complete footstep detection pipeline on video
        
        Args:
            video_path: Path to video file
            verbose: Print processing information
            
        Returns:
            Dictionary containing detection results and metadata
        """
        if verbose:
            print("=" * 60)
            print("FOOTSTEP DETECTION PIPELINE")
            print("=" * 60)
        
        # Step 1: Extract pose landmarks
        if verbose:
            print("\n🏃 Step 1: Extracting pose landmarks...")
        
        landmarks_data, timestamps = self.pose_extractor.get_signal_processor_input(
            video_path, verbose=verbose
        )
        
        # Get original video FPS for timestamp conversion
        video_info = self.pose_extractor.video_validator.validate_video(video_path)
        original_fps = video_info['fps']
        
        # Step 2: Process coordinate signals
        if verbose:
            print("\n📊 Step 2: Processing coordinate signals...")
        
        processed_data = self.signal_processor.process_coordinates(
            landmarks_data, timestamps, verbose=verbose
        )
        
        # Step 3: Detect gait events (frame indices only)
        if verbose:
            print("\n👣 Step 3: Detecting gait events...")
        
        gait_results = self.gait_detector.process_gait_events(
            processed_data, verbose=verbose
        )
        
        # Step 4: Convert frame indices to timestamps (NEW RESPONSIBILITY)
        if verbose:
            print("\n⏰ Step 4: Converting frame indices to timestamps...")
        
        frame_indices = gait_results['heel_strike_frame_indices']
        detected_timestamps = self.convert_frame_indices_to_timestamps(
            frame_indices, processed_data['timestamps'], original_fps
        )
        
        if verbose:
            print(f"   Converted {len(frame_indices)} frame indices to timestamps")
            if len(detected_timestamps) > 0:
                print(f"   Timestamp range: {min(detected_timestamps):.3f}s to {max(detected_timestamps):.3f}s")
        
        # Compile final results
        detection_results = {
            'detected_timestamps': detected_timestamps,  # 🎯 Final timestamps
            'detected_frame_indices': frame_indices,     # 🎯 Original frame indices
            'total_detections': len(detected_timestamps),
            'gait_results': gait_results,
            'processed_data': processed_data,
            'landmarks_data': landmarks_data,
            'video_info': video_info,
            'pipeline_metadata': {
                'video_path': video_path,
                'original_fps': original_fps,
                'processed_fps': processed_data['metadata']['sampling_rate'],
                'pose_extractor_config': {
                    'target_fps': self.config.pose_extractor_fps,
                    'confidence_threshold': self.config.pose_confidence_threshold
                },
                'signal_processor_config': processed_data['metadata']['config'],
                'gait_detector_config': gait_results['metadata']['processing_config']
            }
        }
        
        if verbose:
            print(f"\n✅ Pipeline complete!")
            print(f"   Detected {detection_results['total_detections']} footsteps")
            print(f"   Original video FPS: {original_fps:.2f}")
            print(f"   Processing FPS: {processed_data['metadata']['sampling_rate']:.2f}")
        
        self.last_detection_results = detection_results
        return detection_results
    
    def compare_with_ground_truth(self, detected_timestamps: List[float], 
                                ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare detected footsteps with ground truth annotations
        
        Args:
            detected_timestamps: List of detected heel strike timestamps
            ground_truth: Ground truth data from load_ground_truth()
            
        Returns:
            Dictionary containing comparison results and metrics
        """
        gt_timestamps = ground_truth['timestamps']
        tolerance = self.config.matching_tolerance
        
        # Find matches between detected and ground truth
        matches = []
        matched_gt_indices = set()
        matched_detected_indices = set()
        
        for i, detected_time in enumerate(detected_timestamps):
            for j, gt_time in enumerate(gt_timestamps):
                if j in matched_gt_indices:
                    continue
                
                time_diff = abs(detected_time - gt_time)
                if time_diff <= tolerance:
                    matches.append({
                        'detected_index': i,
                        'detected_timestamp': detected_time,
                        'gt_index': j,
                        'gt_timestamp': gt_time,
                        'time_error': detected_time - gt_time,
                        'abs_time_error': time_diff,
                        'gt_foot': ground_truth['feet'][j],
                        'gt_frame': ground_truth['frames'][j]
                    })
                    matched_gt_indices.add(j)
                    matched_detected_indices.add(i)
                    break
        
        # Calculate metrics
        metrics = self.calculate_metrics(matches, detected_timestamps, ground_truth)
        
        # Identify false positives and missed detections
        false_positives = [
            {'index': i, 'timestamp': ts} 
            for i, ts in enumerate(detected_timestamps) 
            if i not in matched_detected_indices
        ]
        
        missed_detections = [
            {'index': j, 'timestamp': gt_timestamps[j], 'foot': ground_truth['feet'][j]}
            for j in range(len(gt_timestamps))
            if j not in matched_gt_indices
        ]
        
        comparison_results = {
            'matches': matches,
            'false_positives': false_positives,
            'missed_detections': missed_detections,
            'metrics': metrics,
            'matching_tolerance': tolerance,
            'total_detected': len(detected_timestamps),
            'total_ground_truth': len(gt_timestamps)
        }
        
        return comparison_results
    
    def calculate_metrics(self, matches: List[Dict], 
                         detected_timestamps: List[float],
                         ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics from matching results
        
        Args:
            matches: List of matched detections
            detected_timestamps: All detected timestamps
            ground_truth: Ground truth data
            
        Returns:
            Dictionary containing calculated metrics
        """
        n_matches = len(matches)
        n_detected = len(detected_timestamps)
        n_ground_truth = len(ground_truth['timestamps'])
        
        # Basic metrics
        detection_rate = (n_matches / n_ground_truth * 100) if n_ground_truth > 0 else 0
        precision = (n_matches / n_detected * 100) if n_detected > 0 else 0
        false_positive_rate = ((n_detected - n_matches) / n_detected * 100) if n_detected > 0 else 0
        
        # Time accuracy metrics
        if n_matches > 0:
            time_errors = [match['time_error'] for match in matches]
            abs_time_errors = [match['abs_time_error'] for match in matches]
            
            mean_time_error = np.mean(time_errors)
            std_time_error = np.std(time_errors)
            mean_abs_error = np.mean(abs_time_errors)
            max_abs_error = np.max(abs_time_errors)
        else:
            mean_time_error = 0
            std_time_error = 0
            mean_abs_error = 0
            max_abs_error = 0
        
        # F1 Score
        if precision + detection_rate > 0:
            f1_score = 2 * (precision * detection_rate / 100) / (precision + detection_rate) * 100
        else:
            f1_score = 0
        
        metrics = {
            'detection_rate': detection_rate,
            'precision': precision,
            'false_positive_rate': false_positive_rate,
            'f1_score': f1_score,
            'mean_time_error': mean_time_error,
            'std_time_error': std_time_error,
            'mean_abs_error': mean_abs_error,
            'max_abs_error': max_abs_error,
            'total_matches': n_matches,
            'total_detected': n_detected,
            'total_ground_truth': n_ground_truth
        }
        
        return metrics
    
    def generate_visualizations(self, comparison_results: Dict[str, Any], 
                              ground_truth: Dict[str, Any],
                              detected_timestamps: List[float],
                              save_path: str = None) -> None:
        """
        Generate comparison visualizations and save to file
        
        Args:
            comparison_results: Results from compare_with_ground_truth()
            ground_truth: Ground truth data
            detected_timestamps: All detected timestamps
            save_path: Path to save visualization (optional)
        """
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                           dpi=self.config.dpi)
        
        # Timeline comparison plot
        self._plot_timeline_comparison(ax1, comparison_results, ground_truth, detected_timestamps)
        
        # Metrics summary plot
        self._plot_metrics_summary(ax2, comparison_results['metrics'])
        
        # Distance signals and peak detection plot
        self._plot_distance_signals_and_peaks(ax3, ground_truth)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"📊 Visualization saved: {save_path}")
        
        plt.show()
    
    def _plot_timeline_comparison(self, ax, comparison_results: Dict, 
                                ground_truth: Dict, detected_timestamps: List[float]) -> None:
        """Create timeline comparison plot"""
        # Plot ground truth
        gt_timestamps = ground_truth['timestamps']
        gt_feet = ground_truth['feet']
        
        for i, (timestamp, foot) in enumerate(zip(gt_timestamps, gt_feet)):
            color = 'blue' if foot == 'left' else 'red'
            ax.scatter(timestamp, 1, c=color, s=100, marker='o', 
                      label='GT Left' if foot == 'left' and i == 0 else 'GT Right' if foot == 'right' and i == 0 else "")
        
        # Plot detected footsteps
        for timestamp in detected_timestamps:
            ax.scatter(timestamp, 0.5, c='green', s=60, marker='^', alpha=0.7)
        
        # Plot matches with connecting lines
        for match in comparison_results['matches']:
            detected_time = match['detected_timestamp']
            gt_time = match['gt_timestamp']
            ax.plot([detected_time, gt_time], [0.5, 1], 'gray', alpha=0.5, linewidth=1)
        
        # Highlight false positives and missed detections
        for fp in comparison_results['false_positives']:
            ax.scatter(fp['timestamp'], 0.5, c='orange', s=80, marker='x', linewidth=3)
        
        for md in comparison_results['missed_detections']:
            ax.scatter(md['timestamp'], 1, facecolor='none', edgecolor='black', 
                      s=120, marker='o', linewidth=2)
        
        # Customize timeline plot
        ax.set_ylim(-0.1, 1.5)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Detection Level', fontsize=12)
        ax.set_title('Footstep Detection Timeline Comparison', fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='GT Left'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='GT Right'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Detected'),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='orange', markersize=8, label='False Positive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                      markeredgecolor='black', markersize=10, label='Missed')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add tolerance band visualization
        tolerance = comparison_results['matching_tolerance']
        ax.text(0.02, 0.95, f'Matching Tolerance: ±{tolerance}s', 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    def _plot_metrics_summary(self, ax, metrics: Dict[str, float]) -> None:
        """Create metrics summary bar plot"""
        metric_names = ['Detection\nRate (%)', 'Precision\n(%)', 'F1 Score\n(%)', 'False Positive\nRate (%)']
        metric_values = [
            metrics['detection_rate'],
            metrics['precision'], 
            metrics['f1_score'],
            metrics['false_positive_rate']
        ]
        
        colors = ['#2E8B57', '#4169E1', '#9932CC', '#DC143C']  # Different colors for each metric
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 110)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add metrics text box
        metrics_text = f"""
Time Accuracy:
• Mean Error: {metrics['mean_abs_error']:.3f}s
• Max Error: {metrics['max_abs_error']:.3f}s
• Std Error: {metrics['std_time_error']:.3f}s

Detection Summary:
• Matches: {metrics['total_matches']}/{metrics['total_ground_truth']}
• Total Detected: {metrics['total_detected']}
        """.strip()
        
        ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def _plot_distance_signals_and_peaks(self, ax, ground_truth: Dict[str, Any]) -> None:
        """Create distance signals and peak detection visualization"""
        if not self.last_detection_results:
            ax.text(0.5, 0.5, 'No detection results available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get data from last detection results
        gait_results = self.last_detection_results['gait_results']
        distance_signals = gait_results['distance_signals']
        detailed_results = gait_results['detailed_results']
        timestamps = self.last_detection_results['processed_data']['timestamps']
        
        # Color scheme for signals
        colors = {
            'left_hip_heel': '#2E8B57',    # Sea Green
            'right_hip_heel': '#DC143C',   # Crimson
            'left_hip_ankle': '#4169E1',   # Royal Blue
            'right_hip_ankle': '#FF8C00'   # Dark Orange
        }
        
        # Line styles
        line_styles = {
            'left_hip_heel': '-',    # Solid (Primary)
            'right_hip_heel': '-',   # Solid (Primary)
            'left_hip_ankle': '--',  # Dashed (Secondary)
            'right_hip_ankle': '--'  # Dashed (Secondary)
        }
        
        # Plot distance signals
        signal_legends = []
        for signal_name, distances in distance_signals.items():
            if distances is not None:
                color = colors.get(signal_name, 'gray')
                line_style = line_styles.get(signal_name, '-')
                
                # Plot distance signal
                line = ax.plot(timestamps, distances, 
                              color=color, linestyle=line_style, 
                              linewidth=2, alpha=0.8,
                              label=signal_name.replace('_', ' ').title())[0]
                signal_legends.append(line)
                
                # Plot detected peaks
                if signal_name in detailed_results:
                    peak_indices = detailed_results[signal_name]['peak_frame_indices']
                    if peak_indices:
                        peak_timestamps = timestamps[peak_indices]
                        peak_values = distances[peak_indices]
                        
                        ax.scatter(peak_timestamps, peak_values, 
                                  color=color, s=80, marker='o', 
                                  edgecolor='white', linewidth=2,
                                  zorder=5, alpha=0.9)
        
        # Plot ground truth as vertical lines
        gt_timestamps = ground_truth['timestamps']
        gt_feet = ground_truth['feet']
        
        y_min, y_max = ax.get_ylim()
        for timestamp, foot in zip(gt_timestamps, gt_feet):
            line_color = 'blue' if foot == 'left' else 'red'
            ax.axvline(x=timestamp, color=line_color, linestyle=':', 
                      linewidth=2, alpha=0.7, zorder=3)
        
        # Customize plot
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Hip-Foot Distance (normalized)', fontsize=12)
        ax.set_title('Distance Signals and Peak Detection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Create comprehensive legend
        # Distance signals legend
        legend1 = ax.legend(handles=signal_legends, 
                           loc='upper left', fontsize=10,
                           title='Distance Signals')
        
        # Ground truth and peaks legend
        from matplotlib.lines import Line2D
        custom_elements = [
            Line2D([0], [0], color='blue', linestyle=':', linewidth=2, 
                   label='GT Left Foot'),
            Line2D([0], [0], color='red', linestyle=':', linewidth=2, 
                   label='GT Right Foot'),
            Line2D([0], [0], marker='o', color='gray', linestyle='None',
                   markersize=8, markeredgecolor='white', markeredgewidth=2,
                   label='Detected Peaks')
        ]
        
        legend2 = ax.legend(handles=custom_elements, 
                           loc='upper right', fontsize=10,
                           title='Ground Truth & Peaks')
        
        # Add both legends
        ax.add_artist(legend1)
        ax.add_artist(legend2)
   
        # Add information box
        info_text = f"""
Peak Detection Info:
• Primary: Hip-Heel (solid lines)
• Secondary: Hip-Ankle (dashed lines)
• Peaks: Circle markers
• GT: Vertical dotted lines
        """.strip()
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)) 

    def print_detailed_results(self, comparison_results: Dict[str, Any]) -> None:
        """Print detailed comparison results to console"""
        metrics = comparison_results['metrics']
        
        print("\n" + "=" * 60)
        print("FOOTSTEP DETECTION RESULTS")
        print("=" * 60)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Detection Rate:     {metrics['detection_rate']:.1f}%")
        print(f"   Precision:          {metrics['precision']:.1f}%") 
        print(f"   F1 Score:           {metrics['f1_score']:.1f}%")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.1f}%")
        
        print(f"\n⏱️  TIME ACCURACY:")
        print(f"   Mean Absolute Error: {metrics['mean_abs_error']:.3f}s")
        print(f"   Max Absolute Error:  {metrics['max_abs_error']:.3f}s")
        print(f"   Mean Error (bias):   {metrics['mean_time_error']:.3f}s")
        print(f"   Std Error:           {metrics['std_time_error']:.3f}s")
        
        print(f"\n🎯 DETECTION SUMMARY:")
        print(f"   Matches:             {metrics['total_matches']}/{metrics['total_ground_truth']}")
        print(f"   Total Detected:      {metrics['total_detected']}")
        print(f"   False Positives:     {len(comparison_results['false_positives'])}")
        print(f"   Missed Detections:   {len(comparison_results['missed_detections'])}")
        
        # Show detailed matches if there are any
        if comparison_results['matches']:
            print(f"\n✅ SUCCESSFUL MATCHES:")
            for i, match in enumerate(comparison_results['matches']):
                print(f"   {i+1:2d}. GT: {match['gt_timestamp']:.3f}s ({match['gt_foot']}) → "
                      f"Detected: {match['detected_timestamp']:.3f}s "
                      f"(error: {match['time_error']:+.3f}s)")
        
        # Show false positives if any
        if comparison_results['false_positives']:
            print(f"\n❌ FALSE POSITIVES:")
            for i, fp in enumerate(comparison_results['false_positives']):
                print(f"   {i+1:2d}. Detected at {fp['timestamp']:.3f}s (no matching ground truth)")
        
        # Show missed detections if any  
        if comparison_results['missed_detections']:
            print(f"\n⚠️  MISSED DETECTIONS:")
            for i, md in enumerate(comparison_results['missed_detections']):
                print(f"   {i+1:2d}. GT: {md['timestamp']:.3f}s ({md['foot']}) - not detected")
    
    def run_full_analysis(self, video_path: str, ground_truth_path: str, 
                         save_visualization: str = None, verbose: bool = True) -> Dict[str, Any]:
        """
        Run complete footstep detection and evaluation pipeline
        
        Args:
            video_path: Path to video file
            ground_truth_path: Path to ground truth JSON file
            save_visualization: Path to save visualization (optional)
            verbose: Print detailed information
            
        Returns:
            Dictionary containing all results
        """
        try:
            # Load ground truth
            if verbose:
                print("📋 Loading ground truth...")
            ground_truth = self.load_ground_truth(ground_truth_path)
            self.last_ground_truth = ground_truth
            
            if verbose:
                print(f"   Ground truth: {ground_truth['total_steps']} steps")
                print(f"   Video: {ground_truth['video_path']}")
                print(f"   Duration: {ground_truth['video_info']['duration']:.1f}s")
            
            # Run detection pipeline
            detection_results = self.process_video(video_path, verbose=verbose)
            
            # Compare with ground truth
            if verbose:
                print("\n🔍 Comparing with ground truth...")
            
            comparison_results = self.compare_with_ground_truth(
                detection_results['detected_timestamps'], 
                ground_truth
            )
            self.last_comparison_results = comparison_results
            
            # Print detailed results
            if verbose:
                self.print_detailed_results(comparison_results)
            
            # Generate visualizations
            if verbose:
                print(f"\n📊 Generating visualizations...")
            
            self.generate_visualizations(
                comparison_results, 
                ground_truth, 
                detection_results['detected_timestamps'],
                save_visualization
            )
            
            # Compile final results
            final_results = {
                'detection_results': detection_results,
                'ground_truth': ground_truth,
                'comparison_results': comparison_results,
                'config': self.config
            }
            
            if verbose:
                print(f"\n🎉 Analysis complete!")
                if save_visualization:
                    print(f"   Visualization saved to: {save_visualization}")
                
                # Summary of responsibility separation
                print(f"\n📋 Pipeline Responsibility Summary:")
                print(f"   🏃 PoseExtractor: Extracted landmarks from video")
                print(f"   📊 SignalProcessor: Cleaned coordinate signals")
                print(f"   👣 GaitDetector: Detected {len(detection_results['detected_frame_indices'])} frame indices")
                print(f"   ⏰ FootstepDetector: Converted to {len(detection_results['detected_timestamps'])} timestamps")
                print(f"   🔍 FootstepDetector: Compared with ground truth")
                print(f"   📊 FootstepDetector: Generated visualizations")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.pose_extractor:
            self.pose_extractor.cleanup()


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = FootstepDetectorConfig(
        matching_tolerance=0.3,
        pose_extractor_fps=10,
        pose_confidence_threshold=0.7
    )
    
    print("Updated Footstep Detector Configuration:")
    print(f"  Matching tolerance: ±{config.matching_tolerance}s")
    print(f"  Pose extractor FPS: {config.pose_extractor_fps}")
    print(f"  Confidence threshold: {config.pose_confidence_threshold}")
    
    # Initialize detector
    detector = FootstepDetector(config)
    
    print(f"\n✅ Clear Separation of Concerns:")
    print(f"   🎯 gait_detector.py: Pure detection (frame indices only)")
    print(f"   ⏰ footstep_detector.py: Timestamp conversion + evaluation + visualization")
    print(f"   📈 Better modularity and reusability")
    
    # File paths (update these to your actual file paths)
    video_path = "./test_videos/walk4.mp4"
    ground_truth_path = "./test_videos/walk4_ground_truth.json"
    output_path = "./output/walk4_analysis.png"
    
    try:
        # Run full analysis
        results = detector.run_full_analysis(
            video_path=video_path,
            ground_truth_path=ground_truth_path,
            save_visualization=output_path,
            verbose=True
        )
        
        print(f"\n🎯 Final Summary:")
        metrics = results['comparison_results']['metrics']
        print(f"   Detection Rate: {metrics['detection_rate']:.1f}%")
        print(f"   Precision: {metrics['precision']:.1f}%")
        print(f"   Mean Time Error: {metrics['mean_abs_error']:.3f}s")
        
        # Show the pipeline flow
        detection_results = results['detection_results']
        print(f"\n🔄 Pipeline Flow Verification:")
        print(f"   Frame indices detected: {len(detection_results['detected_frame_indices'])}")
        print(f"   Timestamps converted: {len(detection_results['detected_timestamps'])}")
        print(f"   Original video FPS: {detection_results['pipeline_metadata']['original_fps']:.2f}")
        print(f"   Processing FPS: {detection_results['pipeline_metadata']['processed_fps']:.2f}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please update the file paths in the script")
        
        # Show what the detector expects
        print(f"\nExpected files:")
        print(f"  Video: {video_path}")
        print(f"  Ground truth: {ground_truth_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Clean up
        detector.cleanup()
        print("\nFootstep detector cleanup complete!")
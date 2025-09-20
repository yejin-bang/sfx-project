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

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.footstep_detector import FootstepDetector, FootstepDetectorConfig


@dataclass
class VisualizerConfig:
    """Configuration for visualization and evaluation"""
    # Matching parameters
    matching_tolerance: float = 0.3  # ±0.3 seconds for ground truth matching
    
    # Visualization parameters
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    
    # Distance signals plot (large single plot)
    distance_plot_size: Tuple[int, int] = (15, 8)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.matching_tolerance <= 0:
            raise ValueError("matching_tolerance must be positive")


class FootstepVisualizer:
    """
    Visualization and evaluation module for footstep detection
    
    Features:
    1. Load ground truth annotations from JSON
    2. Compare detection results with ground truth
    3. Calculate performance metrics
    4. Generate comprehensive visualizations
    5. Create detailed analysis reports
    
    This module is independent and only used for verification/testing
    """
    
    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize footstep visualizer
        
        Args:
            config: VisualizerConfig object. If None, uses default values.
        """
        self.config = config if config is not None else VisualizerConfig()
        
        # Storage for analysis results
        self.last_ground_truth = None
        self.last_comparison_results = None
        self.last_detection_results = None
    
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
    
    def create_distance_signals_plot(self, detection_results: Dict[str, Any], 
                                   ground_truth: Dict[str, Any],
                                   save_path: str = None) -> None:
        """
        Create large single plot for distance signals and peak detection
        
        Args:
            detection_results: Results from footstep_detector
            ground_truth: Ground truth data
            save_path: Path to save plot (optional)
        """
        # Create large single figure
        fig, ax = plt.subplots(1, 1, figsize=self.config.distance_plot_size, 
                              dpi=self.config.dpi)
        
        # Get data from detection results
        gait_results = detection_results['gait_results']
        distance_signals = gait_results['distance_signals']
        detailed_results = gait_results['detailed_results']
        timestamps = detection_results['processed_data']['timestamps']
        
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
                              linewidth=2.5, alpha=0.8,
                              label=signal_name.replace('_', ' ').title())[0]
                signal_legends.append(line)
                
                # Plot detected peaks
                if signal_name in detailed_results:
                    peak_indices = detailed_results[signal_name]['peak_frame_indices']
                    if peak_indices:
                        peak_timestamps = timestamps[peak_indices]
                        peak_values = distances[peak_indices]
                        
                        ax.scatter(peak_timestamps, peak_values, 
                                  color=color, s=120, marker='o', 
                                  edgecolor='white', linewidth=3,
                                  zorder=5, alpha=0.9)
        
        # Plot ground truth as vertical lines
        gt_timestamps = ground_truth['timestamps']
        gt_feet = ground_truth['feet']
        
        y_min, y_max = ax.get_ylim()
        for timestamp, foot in zip(gt_timestamps, gt_feet):
            line_color = 'blue' if foot == 'left' else 'red'
            ax.axvline(x=timestamp, color=line_color, linestyle=':', 
                      linewidth=3, alpha=0.7, zorder=3)
        
        # Customize plot
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hip-Foot Distance (normalized)', fontsize=14, fontweight='bold')
        ax.set_title('Distance Signals and Peak Detection Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Create legends with minimal space usage
        # Distance signals legend (top left)
        legend1 = ax.legend(handles=signal_legends, 
                           loc='upper left', fontsize=11,
                           title='Distance Signals', title_fontsize=12)
        legend1.get_frame().set_alpha(0.9)
        
        # Ground truth and peaks legend (top right)
        from matplotlib.lines import Line2D
        custom_elements = [
            Line2D([0], [0], color='blue', linestyle=':', linewidth=3, 
                   label='GT Left Foot'),
            Line2D([0], [0], color='red', linestyle=':', linewidth=3, 
                   label='GT Right Foot'),
            Line2D([0], [0], marker='o', color='gray', linestyle='None',
                   markersize=10, markeredgecolor='white', markeredgewidth=3,
                   label='Detected Peaks')
        ]
        
        legend2 = ax.legend(handles=custom_elements, 
                           loc='upper right', fontsize=11,
                           title='Ground Truth & Peaks', title_fontsize=12)
        legend2.get_frame().set_alpha(0.9)
        
        # Add both legends
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        
        # Minimal info box at bottom right
        info_text = "Primary: Hip-Heel (solid) | Secondary: Hip-Ankle (dashed)"
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"📊 Distance signals plot saved: {save_path}")
        
        plt.show()
    
    def create_comparison_subplots(self, comparison_results: Dict[str, Any], 
                                 ground_truth: Dict[str, Any],
                                 detected_timestamps: List[float],
                                 save_path: str = None) -> None:
        """
        Create comparison visualization with 2 subplots (timeline + metrics)
        
        Args:
            comparison_results: Results from compare_with_ground_truth()
            ground_truth: Ground truth data
            detected_timestamps: All detected timestamps
            save_path: Path to save visualization (optional)
        """
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size, 
                                      dpi=self.config.dpi)
        
        # Timeline comparison plot
        self._plot_timeline_comparison(ax1, comparison_results, ground_truth, detected_timestamps)
        
        # Metrics summary plot
        self._plot_metrics_summary(ax2, comparison_results['metrics'])
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"📊 Comparison plots saved: {save_path}")
        
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
    
    def print_detailed_results(self, comparison_results: Dict[str, Any]) -> None:
        """Print detailed comparison results to console"""
        metrics = comparison_results['metrics']
        
        print("\n" + "=" * 60)
        print("FOOTSTEP DETECTION EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Detection Rate:     {metrics['detection_rate']:.1f}%")
        print(f"   Precision:          {metrics['precision']:.1f}%") 
        print(f"   F1 Score:           {metrics['f1_score']:.1f}%")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.1f}%")
        
        print(f"\nⱖⅈ TIME ACCURACY:")
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
            print(f"\n⚠️ MISSED DETECTIONS:")
            for i, md in enumerate(comparison_results['missed_detections']):
                print(f"   {i+1:2d}. GT: {md['timestamp']:.3f}s ({md['foot']}) - not detected")
    
    def analyze_video_with_ground_truth(self, video_path: str, ground_truth_path: str,
                                      save_distance_plot: str = None,
                                      save_comparison_plot: str = None,
                                      verbose: bool = True) -> Dict[str, Any]:
        """
        Complete analysis pipeline with ground truth comparison
        
        Args:
            video_path: Path to video file
            ground_truth_path: Path to ground truth JSON file
            save_distance_plot: Path to save distance signals plot (optional)
            save_comparison_plot: Path to save comparison plot (optional)
            verbose: Print detailed information
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            if verbose:
                print("=" * 60)
                print("FOOTSTEP DETECTION EVALUATION PIPELINE")
                print("=" * 60)
            
            # Step 1: Load ground truth
            if verbose:
                print("\n📋 Step 1: Loading ground truth...")
            ground_truth = self.load_ground_truth(ground_truth_path)
            self.last_ground_truth = ground_truth
            
            if verbose:
                print(f"   Ground truth: {ground_truth['total_steps']} steps")
                print(f"   Left: {ground_truth['left_steps']}, Right: {ground_truth['right_steps']}")
                print(f"   Video duration: {ground_truth['video_info']['duration']:.1f}s")
            
            # Step 2: Run detection pipeline
            if verbose:
                print(f"\n🔍 Step 2: Running detection pipeline...")
            
            detector = FootstepDetector()
            detection_results = detector.process_video(video_path, verbose=verbose)
            self.last_detection_results = detection_results
            
            # Step 3: Compare with ground truth
            if verbose:
                print(f"\n📊 Step 3: Comparing with ground truth...")
            
            comparison_results = self.compare_with_ground_truth(
                detection_results['detected_timestamps'], 
                ground_truth
            )
            self.last_comparison_results = comparison_results
            
            # Step 4: Print detailed results
            if verbose:
                self.print_detailed_results(comparison_results)
            
            # Step 5: Generate visualizations
            if verbose:
                print(f"\n📈 Step 4: Generating visualizations...")
            
            # Create distance signals plot (large single plot)
            if save_distance_plot or verbose:
                if verbose:
                    print("   Creating distance signals analysis plot...")
                self.create_distance_signals_plot(
                    detection_results, 
                    ground_truth,
                    save_distance_plot
                )
            
            # Create comparison subplots
            if save_comparison_plot or verbose:
                if verbose:
                    print("   Creating comparison analysis plots...")
                self.create_comparison_subplots(
                    comparison_results, 
                    ground_truth, 
                    detection_results['detected_timestamps'],
                    save_comparison_plot
                )
            
            # Compile final results
            analysis_results = {
                'detection_results': detection_results,
                'ground_truth': ground_truth,
                'comparison_results': comparison_results,
                'visualizer_config': self.config
            }
            
            if verbose:
                print(f"\n🎉 Evaluation complete!")
                if save_distance_plot:
                    print(f"   Distance plot saved to: {save_distance_plot}")
                if save_comparison_plot:
                    print(f"   Comparison plot saved to: {save_comparison_plot}")
                
                # Summary
                metrics = comparison_results['metrics']
                print(f"\n📋 Final Summary:")
                print(f"   Detection Rate: {metrics['detection_rate']:.1f}%")
                print(f"   Precision: {metrics['precision']:.1f}%")
                print(f"   F1 Score: {metrics['f1_score']:.1f}%")
                print(f"   Mean Time Error: {metrics['mean_abs_error']:.3f}s")
            
            # Clean up detector
            detector.cleanup()
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            raise
    
    def quick_analysis(self, video_path: str, ground_truth_path: str) -> Dict[str, float]:
        """
        Quick analysis that returns only key metrics
        
        Args:
            video_path: Path to video file
            ground_truth_path: Path to ground truth JSON file
            
        Returns:
            Dictionary with key performance metrics
        """
        analysis_results = self.analyze_video_with_ground_truth(
            video_path, ground_truth_path, verbose=False
        )
        
        metrics = analysis_results['comparison_results']['metrics']
        
        return {
            'detection_rate': metrics['detection_rate'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'mean_abs_error': metrics['mean_abs_error']
        }


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = VisualizerConfig(
        matching_tolerance=0.3,
        figure_size=(12, 8),
        distance_plot_size=(15, 8)
    )
    
    print("Footstep Visualizer Configuration:")
    print(f"  Matching tolerance: ±{config.matching_tolerance}s")
    print(f"  Figure size: {config.figure_size}")
    print(f"  Distance plot size: {config.distance_plot_size}")
    
    # Initialize visualizer
    visualizer = FootstepVisualizer(config)
    
    print(f"\n✅ Visualization & Evaluation Module:")
    print(f"   🎯 Independent from main detection pipeline")
    print(f"   🎯 Used for verification and testing only")
    print(f"   🎯 Large distance signals plot + comparison subplots")
    
    # File paths (update these to your actual file paths)
    video_path = "./test_videos/walk4.mp4"
    ground_truth_path = "./test_videos/walk4_ground_truth.json"
    distance_plot_path = "./output/walk4_distance_analysis.png"
    comparison_plot_path = "./output/walk4_comparison_analysis.png"
    
    try:
        # Method 1: Full analysis with visualizations
        analysis_results = visualizer.analyze_video_with_ground_truth(
            video_path=video_path,
            ground_truth_path=ground_truth_path,
            save_distance_plot=distance_plot_path,
            save_comparison_plot=comparison_plot_path,
            verbose=True
        )
        
        print(f"\n🎯 Analysis Complete!")
        print(f"   Detection pipeline: Independent")
        print(f"   Evaluation pipeline: This module")
        print(f"   Visualizations: Generated")
        
        # Method 2: Quick analysis (metrics only)
        print(f"\n⚡ Quick Analysis Example:")
        quick_metrics = visualizer.quick_analysis(video_path, ground_truth_path)
        for metric, value in quick_metrics.items():
            print(f"   {metric}: {value:.1f}{'%' if 'rate' in metric or 'precision' in metric or 'f1' in metric else 's'}")
        
        # Show module separation
        print(f"\n📁 Module Separation Summary:")
        print(f"   📊 footstep_detector.py: Pure detection only")
        print(f"   📈 footstep_visualizer.py: Evaluation & visualization")
        print(f"   🎵 Ready for audio pipeline integration")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please update the file paths in the script")
        
        # Show what the visualizer expects
        print(f"\nExpected files:")
        print(f"  Video: {video_path}")
        print(f"  Ground truth: {ground_truth_path}")
        
        # Show expected usage
        print(f"\nExpected usage:")
        print(f"1. visualizer = FootstepVisualizer()")
        print(f"2. results = visualizer.analyze_video_with_ground_truth('video.mp4', 'gt.json')")
        print(f"3. metrics = visualizer.quick_analysis('video.mp4', 'gt.json')")
        
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nFootstep visualizer example complete!")
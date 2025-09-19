import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings

@dataclass
class GaitDetectorConfig:
    """Configuration for gait detection parameters"""
    # Peak detection parameters
    peak_height_threshold: float = 0.1  # Minimum peak height (relative)
    peak_prominence: float = 0.05  # Minimum peak prominence
    peak_distance: int = 5  # Minimum distance between peaks (frames)
    # 10fps = 5 frames = 0.5 sec
    
    # Distance calculation parameters
    normalize_distances: bool = True  # Normalize distances to 0-1 range
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.peak_height_threshold < 0:
            raise ValueError("peak_height_threshold must be non-negative")
        if self.peak_prominence < 0:
            raise ValueError("peak_prominence must be non-negative")
        if self.peak_distance < 1:
            raise ValueError("peak_distance must be positive")


class GaitDetector:
    """
    Detect gait events (heel strikes) from clean coordinate data
    
    Simplified Pipeline:
    1. Calculate Hip-Foot Euclidean distances for both legs
    2. Calculate Hip-Ankle distances as backup method
    3. Detect peaks in distance signals (heel strikes)
    4. Return frame indices (NOT timestamps - that's footstep_detector's job)
    
    Distance signals calculated:
    - LEFT_HIP-LEFT_HEEL (primary left leg)
    - RIGHT_HIP-RIGHT_HEEL (primary right leg)  
    - LEFT_HIP-LEFT_ANKLE (secondary left leg)
    - RIGHT_HIP-RIGHT_ANKLE (secondary right leg)
    """
    
    def __init__(self, config: Optional[GaitDetectorConfig] = None):
        """
        Initialize gait detector
        
        Args:
            config: GaitDetectorConfig object. If None, uses default values.
        """
        self.config = config if config is not None else GaitDetectorConfig()
        
        # Define distance signal pairs
        self.distance_pairs = {
            'primary': {
                'left': ('LEFT_HIP', 'LEFT_HEEL'),
                'right': ('RIGHT_HIP', 'RIGHT_HEEL')
            },
            'secondary': {
                'left': ('LEFT_HIP', 'LEFT_ANKLE'), 
                'right': ('RIGHT_HIP', 'RIGHT_ANKLE')
            }
        }
    
    def calculate_euclidean_distance(self, point1_coords: Dict, point2_coords: Dict) -> np.ndarray:
        """
        Calculate Euclidean distance between two coordinate time series
        
        Args:
            point1_coords: Dictionary with 'x', 'y' coordinate arrays
            point2_coords: Dictionary with 'x', 'y' coordinate arrays
            
        Returns:
            Array of Euclidean distances for each frame
        """
        x1, y1 = point1_coords['x'], point1_coords['y']
        x2, y2 = point2_coords['x'], point2_coords['y']
        
        # Stack coordinates into (N, 2) arrays for vectorized calculation
        points1 = np.column_stack([x1, y1])
        points2 = np.column_stack([x2, y2])
        
        # Calculate Euclidean distance using NumPy's optimized norm function
        distances = np.linalg.norm(points2 - points1, axis=1)
        
        return distances
    
    def calculate_hip_foot_distances(self, clean_coordinates: Dict) -> Dict[str, np.ndarray]:
        """
        Calculate all 4 distance signals from clean coordinate data
        
        Args:
            clean_coordinates: Clean coordinates from signal_processor
            
        Returns:
            Dictionary containing all distance signals
        """
        distance_signals = {}
        
        # Calculate primary distances (HIP-HEEL)
        for side in ['left', 'right']:
            hip_name, heel_name = self.distance_pairs['primary'][side]
            
            try:
                hip_coords = clean_coordinates['primary'][hip_name]
                heel_coords = clean_coordinates['primary'][heel_name]
                
                distance = self.calculate_euclidean_distance(hip_coords, heel_coords)
                signal_name = f"{side}_hip_heel"
                distance_signals[signal_name] = distance
                
            except KeyError as e:
                warnings.warn(f"Missing landmark for {signal_name}: {e}")
                distance_signals[signal_name] = None
        
        # Calculate secondary distances (HIP-ANKLE)  
        for side in ['left', 'right']:
            hip_name, ankle_name = self.distance_pairs['secondary'][side]
            
            try:
                hip_coords = clean_coordinates['secondary'][hip_name] 
                ankle_coords = clean_coordinates['secondary'][ankle_name]
                
                distance = self.calculate_euclidean_distance(hip_coords, ankle_coords)
                signal_name = f"{side}_hip_ankle"
                distance_signals[signal_name] = distance
                
            except KeyError as e:
                warnings.warn(f"Missing landmark for {signal_name}: {e}")
                distance_signals[signal_name] = None
        
        # Normalize distances if requested
        if self.config.normalize_distances:
            distance_signals = self._normalize_distance_signals(distance_signals)
        
        return distance_signals
    
    def detect_heel_strikes_basic(self, distance_signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Basic peak detection for heel strikes using scipy.signal.find_peaks
        
        Args:
            distance_signals: Dictionary of distance signals
            
        Returns:
            Dictionary containing heel strike detections for each signal (FRAME INDICES ONLY)
        """
        heel_strike_results = {}
        
        for signal_name, distances in distance_signals.items():
            if distances is None:
                heel_strike_results[signal_name] = {
                    'peak_frame_indices': [],
                    'peak_values': [],
                    'detection_count': 0
                }
                continue
            
            # Find peaks (heel strikes = maximum Hip-Foot distance)
            peak_indices, peak_properties = signal.find_peaks(
                distances,
                height=self.config.peak_height_threshold,
                prominence=self.config.peak_prominence, 
                distance=self.config.peak_distance
            )
            
            # Get peak values
            peak_values = distances[peak_indices] if len(peak_indices) > 0 else []
            
            heel_strike_results[signal_name] = {
                'peak_frame_indices': peak_indices.tolist(),  # 🎯 Frame indices only!
                'peak_values': peak_values.tolist(),
                'detection_count': len(peak_indices),
                'peak_properties': peak_properties
            }
        
        return heel_strike_results
    
    def process_gait_events(self, processed_data: Dict, verbose: bool = True) -> Dict[str, Any]:
        """
        Main pipeline method to detect gait events from signal processor output
        
        Args:
            processed_data: Output from signal_processor.process_coordinates()
            verbose: Print processing information
            
        Returns:
            Dictionary containing heel strike FRAME INDICES and metadata (NO TIMESTAMPS)
        """
        if verbose:
            print("Starting gait event detection...")
            print(f"Configuration: {self.config}")
        
        # Extract data from signal processor output
        clean_coordinates = processed_data['clean_coordinates']
        total_frames = len(processed_data['timestamps'])
        
        if verbose:
            print(f"Processing {total_frames} frames")
        
        # Step 1: Calculate Hip-Foot distances
        if verbose:
            print("Step 1: Calculating Hip-Foot distances...")
        
        distance_signals = self.calculate_hip_foot_distances(clean_coordinates)
        
        # Print distance signal info
        if verbose:
            print("Distance signals calculated:")
            for signal_name, distances in distance_signals.items():
                if distances is not None:
                    print(f"  {signal_name}: mean={np.mean(distances):.2f}, std={np.std(distances):.2f}")
                else:
                    print(f"  {signal_name}: MISSING DATA")
        
        # Step 2: Detect heel strikes
        if verbose:
            print("Step 2: Detecting heel strikes...")
        
        heel_strike_results = self.detect_heel_strikes_basic(distance_signals)
        
        # Print detection results
        if verbose:
            print("Heel strike detection results:")
            total_detections = 0
            for signal_name, results in heel_strike_results.items():
                count = results['detection_count']
                total_detections += count
                print(f"  {signal_name}: {count} heel strikes detected")
        
        # Step 3: Combine results (frame indices only)
        combined_frame_indices = self._combine_heel_strike_frame_indices(heel_strike_results)
        
        if verbose:
            print(f"Total unique heel strikes: {len(combined_frame_indices)}")
            print("Gait event detection complete!")
        
        # Prepare final result (NO TIMESTAMP CONVERSION)
        result = {
            'heel_strike_frame_indices': combined_frame_indices,  # 🎯 Frame indices only!
            'detailed_results': heel_strike_results,
            'distance_signals': distance_signals,
            'metadata': {
                'total_detections': len(combined_frame_indices),
                'processing_config': self.config,
                'signal_pairs': self.distance_pairs,
                'total_frames': total_frames,
                'sampling_rate': processed_data['metadata']['sampling_rate'],
                'data_coverage': processed_data['metadata']['data_coverage']
            }
        }
        
        return result
    
    def get_heel_strike_frame_indices_only(self, processed_data: Dict) -> List[int]:
        """
        Convenience method to get just the heel strike frame indices
        
        Returns:
            List of heel strike frame indices
        """
        result = self.process_gait_events(processed_data, verbose=False)
        return result['heel_strike_frame_indices']
    
    def _normalize_distance_signals(self, distance_signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize distance signals to 0-1 range"""
        normalized_signals = {}
        
        for signal_name, distances in distance_signals.items():
            if distances is not None:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > min_dist:  # Avoid division by zero
                    normalized = (distances - min_dist) / (max_dist - min_dist)
                    normalized_signals[signal_name] = normalized
                else:
                    normalized_signals[signal_name] = np.zeros_like(distances)
            else:
                normalized_signals[signal_name] = None
        
        return normalized_signals
    
    def _combine_heel_strike_frame_indices(self, heel_strike_results: Dict) -> List[int]:
        """
        Combine heel strike frame indices from all signals and remove duplicates
        
        Args:
            heel_strike_results: Results from detect_heel_strikes_basic
            
        Returns:
            Sorted list of unique heel strike frame indices
        """
        all_frame_indices = []
        
        for signal_name, results in heel_strike_results.items():
            if results['detection_count'] > 0:
                all_frame_indices.extend(results['peak_frame_indices'])
        
        # Remove duplicates and sort
        # Use small tolerance for frame index comparison (usually 1-2 frames)
        unique_frame_indices = []
        tolerance = 2  # 2 frames tolerance for duplicate removal
        
        for frame_idx in sorted(all_frame_indices):
            is_duplicate = False
            for existing in unique_frame_indices:
                if abs(frame_idx - existing) <= tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_frame_indices.append(frame_idx)
        
        return sorted(unique_frame_indices)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = GaitDetectorConfig(
        peak_height_threshold=0.1,
        peak_prominence=0.05,
        peak_distance=5,
        normalize_distances=True
    )
    
    print(f"Gait Detector Configuration:")
    print(f"  Peak height threshold: {config.peak_height_threshold}")
    print(f"  Peak prominence: {config.peak_prominence}")
    print(f"  Peak distance: {config.peak_distance} frames")
    print(f"  Normalize distances: {config.normalize_distances}")
    
    # Initialize detector
    detector = GaitDetector(config)
    
    print(f"\nDistance pairs for calculation:")
    print(f"  Primary: {detector.distance_pairs['primary']}")
    print(f"  Secondary: {detector.distance_pairs['secondary']}")
    
    print(f"\nGait detector ready!")
    print(f"Input: Clean coordinates from signal_processor.py")
    print(f"Output: Heel strike FRAME INDICES for footstep_detector.py")
    
    # Test with signal_processor if available
    print(f"\n=== Testing Integration with Signal Processor ===")
    try:
        from signal_processor import SignalProcessor, SignalProcessorConfig
        from pose_extractor import PoseExtractor
        
        # Create sample data pipeline
        print("Creating test pipeline...")
        
        # This would normally come from a real video
        print("Note: Replace with real video processing for actual use")
        print("Example integration:")
        print("1. extractor = PoseExtractor()")
        print("2. landmarks_data, timestamps = extractor.get_signal_processor_input('video.mp4')")
        print("3. processor = SignalProcessor()")  
        print("4. processed_data = processor.process_coordinates(landmarks_data, timestamps)")
        print("5. detector = GaitDetector()")
        print("6. frame_indices = detector.get_heel_strike_frame_indices_only(processed_data)")
        print("7. # footstep_detector.py converts frame_indices to timestamps")
        
    except ImportError as e:
        print(f"Some modules not available: {e}")
        print("This is normal if running gait_detector.py independently")
    
    print(f"\n=== Simplified Gait Detector Ready ===")
    print("✅ Only detects frame indices - timestamp conversion moved to footstep_detector.py")
    print("✅ Clear separation of concerns")
    print("✅ Better reusability")
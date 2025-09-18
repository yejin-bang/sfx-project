import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings

@dataclass
class SignalProcessorConfig:
    """Configuration for signal processing parameters"""
    cutoff_frequency: float = 0.1752  # Research-validated Butterworth cutoff
    filter_order: int = 10  # 10th-order Butterworth filter
    min_data_threshold: float = 0.5  # Minimum 50% valid data required
    confidence_threshold: float = 0.7  # Minimum landmark confidence
    interpolation_method: str = 'cubic'  # Cubic spline interpolation
    interpolation_max_gap: int = 10  # Maximum consecutive frames to interpolate
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not 0 < self.cutoff_frequency < 0.5:
            raise ValueError("cutoff_frequency must be between 0 and 0.5 (Nyquist frequency)")
        if self.filter_order < 1:
            raise ValueError("filter_order must be positive")
        if not 0 < self.min_data_threshold <= 1:
            raise ValueError("min_data_threshold must be between 0 and 1")


class SignalProcessor:
    """
    Clean coordinate time series for gait detection
    
    Pipeline:
    1. Extract coordinate time series from landmarks
    2. Cubic spline interpolation for missing values  
    3. 10th-order Butterworth low-pass filtering
    4. Return clean coordinates for gait_detector.py
    
    Primary landmarks: HIP, HEEL
    Secondary landmarks: HIP, ANKLE  
    """
    
    def __init__(self, config: Optional[SignalProcessorConfig] = None):
        """
        Initialize signal processor
        
        Args:
            config: SignalProcessorConfig object. If None, uses default values.
        """
        self.config = config if config is not None else SignalProcessorConfig()
        
        # Define landmark pairs for processing
        self.landmark_pairs = {
            'primary': ['LEFT_HIP', 'LEFT_HEEL', 'RIGHT_HIP', 'RIGHT_HEEL'],
            'secondary': ['LEFT_HIP', 'LEFT_ANKLE', 'RIGHT_HIP', 'RIGHT_ANKLE']
        }
    
    def extract_coordinate_timeseries(self, landmarks_data: List[Dict], 
                                    timestamps: List[float]) -> Dict[str, Any]:
        """
        Extract coordinate time series from landmark data
        
        Args:
            landmarks_data: List of landmark dictionaries from pose_extractor
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary containing coordinate time series for each landmark
        """
        # Initialize coordinate arrays
        coordinate_series = {}
        all_landmarks = set(self.landmark_pairs['primary'] + self.landmark_pairs['secondary'])
        
        for landmark_name in all_landmarks:
            coordinate_series[landmark_name] = {
                'x': [],
                'y': [],
                'confidence': []
            }
        
        total_frames = len(landmarks_data)
        frames_with_data = 0
        valid_timestamps = []
        
        # Extract coordinates frame by frame
        for i, (landmarks, timestamp) in enumerate(zip(landmarks_data, timestamps)):
            frame_has_data = False
            
            # Process each landmark
            for landmark_name in all_landmarks:
                if landmarks is None or landmark_name not in landmarks:
                    # Missing landmark data
                    coordinate_series[landmark_name]['x'].append(None)
                    coordinate_series[landmark_name]['y'].append(None)
                    coordinate_series[landmark_name]['confidence'].append(0.0)
                else:
                    x, y, confidence = landmarks[landmark_name]
                    
                    # Check confidence threshold
                    if confidence >= self.config.confidence_threshold and x is not None and y is not None:
                        coordinate_series[landmark_name]['x'].append(x)
                        coordinate_series[landmark_name]['y'].append(y)
                        coordinate_series[landmark_name]['confidence'].append(confidence)
                        frame_has_data = True
                    else:
                        # Low confidence or invalid coordinates
                        coordinate_series[landmark_name]['x'].append(None)
                        coordinate_series[landmark_name]['y'].append(None)
                        coordinate_series[landmark_name]['confidence'].append(confidence)
            
            if frame_has_data:
                frames_with_data += 1
            
            valid_timestamps.append(timestamp)
        
        # Check if we have enough data to proceed
        data_coverage = frames_with_data / total_frames if total_frames > 0 else 0
        
        if data_coverage < self.config.min_data_threshold:
            raise ValueError(f"Insufficient data coverage: {data_coverage:.2%} "
                           f"(minimum required: {self.config.min_data_threshold:.2%})")
        
        # Convert to numpy arrays
        for landmark_name in all_landmarks:
            coordinate_series[landmark_name]['x'] = np.array(coordinate_series[landmark_name]['x'], dtype=float)
            coordinate_series[landmark_name]['y'] = np.array(coordinate_series[landmark_name]['y'], dtype=float)
            coordinate_series[landmark_name]['confidence'] = np.array(coordinate_series[landmark_name]['confidence'])
        
        return {
            'coordinates': coordinate_series,
            'timestamps': np.array(valid_timestamps),
            'data_coverage': data_coverage,
            'total_frames': total_frames,
            'frames_with_data': frames_with_data
        }
    
    def interpolate_coordinate_series(self, coord_array: np.ndarray, 
                                    timestamps: np.ndarray, 
                                    coordinate_name: str) -> np.ndarray:
        """
        Fill missing values in coordinate time series using cubic spline interpolation
        
        Args:
            coord_array: Array with None/NaN values for missing coordinates
            timestamps: Corresponding timestamps
            coordinate_name: Name for debugging (e.g., "LEFT_HIP_x")
            
        Returns:
            Interpolated coordinate array
        """
        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(coord_array)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            warnings.warn(f"Not enough valid points for {coordinate_name} interpolation")
            return coord_array
        
        # Check for interpolation gaps that are too large
        if self._has_large_gaps(valid_indices):
            warnings.warn(f"Found gaps larger than {self.config.interpolation_max_gap} frames in {coordinate_name}")
        
        # Perform cubic spline interpolation
        try:
            interpolator = interp1d(
                timestamps[valid_indices], 
                coord_array[valid_indices],
                kind=self.config.interpolation_method,
                fill_value='extrapolate',
                bounds_error=False
            )
            
            # Interpolate only for missing values, keep original valid values
            interpolated = coord_array.copy()
            missing_mask = np.isnan(coord_array)
            interpolated[missing_mask] = interpolator(timestamps[missing_mask])
            
            return interpolated
            
        except Exception as e:
            warnings.warn(f"Interpolation failed for {coordinate_name}: {e}. Returning original signal.")
            return coord_array
    
    def apply_butterworth_filter(self, coord_array: np.ndarray, 
                               sampling_rate: float, 
                               coordinate_name: str) -> np.ndarray:
        """
        Apply 10th-order Butterworth low-pass filter to coordinate time series
        
        Args:
            coord_array: Input coordinate array
            sampling_rate: Sampling rate in Hz
            coordinate_name: Name for debugging
            
        Returns:
            Filtered coordinate array
        """
        try:
            # Design Butterworth filter
            nyquist_freq = sampling_rate / 2
            normalized_cutoff = self.config.cutoff_frequency * nyquist_freq
            
            # Create filter coefficients
            b, a = signal.butter(
                self.config.filter_order,
                normalized_cutoff,
                btype='low',
                analog=False,
                output='ba',
                fs=sampling_rate
            )
            
            # Apply forward-backward filter for zero phase distortion
            filtered_coords = signal.filtfilt(b, a, coord_array)
            
            return filtered_coords
            
        except Exception as e:
            warnings.warn(f"Filtering failed for {coordinate_name}: {e}. Returning original signal.")
            return coord_array
    
    def process_coordinates(self, landmarks_data: List[Dict], 
                          timestamps: List[float], 
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Complete coordinate processing pipeline
        
        Args:
            landmarks_data: List of landmark dictionaries from pose_extractor
            timestamps: List of timestamps for each frame
            verbose: Print processing information
            
        Returns:
            Dictionary containing clean coordinate time series for gait_detector
        """
        if verbose:
            print("Starting coordinate signal processing...")
            print(f"Configuration: {self.config}")
            print(f"Processing landmarks: {self.landmark_pairs}")
        
        # Step 1: Extract raw coordinate time series
        if verbose:
            print("Step 1: Extracting coordinate time series...")
        
        raw_coords = self.extract_coordinate_timeseries(landmarks_data, timestamps)
        
        if verbose:
            print(f"Data coverage: {raw_coords['data_coverage']:.2%}")
            print(f"Frames processed: {raw_coords['frames_with_data']}/{raw_coords['total_frames']}")
        
        # Calculate sampling rate
        if len(timestamps) > 1:
            avg_interval = np.mean(np.diff(timestamps))
            sampling_rate = 1.0 / avg_interval
        else:
            sampling_rate = 10.0  # Default fallback
        
        if verbose:
            print(f"Sampling rate: {sampling_rate:.2f} Hz")
        
        # Step 2: Process each landmark's coordinates
        processed_coordinates = {}
        all_landmarks = set(self.landmark_pairs['primary'] + self.landmark_pairs['secondary'])
        
        for landmark_name in all_landmarks:
            if verbose:
                print(f"Processing {landmark_name}...")
            
            processed_coordinates[landmark_name] = {}
            
            # Process X and Y coordinates separately
            for coord_type in ['x', 'y']:
                coord_name = f"{landmark_name}_{coord_type}"
                
                # Step 2a: Interpolate missing values
                interpolated = self.interpolate_coordinate_series(
                    raw_coords['coordinates'][landmark_name][coord_type],
                    raw_coords['timestamps'],
                    coord_name
                )
                
                # Step 2b: Apply Butterworth filter
                filtered = self.apply_butterworth_filter(
                    interpolated,
                    sampling_rate,
                    coord_name
                )
                
                processed_coordinates[landmark_name][coord_type] = filtered
            
            # Keep confidence scores (no processing needed)
            processed_coordinates[landmark_name]['confidence'] = raw_coords['coordinates'][landmark_name]['confidence']
        
        if verbose:
            print("Coordinate signal processing complete!")
        
        # Prepare final output for gait_detector.py
        result = {
            'clean_coordinates': {
                'primary': {
                    landmark: processed_coordinates[landmark] 
                    for landmark in self.landmark_pairs['primary']
                },
                'secondary': {
                    landmark: processed_coordinates[landmark] 
                    for landmark in self.landmark_pairs['secondary']
                }
            },
            'raw_coordinates': raw_coords['coordinates'],
            'timestamps': raw_coords['timestamps'],
            'metadata': {
                'sampling_rate': sampling_rate,
                'data_coverage': raw_coords['data_coverage'],
                'total_frames': raw_coords['total_frames'],
                'frames_with_data': raw_coords['frames_with_data'],
                'config': self.config,
                'landmark_pairs': self.landmark_pairs
            }
        }
        
        return result
    
    def get_coordinate_quality_report(self, processed_data: Dict) -> Dict[str, Dict]:
        """
        Generate quality report for processed coordinates
        
        Returns:
            Dictionary with statistics for each landmark
        """
        quality_report = {}
        all_landmarks = set(self.landmark_pairs['primary'] + self.landmark_pairs['secondary'])
        
        for landmark_name in all_landmarks:
            # Calculate detection rate from confidence scores
            confidence_scores = processed_data['raw_coordinates'][landmark_name]['confidence']
            valid_detections = np.sum(confidence_scores >= self.config.confidence_threshold)
            total_frames = len(confidence_scores)
            
            detection_rate = (valid_detections / total_frames * 100) if total_frames > 0 else 0
            avg_confidence = np.mean(confidence_scores[confidence_scores > 0]) if np.any(confidence_scores > 0) else 0
            
            quality_report[landmark_name] = {
                'detection_rate': detection_rate,
                'average_confidence': avg_confidence,
                'valid_frames': valid_detections,
                'total_frames': total_frames
            }
        
        return quality_report
    
    def _has_large_gaps(self, valid_indices: np.ndarray) -> bool:
        """Check if there are gaps larger than max_gap in valid indices"""
        if len(valid_indices) < 2:
            return False
        
        gaps = np.diff(valid_indices)
        return np.any(gaps > self.config.interpolation_max_gap)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = SignalProcessorConfig(
        cutoff_frequency=0.1752,
        filter_order=10,
        min_data_threshold=0.5,
        confidence_threshold=0.7
    )
    
    print(f"Signal Processor Configuration:")
    print(f"  Cutoff frequency: {config.cutoff_frequency}")
    print(f"  Filter order: {config.filter_order}")
    print(f"  Min data threshold: {config.min_data_threshold:.1%}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    
    # Initialize processor
    processor = SignalProcessor(config)
    
    print(f"\nLandmark pairs for processing:")
    print(f"  Primary: {processor.landmark_pairs['primary']}")
    print(f"  Secondary: {processor.landmark_pairs['secondary']}")
    
    # Update configuration example
    processor.config.cutoff_frequency = 0.2
    print(f"\nUpdated cutoff frequency: {processor.config.cutoff_frequency}")
    
    print("\nSignal processor ready!")
    print("Input: Raw landmark coordinates from pose_extractor.py")
    print("Output: Clean coordinate time series for gait_detector.py")
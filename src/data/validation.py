'''
This module provides comprehensive data validation for audio ML pipelines.
it's used to ensure data quality before training models.

Date: June 2025
'''

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import warnings
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

class AudioDataValidator:
    def __init__(self, data_path:str, labels_file: Optional[str] = None):
        self.data_path = Path(data_path)
        self.labels_file = labels_file
        self.audio_files = []
        self.labels_df = None
        self.validation_results = {}

        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aiff'}
        self.target_sample_rate = 44100
        self.min_duration = 0.1
        self.max_duration = 30.0
        self.silence_threshold = 0.001
        print("=" * 80)
        print(f'Data path: {self.data_path}')

    def discover_files(self) -> List[Path]:# -> list: 
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(self.data_path.glob(f'**/*{ext}'))

        self.audio_files = sorted(audio_files)
        print(f"Found {len(self.audio_files)} audio files")
        if len(self.audio_files) == 0:
            print("No audio files found.")

        return self.audio_files
    
    def load_labels(self):
        if not self.labels_file:
            print(f'No labels file provided. Validation will be limited')
            return
        
        try: 
            self.labels_df = pd.read_csv(self.labels_file)
            print(f'Loaded labels: {len(self.labels_df)} entries')

            required_cols = ['filename', 'category']
            missing_cols = [col for col in required_cols if col not in self.labels_df.columns]
            if missing_cols:
                raise ValueError(f'Missing required columns: {missing_cols}')
            
            if 'category' in self.labels_df.columns and 'label' not in self.labels_df.columns:
                self.labels_df['label'] = self.labels_df['category']               

        except Exception as e:
            print(f'Error loading labels: {e}')
            self.labels_df = None
    
    def check_file_formats(self) -> Dict:
        '''Automated check 1: verify all files are acutally valid audio files.
        
        Returns: 
            Dict with format distribution and corruption info'''
        
        print(f'\nAutomated check 1: File Format Validation')
        
        format_counts = Counter()
        corrupted_files = []
        sample_rate_issues = []

        for i, file_path in enumerate(self.audio_files):

            format_counts[file_path.suffix.lower()] += 1

            try:
                y, sr = librosa.load(file_path, sr=None, duration=1.0)

                if sr != self.target_sample_rate:
                    sample_rate_issues.append({
                        'file': file_path.name,
                        'sample_rate': sr,
                        'expected': self.target_sample_rate
                    })
            except Exception as e:
                corrupted_files.append({
                    'file': file_path.name,
                    'error': str(e)
                })
        results = {
            'total_files': len(self.audio_files),
            'format_distribution': dict(format_counts),
            'corrupted_files': corrupted_files,
            'sample_rate_issues': sample_rate_issues,
            'corruption_rate' : len(corrupted_files) / len(self.audio_files) * 100
        }

        print(f"Format distribution: {dict(format_counts)}")
        print(f"Corrupted files: {len(corrupted_files)} ({results['corruption_rate']:.2f}%)")
        print(f"Sample rate issues: {len(sample_rate_issues)}")
        
        self.validation_results['file_formats'] = results
        return results

    def check_duration_distribution(self) -> Dict:
        '''Automated check 2: Analyse audio duration distribution.
        
        Return: 
            Dict with duration statistics and problematic files'''
        
        print(f'\nAutomated check 2: Duration Distribution Analysis')

        durations = []
        too_short = []
        too_long = []

        for i, file_path in enumerate(self.audio_files):

            try:
                duration = librosa.get_duration(path=file_path)
                durations.append(duration)

                if duration < self.min_duration:
                    too_short.append({'file': file_path.name, 'duration': duration})
                elif duration > self.max_duration:
                    too_long.append({'file': file_path.name, 'duration': duration})
                                    
            except Exception as e:
                print(f"Couldn't get duration for {file_path.name}: {e}")

        durations = np.array(durations)
        
        results = {
            'total_analysed': len(durations),
            'mean_duration': float(np.mean(durations)),
            'median_duration': float(np.median(durations)),
            'std_duration': float(np.std(durations)),
            'min_duration': float(np.min(durations)),
            'max_duration': float(np.max(durations)),
            'too_short': too_short,
            'too_long': too_long,
            'duration_percentiles': {
                '25th': float(np.percentile(durations, 25)),
                '75th': float(np.percentile(durations, 75)),
                '95th': float(np.percentile(durations, 95))
            }
        }
        
        print(f"Duration stats:")
        print(f'    Mean: {results['mean_duration']:.2f}')
        print(f'    Median: {results['median_duration']:.2f}')
        print(f'    Range: {results['min_duration']:.2f}s - {results['max_duration']:.2f}s')
        print(f'    Too short (<{self.min_duration}s) : {len(too_short)}')
        print(f'    Too long (>{self.max_duration}s): {len(too_long)}')

        self.validation_results['durations'] = results
        return results


    def check_for_silence(self, sample_size: int=100) -> Dict:
        '''Automated check 3: Detect silent or nearly-silent audio files.
        
        Args:
            sample_size: Number of files to sample for silence detection
            
        Returns:
            Dict with silence detection results
        '''
        print(f'\nAutomated check 3: Silence Detection(sampling {sample_size} files)')

        sample_files = np.random.choice(
            self.audio_files,
            size =min(sample_size, len(self.audio_files)),
            replace=False
        )
            
        silent_files = []
        rms_values = []

        for i, file_path in enumerate(sample_files):
            try:
                y, sr = librosa.load(file_path, sr=self.target_sample_rate)
                rms = librosa.feature.rms(y=y)[0]
                mean_rms = np.mean(rms)
                rms_values.append(mean_rms)

                if mean_rms < self.silence_threshold:
                    silent_files.append({
                        'file' : file_path.name,
                        'rms': float(mean_rms)
                    })
    
            except Exception as e:
                print(f'Error processing {file_path.name}: {e}')

        rms_values = np.array(rms_values)

        results = {
            'files_sampled': len(sample_files),
            'silent_files': silent_files,
            'silence_rate': len(silent_files) / len(sample_files) * 100,
            'rms_stats': {
                'mean': float(np.mean(rms_values)),
                'median': float(np.median(rms_values)),
                'std': float(np.std(rms_values)),
                'min': float(np.min(rms_values)),
                'max': float(np.max(rms_values))
            }
        }

        print(f"Silent files found: {len(silent_files)} ({results['silence_rate']:.2f}%)")
        print(f"RMS range: {results['rms_stats']['min']:.6f} - {results['rms_stats']['max']:.6f}")
        
        self.validation_results['silence'] = results
        return results
    
    def check_label_distribution(self) -> Dict:
        '''Automated check 4: Analyse label distribution and balance.
        
        Returns:
            Dict with label statistics'''
        
        print("\nAutomated check 4: Label Distribution Analysis")

        if self.labels_df is None:
            print("No labels available - skipping this check")
            return {'error': 'No labels available'}
        
        label_counts = self.labels_df['label'].value_counts()
        total_labels = len(self.labels_df)

        expected_per_class = total_labels / len(label_counts)
        imbalance_ratio = label_counts.max() / label_counts.min()

        labeled_files = set(self.labels_df['filename'])
        audio_filenames = set(f.name for f in self.audio_files)

        missing_labels = audio_filenames - labeled_files
        missing_files = labeled_files - audio_filenames

        results = {
            'total_labels': total_labels,
            'num_classes': len(label_counts),
            'label_counts': label_counts.to_dict(),
            'class_percentages': (label_counts / total_labels * 100).to_dict(),
            'imbalance_ratio': float(imbalance_ratio),
            'missing_labels': list(missing_labels),
            'missing_files': list(missing_files),
            'coverage': len(labeled_files & audio_filenames) / len(audio_filenames) * 100
        }

        print(f"Classes: {len(label_counts)}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f} x (1.0 = perfectly balanced)")
        print(f"Coverage: {results['coverage']:.1f}% of audio files have labels")
        print(f"Class distribution:")
        for label, count in label_counts.head().items():
            percentage = count /total_labels * 100
            print(f"    {label}: {count} ({percentage:.1f}%)")
        
        if len(missing_labels) > 0:
            print(f"    Files missing labels: {len(missing_labels)}")

        self.validation_results['labels'] = results
        return results
    
    def manual_sampling_plan(self) -> Dict:
        '''Manual validation: Generate sampling plan for human review.
        Returns:
            Dict with files selected for manual review'''
        
        print(f"\nManual Validation: Generateing Sampling Plan")

        n_random = min(50, len(self.audio_files))
        random_sample = np.random.choice(self.audio_files, size=n_random, replace=False)

        edge_cases = []

        if 'durations' in self.validation_results:
            duration_results = self.validation_results['durations']

            if duration_results['too_short']:
                edge_cases.extend([f['file'] for f in duration_results['too_short'][:5]])
            if duration_results['too_long']:
                edge_cases.extend([f['file'] for f in duration_results['too_long'][:5]])

        if 'silence' in self.validation_results:
            silence_results = self.validation_results['silence']
            if silence_results['silent_files']:
                edge_cases.extend([f['file'] for f in silence_results['silent_files'][:5]])

        class_samples = {}
        if self.labels_df is not None:
            for label in self.labels_df['label'].unique():
                label_files = self.labels_df[self.labels_df['label'] == label]['filename']
                sample_size = min(10, len(label_files))
                if sample_size > 0:
                    class_samples[label] = np.random.choice(label_files, size=sample_size, replace=False).tolist()

            results = {
                'random_sample': [f.name for f in random_sample],
                'edge_cases': list(set(edge_cases)),
                'class_samples': class_samples,
                'total_for_review': len(random_sample) + len(set(edge_cases)) + sum(len(samples) for samples in class_samples.values())

            }
            print(f"Random sample: {len(random_sample)} files")
            print(f"Edge cases: {len(set(edge_cases))} files")
            print(f"Per-class samples: {sum(len(samples) for samples in class_samples.values())} files")
            print(f"Total for manual review: {results['total_for_review']} files")
                
            self.validation_results['manual_sampling'] = results
            return results
            
    def generate_report(self, output_path: str = "audio_validation_report.json"):
        '''Generate comprehensive validation report
        Args:
            output_path: Path to save the JSON report'''
        print(f"\nGenerating Validation Report....")

        report = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'data_path': str(self.data_path),
                'total_files': len(self.audio_files),
                'validator_version': '1.0.0'

            },
            'results': self.validation_results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"    Report saved to: {output_path}")

        print(f"\nVALIDATION SUMMARY")
        print(f"\nTotal files analysed: {len(self.audio_files)}")

        if 'file_formats' in self.validation_results:
            corruption_rate = self.validation_results['file_formats']['corruption_rate']
            print(f"Corruption rate: {corruption_rate:.2f}%")

        if 'durations' in self.validation_results:
            duration_issues = (
                len(self.validation_results['durations']['too_short']) +
                len(self.validation_results['durations']['too_long'])
            )
            print(f"Duration issues: {duration_issues} files")

        if 'silence' in self.validation_results:
            silence_rate = self.validation_results['silence']['silence_rate']
            print(f"Silence rate: {silence_rate:.2f}%")

        if 'labels' in self.validation_results and 'imbalance_ratio' in self.validation_results['labels']:
            imbalance = self.validation_results['labels']['imbalance_ratio']
            print(f"Class imbalance: {imbalance:.2f}x")

        issues = 0
        if 'file_formats' in self.validation_results:
            if self.validation_results['file_formats']['corruption_rate'] > 5:
                issues += 1

        if 'silence' in self.validation_results:
            if self.validation_results['silence']['silence_rate'] > 10:
                issues += 1
        
        if 'labels' in self.validation_results and 'imbalance_ratio' in self.validation_results['labels']:
            if self.validation_results['labels']['imbalance_ratio'] > 10:
                issues += 1

        if issues == 0:
            print(f"\nQuality: GOOD - Ready for ML training")
        elif issues == 1:
            print(f"\nQuality: FAIR - Consider data cleaning")
        else:
            print(f"\nQuality: POOR - Data cleaning required")

        return report 
    
    def run_full_validation(self):
        '''Run complete validation pipline.'''
        print(f"\nStarting Full Audio Data Validation Pipeline!")
        
        self.discover_files()
        if len(self.audio_files) == 0:
            return 
        
        self.load_labels()
        self.check_file_formats()
        self.check_duration_distribution()
        self.check_for_silence()
        self.check_label_distribution()
        self.manual_sampling_plan()
        
        report = self.generate_report()

        print("\nValidation Complete!")
        print("="*80)
        return report
    
if __name__ == "__main__":
    validator = AudioDataValidator(
        data_path = './data/ESC-50/audio',
        labels_file = './data/ESC-50/meta/esc50.csv'
    )

    report = validator.run_full_validation()
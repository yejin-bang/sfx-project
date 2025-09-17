import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class GroundTruthValidator:
    def __init__(self):
        self.data = {}
        
    def load_annotations(self, json_files):
        """Load all annotation files"""
        loaded_count = 0
        for file_path in json_files:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                video_name = Path(file_path).stem.replace('_ground_truth', '')
                self.data[video_name] = data
                loaded_count += 1
                print(f"✅ Loaded: {video_name} ({len(data['annotations'])} steps)")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                
        return loaded_count
    
    def validate_single_video(self, video_name):
        """Analyze annotations for one video"""
        if video_name not in self.data:
            return None
            
        data = self.data[video_name]
        annotations = data['annotations']
        
        if not annotations:
            return {"error": "No annotations found"}
        
        # Extract basic info
        timestamps = [a['timestamp'] for a in annotations]
        feet = [a['foot'] for a in annotations]
        duration = data['video_info']['duration']
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
        
        # Check for obvious annotation errors (not movement patterns)
        definite_errors = []
        
        # Only flag clearly impossible intervals (likely annotation mistakes)
        for i, interval in enumerate(intervals):
            if interval < 0.1:  # Physically impossible in walking - probably double-clicked
                definite_errors.append({
                    'position': i+1,
                    'timestamp': timestamps[i+1],
                    'interval': interval,
                    'type': 'impossible_speed',
                    'issue': f"Interval too fast: {interval:.3f}s (likely annotation error)"
                })
        
        # Check for potential annotation mistakes (same foot consecutive)
        potential_errors = []
        for i in range(1, len(feet)):
            if feet[i] == feet[i-1]:
                potential_errors.append({
                    'position': i,
                    'timestamp': timestamps[i],
                    'foot': feet[i],
                    'type': 'consecutive_foot',
                    'issue': f"Same foot ({feet[i]}) twice in a row"
                })
        
        # Descriptive statistics 
        step_rate = len(annotations) / duration * 60 if duration > 0 else 0
        
        validation = {
            'video_name': video_name,
            'total_steps': len(annotations),
            'left_steps': len([f for f in feet if f == 'left']),
            'right_steps': len([f for f in feet if f == 'right']),
            'duration': duration,
            'step_rate': step_rate,
            'intervals': intervals,
            'avg_interval': np.mean(intervals) if intervals else 0,
            'std_interval': np.std(intervals) if intervals else 0,
            'min_interval': min(intervals) if intervals else 0,
            'max_interval': max(intervals) if intervals else 0,
            'definite_errors': definite_errors,  # Clear mistakes
            'potential_issues': potential_errors,  # Might be intentional
            'timestamps': timestamps,
            'feet': feet
        }
        
        return validation
    
    def print_validation_report(self, validation):
        """Print descriptive report (not judgmental)"""
        if not validation or 'error' in validation:
            print(f"❌ Cannot analyze: {validation.get('error', 'Unknown error')}")
            return
        
        v = validation
        print(f"\n{'='*60}")
        print(f"GROUND TRUTH ANALYSIS: {v['video_name']}")
        print(f"{'='*60}")
        
        # Descriptive statistics
        print(f"📊 DESCRIPTIVE STATISTICS:")
        print(f"   Duration: {v['duration']:.2f} seconds")
        print(f"   Total steps: {v['total_steps']}")
        print(f"   Left steps: {v['left_steps']} ({v['left_steps']/v['total_steps']*100:.1f}%)")
        print(f"   Right steps: {v['right_steps']} ({v['right_steps']/v['total_steps']*100:.1f}%)")
        print(f"   Step rate: {v['step_rate']:.1f} steps/minute")
        
        # Timing patterns (descriptive)
        print(f"\n⏱️  TIMING PATTERNS:")
        if v['intervals']:
            print(f"   Average interval: {v['avg_interval']:.3f}s")
            print(f"   Variability (std): {v['std_interval']:.3f}s")
            print(f"   Range: {v['min_interval']:.3f}s - {v['max_interval']:.3f}s")
            
            # Describe variability neutrally
            cv = v['std_interval'] / v['avg_interval'] if v['avg_interval'] > 0 else 0
            print(f"   Consistency (CV): {cv:.2f}")
            
            # Describe intervals neutrally
            short_intervals = len([i for i in v['intervals'] if i < 0.5])
            long_intervals = len([i for i in v['intervals'] if i > 1.5])
            print(f"   Quick intervals (<0.5s): {short_intervals}")
            print(f"   Longer intervals (>1.5s): {long_intervals}")
        
        # Error analysis (only clear mistakes)
        print(f"\n🔍 ANNOTATION QUALITY CHECK:")
        
        # Definite errors (likely mistakes)
        if not v['definite_errors']:
            print(f"   ✅ No obvious annotation errors detected")
        else:
            print(f"   ❌ Found {len(v['definite_errors'])} likely annotation errors:")
            for error in v['definite_errors']:
                print(f"      • {error['issue']}")
        
        # Potential issues (might be intentional)
        if not v['potential_issues']:
            print(f"   ✅ Perfect left/right alternation")
        else:
            print(f"   ⚠️  Found {len(v['potential_issues'])} potential patterns to review:")
            for issue in v['potential_issues'][:3]:  # Show first 3
                print(f"      • {issue['issue']} at {issue['timestamp']:.3f}s")
            if len(v['potential_issues']) > 3:
                print(f"      • ... and {len(v['potential_issues']) - 3} more")
            print(f"      (Note: These might be intentional movement patterns)")
        
        # Data readiness assessment
        print(f"\n📝 DATA READINESS FOR AI TESTING:")
        error_count = len(v['definite_errors'])
        if error_count == 0:
            print(f"   ✅ READY: No clear annotation errors found")
            print(f"   ✅ This data can be used to test MediaPipe accuracy")
        elif error_count <= 2:
            print(f"   ⚠️  MOSTLY READY: {error_count} errors found, recommend quick review")
        else:
            print(f"   ❌ NEEDS CLEANUP: {error_count} errors found, fix before AI testing")
        
        print(f"{'='*60}\n")
    
    def create_timeline_plot(self, validation):
        """Create visual timeline showing the footstep patterns"""
        if not validation or 'error' in validation:
            return
        
        v = validation
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Timeline with L/R markers
        left_times = [t for t, f in zip(v['timestamps'], v['feet']) if f == 'left']
        right_times = [t for t, f in zip(v['timestamps'], v['feet']) if f == 'right']
        
        ax1.scatter(left_times, [1]*len(left_times), color='green', s=100, alpha=0.7, label='Left foot', marker='o')
        ax1.scatter(right_times, [0]*len(right_times), color='red', s=100, alpha=0.7, label='Right foot', marker='s')
        
        # Mark potential issues 
        for issue in v['potential_issues']:
            foot_y = 1 if issue['foot'] == 'left' else 0
            ax1.scatter(issue['timestamp'], foot_y, color='orange', s=200, alpha=0.5, marker='x')
        
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlim(0, v['duration'])
        ax1.set_ylabel('Foot')
        ax1.set_title(f'Footstep Timeline: {v["video_name"]} ({v["total_steps"]} steps)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Right', 'Left'])
        
        # Plot 2: Step intervals (showing patterns)
        if v['intervals']:
            ax2.plot(v['timestamps'][1:], v['intervals'], 'b-o', markersize=4, alpha=0.7)
            ax2.axhline(y=np.mean(v['intervals']), color='r', linestyle='--', alpha=0.7, 
                       label=f'Average: {np.mean(v["intervals"]):.3f}s')
            
            # Highlight only definite errors
            for error in v['definite_errors']:
                ax2.scatter(error['timestamp'], error['interval'], color='red', s=100, marker='x', zorder=5)
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Interval (seconds)')
            ax2.set_title('Step Intervals Over Time (Natural Variation Expected)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, v['duration'])
        
        plt.tight_layout()
        plt.show()
    
    def validate_all_videos(self):
        """Analyze all loaded videos for AI testing readiness"""
        if not self.data:
            print("❌ No data loaded!")
            return
        
        print(f"\n🔍 ANALYZING {len(self.data)} VIDEOS FOR AI TESTING...")
        
        all_validations = {}
        summary_stats = []
        
        for video_name in self.data:
            validation = self.validate_single_video(video_name)
            if validation and 'error' not in validation:
                all_validations[video_name] = validation
                self.print_validation_report(validation)
                
                # Collect for summary
                summary_stats.append({
                    'video': video_name,
                    'steps': validation['total_steps'],
                    'rate': validation['step_rate'],
                    'definite_errors': len(validation['definite_errors']),
                    'potential_issues': len(validation['potential_issues'])
                })
        
        # Overall summary for AI testing
        if summary_stats:
            print(f"\n{'='*60}")
            print(f"OVERALL SUMMARY FOR AI TESTING")
            print(f"{'='*60}")
            
            total_steps = sum(s['steps'] for s in summary_stats)
            total_errors = sum(s['definite_errors'] for s in summary_stats)
            total_issues = sum(s['potential_issues'] for s in summary_stats)
            
            print(f"📊 Total annotated steps: {total_steps}")
            print(f"📊 Videos analyzed: {len(summary_stats)}")
            print(f"📊 Clear annotation errors: {total_errors}")
            print(f"📊 Potential review items: {total_issues}")
            
            # AI testing readiness
            error_rate = total_errors / total_steps if total_steps > 0 else 0
            print(f"\n🤖 AI TESTING READINESS:")
            if error_rate < 0.02:  # Less than 2% clear errors
                print(f"✅ EXCELLENT: Data is ready for MediaPipe testing")
                print(f"✅ Error rate: {error_rate*100:.1f}% - This is high-quality ground truth")
            elif error_rate < 0.05:  # Less than 5% clear errors
                print(f"⚠️  GOOD: Mostly ready, minor cleanup recommended")
                print(f"⚠️  Error rate: {error_rate*100:.1f}% - Still usable for testing")
            else:
                print(f"❌ NEEDS WORK: Fix clear errors before AI testing")
                print(f"❌ Error rate: {error_rate*100:.1f}% - Too many mistakes for reliable testing")
        
        return all_validations

# Usage function
def validate_for_ai_testing(json_files, show_plots=True):
    """Main function to validate ground truth for AI testing"""
    validator = GroundTruthValidator()
    
    # Load files
    loaded = validator.load_annotations(json_files)
    if loaded == 0:
        print("❌ No valid files loaded!")
        return None
    
    # Validate all
    validations = validator.validate_all_videos()
    
    # Show plots if requested
    if show_plots and validations:
        for video_name, validation in validations.items():
            validator.create_timeline_plot(validation)
    
    return validations

if __name__ == "__main__":
    # Get file paths from user
    print("Enter your JSON file paths (press Enter after each, empty line to finish):")
    user_files = []
    while True:
        file_path = input().strip()
        if not file_path:
            break
        user_files.append(file_path)
    
    if user_files:
        validate_for_ai_testing(user_files)
    else:
        print("No files provided for validation.")
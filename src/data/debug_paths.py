# Add this debugging code to your validator class or run it separately

import os
from pathlib import Path

def debug_find_audio_files():
    """Debug function to find where your audio files actually are"""
    
    print("🔍 DEBUGGING: Finding your ESC-50 audio files")
    print("="*50)
    
    # Check different possible paths
    possible_paths = [
        '../data/ESC-50/audio',           # Your current attempt
        '../data/ESC-50',                 # Maybe audio files are directly here
        './data/ESC-50/audio',            # Maybe you need ./
        'data/ESC-50/audio',              # Maybe no ../
        '../ESC-50/audio',                # Maybe ESC-50 is in parent directory
        './ESC-50/audio',                 # Maybe ESC-50 is in current directory
    ]
    
    print("1. Checking if paths exist:")
    for path in possible_paths:
        exists = Path(path).exists()
        print(f"   {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    print("\n2. Current working directory:")
    print(f"   {os.getcwd()}")
    
    print("\n3. Contents of current directory:")
    current_dir = Path('.')
    for item in current_dir.iterdir():
        print(f"   {'📁' if item.is_dir() else '📄'} {item.name}")
    
    print("\n4. Looking for 'data' folder:")
    data_paths = [
        './data',
        '../data', 
        '../../data'
    ]
    
    for data_path in data_paths:
        if Path(data_path).exists():
            print(f"   ✅ Found data folder at: {data_path}")
            print(f"   Contents of {data_path}:")
            for item in Path(data_path).iterdir():
                print(f"      {'📁' if item.is_dir() else '📄'} {item.name}")
                
                # If we find ESC-50, look inside it
                if item.name == 'ESC-50' and item.is_dir():
                    print(f"   ✅ Found ESC-50 folder!")
                    print(f"   Contents of ESC-50:")
                    for esc_item in item.iterdir():
                        print(f"      {'📁' if esc_item.is_dir() else '📄'} {esc_item.name}")
                        
                        # Check audio folder
                        if esc_item.name == 'audio' and esc_item.is_dir():
                            print(f"   ✅ Found audio folder!")
                            print(f"   Sample audio files:")
                            audio_files = list(esc_item.glob('*.wav'))[:5]  # First 5 .wav files
                            for audio_file in audio_files:
                                print(f"      🎵 {audio_file.name}")
                            print(f"   Total .wav files: {len(list(esc_item.glob('*.wav')))}")
            break
    else:
        print("   ❌ No data folder found in common locations")
    
    print("\n5. Manual search for any .wav files:")
    # Search for .wav files in current and parent directories
    search_dirs = ['.', '..', '../..']
    for search_dir in search_dirs:
        wav_files = list(Path(search_dir).glob('**/*.wav'))
        if wav_files:
            print(f"   Found .wav files in {search_dir}:")
            for wav_file in wav_files[:5]:  # Show first 5
                print(f"      🎵 {wav_file}")
            if len(wav_files) > 5:
                print(f"      ... and {len(wav_files) - 5} more files")

# Run the debug function
debug_find_audio_files()
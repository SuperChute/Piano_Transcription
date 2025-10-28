import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os

def load_audio_file(filepath, target_sample_rate=None):
    """
    Load an audio file (mp3, m4a, etc.) and convert to numpy array
    Returns: (audio_data, sample_rate)
    
    Args:
        filepath: Path to audio file
        target_sample_rate: If specified, resample to this rate (e.g., 44100)
    """
    # Load audio file using pydub
    if filepath.endswith('.mp3'):
        audio = AudioSegment.from_mp3(filepath)
    elif filepath.endswith('.m4a'):
        audio = AudioSegment.from_file(filepath, format='m4a')
    else:
        audio = AudioSegment.from_file(filepath)
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Resample if target sample rate is specified
    if target_sample_rate and audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
    
    # Get sample rate
    sample_rate = audio.frame_rate
    
    # Convert to numpy array (normalized to -1 to 1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / (2**15)  # Normalize 16-bit audio
    
    return samples, sample_rate

def combine_audio_signals(file_list):
    """
    Load multiple audio files and combine them by summing their waveforms
    Returns: (combined_audio, sample_rate, individual_signals)
    """
    all_signals = []
    sample_rates = []
    
    for filepath in file_list:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping...")
            continue
            
        samples, sr = load_audio_file(filepath)
        all_signals.append(samples)
        sample_rates.append(sr)
    
    if not all_signals:
        raise ValueError("No valid audio files loaded!")
    
    # Check if all sample rates are the same
    if len(set(sample_rates)) > 1:
        print(f"Warning: Different sample rates detected: {set(sample_rates)}")
        print("Using the first file's sample rate as reference")
    
    sample_rate = sample_rates[0]
    
    # Pad signals to same length (use the longest one)
    max_length = max(len(sig) for sig in all_signals)
    padded_signals = []
    
    for sig in all_signals:
        if len(sig) < max_length:
            padded = np.pad(sig, (0, max_length - len(sig)), mode='constant')
            padded_signals.append(padded)
        else:
            padded_signals.append(sig)
    
    # Combine by summing and normalize
    combined = np.sum(padded_signals, axis=0)
    combined = combined / len(padded_signals)  # Average to prevent clipping
    
    return combined, sample_rate, padded_signals

def plot_waveforms(file_list, show_individual=True, show_combined=True):
    """
    Plot audio waveforms from multiple files
    """
    combined, sample_rate, individual_signals = combine_audio_signals(file_list)
    
    # Create time axis in seconds
    time_combined = np.arange(len(combined)) / sample_rate
    
    # Determine subplot layout
    num_plots = 0
    if show_individual and len(file_list) > 1:
        num_plots += len(file_list)
    if show_combined:
        num_plots += 1
    
    if num_plots == 0:
        num_plots = 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot individual signals
    if show_individual and len(file_list) > 1:
        for i, (filepath, signal) in enumerate(zip(file_list, individual_signals)):
            time = np.arange(len(signal)) / sample_rate
            axes[plot_idx].plot(time, signal, linewidth=0.5)
            axes[plot_idx].set_title(f'Individual: {os.path.basename(filepath)}')
            axes[plot_idx].set_xlabel('Time (seconds)')
            axes[plot_idx].set_ylabel('Amplitude')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_ylim(-1, 1)
            plot_idx += 1
    
    # Plot combined signal
    if show_combined:
        axes[plot_idx].plot(time_combined, combined, linewidth=0.5, color='red')
        if len(file_list) > 1:
            axes[plot_idx].set_title(f'Combined Waveform ({len(file_list)} notes)')
        else:
            axes[plot_idx].set_title(f'Waveform: {os.path.basename(file_list[0])}')
        axes[plot_idx].set_xlabel('Time (seconds)')
        axes[plot_idx].set_ylabel('Amplitude')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()
    
    return combined, sample_rate

# Example usage
if __name__ == "__main__":
    # Example 1: Multiple notes combined into a chord
    test_files = [
         "pure_notes/C4_real.m4a",
            "pure_notes/E4_real.m4a",
            "pure_notes/G4_real.m4a",
    ]
    
    # Example 2: Single note
    # test_files = ["pure_notes/c5.mp3"]
    
    # Example 3: Already a chord
    # test_files = ["pure_notes/c4_e4_g4_chord.mp3"]
    
    print(f"Loading {len(test_files)} audio file(s)...")
    
    try:
        # Plot with both individual and combined waveforms
        combined_audio, sr = plot_waveforms(
            test_files, 
            show_individual=True,  # Set to False to hide individual plots
            show_combined=True
        )
        
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(combined_audio) / sr:.2f} seconds")
        print(f"Combined signal shape: {combined_audio.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
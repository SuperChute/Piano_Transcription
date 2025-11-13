import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, chord, midi, pitch
import re

# --- Configuration and Utility Functions ---

def frequency_to_midi(freq):
    """
    Convert frequency in Hz to MIDI note number using the A440 standard.
    """
    if freq <= 0:
        return None
    midi_note = (12 * np.log2(freq / 440)) + 69 
    return round(midi_note)

def midi_to_note_name(midi_note):
    """
    Convert MIDI note number to musical note name (e.g., 60 -> C4).
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note_name = notes[midi_note % 12]
    return f"{note_name}{octave}"


# --- Audio Preprocessing ---

def trim_to_fundamental(signal, sr=44100, skip_oscillations=10000, keep_oscillations=20000):
    """
    Trim audio to capture only the fundamental frequency by:
    1. Finding the maximum amplitude (when note is struck)
    2. Skipping 'skip_oscillations' from that point (removes key strike noise)
    3. Keeping 'keep_oscillations' of clean audio
    4. Discarding everything else
    
    Args:
        signal: Audio signal array
        sr: Sample rate
        skip_oscillations: Number of oscillations to skip after max amplitude
        keep_oscillations: Number of oscillations to keep for analysis
    
    Returns:
        Trimmed signal array
    """
    # Find the index of maximum amplitude
    max_idx = np.argmax(np.abs(signal))
    
    print(f"  Max amplitude at sample {max_idx} ({max_idx/sr:.4f} seconds)")
    
    # Calculate start index: max + skip_oscillations samples
    start_idx = max_idx + skip_oscillations
    
    # Calculate end index: start + keep_oscillations samples
    end_idx = start_idx + keep_oscillations
    
    # Ensure we don't go out of bounds
    if start_idx >= len(signal):
        print(f"  Warning: Start index {start_idx} exceeds signal length {len(signal)}")
        print(f"  Using last {keep_oscillations} samples instead")
        start_idx = max(0, len(signal) - keep_oscillations)
        end_idx = len(signal)
    elif end_idx > len(signal):
        print(f"  Warning: End index {end_idx} exceeds signal length {len(signal)}")
        print(f"  Trimming to available samples: {len(signal) - start_idx}")
        end_idx = len(signal)
    
    trimmed = signal[start_idx:end_idx]
    
    print(f"  Trimmed: {len(trimmed)} samples ({len(trimmed)/sr:.4f} seconds)")
    print(f"  Original: {len(signal)} samples → Kept: {len(trimmed)} samples")
    
    return trimmed


def load_and_trim_audio(audio_file, sr=44100, skip_oscillations=10000, keep_oscillations=20000):
    """
    Load an audio file and trim it to capture only the fundamental frequency.
    
    Args:
        audio_file: Path to audio file
        sr: Sample rate
        skip_oscillations: Number of oscillations to skip after max amplitude
        keep_oscillations: Number of oscillations to keep
    
    Returns:
        Trimmed signal array, sample rate
    """
    # Load audio
    signal, sr_loaded = librosa.load(audio_file, sr=sr, mono=True)
    
    # Trim to fundamental
    trimmed_signal = trim_to_fundamental(signal, sr, skip_oscillations, keep_oscillations)
    
    return trimmed_signal, sr


def load_and_mix_signals(audio_files, sr=44100):
    """
    Load multiple mono files, align their lengths, and mix them into one signal 
    to simulate a chord.
    
    Returns: mixed signal (np.array), sample rate (int)
    """
    signals = []
    for path in audio_files:
        sig, sr_loaded = librosa.load(path, sr=sr, mono=True)
        signals.append(sig)

    max_len = max(len(s) for s in signals)
    padded = [np.pad(s, (0, max_len - len(s))) for s in signals]

    mixed = np.sum(np.stack(padded, axis=0), axis=0)
    return mixed, sr 

def save_mixed_signal_as_mp3(mixed_signal, sr, output_file='mixed_output.mp3'):
    """
    Save the mixed signal as an MP3 file.
    
    Args:
        mixed_signal: Mixed audio signal array
        sr: Sample rate
        output_file: Path to save MP3 file
    """
    import soundfile as sf
    from pydub import AudioSegment
    import os
    
    # First save as WAV (temporary)
    temp_wav = 'temp_mixed.wav'
    sf.write(temp_wav, mixed_signal, sr)
    
    # Convert WAV to MP3
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(output_file, format='mp3', bitrate='192k')
    
    # Clean up temporary file
    os.remove(temp_wav)
    
    print(f"\n✓ Mixed audio saved: {output_file}")


# --- Frequency Binning ---

def create_frequency_bins(note_range=['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'], 
                          num_harmonics=10):
    """
    Create frequency bins centered around each note's fundamental and harmonics.
    
    Args:
        note_range: List of note names to cover
        num_harmonics: Number of harmonics to include per note (including fundamental)
    
    Returns:
        bin_centers: Array of frequency bin centers (Hz)
    """
    bin_centers = []
    
    for note_name in note_range:
        p = pitch.Pitch(note_name)
        fundamental = p.frequency
        
        # Add fundamental + harmonics
        for harmonic in range(1, num_harmonics + 1):
            bin_centers.append(fundamental * harmonic)
    
    # Sort and remove duplicates
    bin_centers = sorted(set(bin_centers))
    
    return np.array(bin_centers)


def quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width=10.0):
    """
    Quantize full FFT spectrum into predefined frequency bins.
    
    Args:
        freqs: Full FFT frequency array
        magnitude: Full FFT magnitude array
        bin_centers: Target frequency bins (Hz)
        bin_width: Width of each bin in Hz (± this value around center)
    
    Returns:
        quantized_magnitude: Array of magnitudes at each bin center
    """
    quantized_magnitude = np.zeros(len(bin_centers))
    
    for i, fc in enumerate(bin_centers):
        # Find all FFT bins within ± bin_width of this center frequency
        mask = (freqs >= fc - bin_width) & (freqs <= fc + bin_width)
        
        # Sum magnitudes in this range
        if np.any(mask):
            quantized_magnitude[i] = np.sum(magnitude[mask])
    
    return quantized_magnitude


# --- Basis Matrix Construction --- 

def filename_stem_to_pitchname(stem: str) -> str:
    """
    Convert filename stems like 'dflat4', 'gsharp3', 'Db4', 'C#5' to canonical
    music21-friendly names like 'Db4', 'G#3', etc.
    """
    s = stem.strip().lower()
    # normalize words to accidentals
    s = s.replace('sharp', '#').replace('flat', 'b')

    # common patterns: c4, db4, g#3
    m = re.match(r'^([a-g])([b#]?)(\d+)$', s)
    if not m:
        # fallback: leave as-is
        return stem

    letter, acc, octv = m.groups()
    return letter.upper() + acc + octv

def build_basis_matrix(note_files, bin_centers, sr=44100, bin_width=10.0, 
                      use_trimming=True, skip_oscillations=10000, keep_oscillations=20000,
                      plot_first_n=0):
    """
    Load each pure note, compute FFT, quantize to bins, and stack into matrix A.
    
    Args:
        note_files: List of paths to pure note audio files
        bin_centers: Frequency bins to use
        sr: Sample rate
        bin_width: Bin width for quantization
        use_trimming: Whether to trim audio to fundamental frequency
        skip_oscillations: Number of oscillations to skip after max amplitude
        keep_oscillations: Number of oscillations to keep
        plot_first_n: Number of first notes to plot before/after trimming (0 to disable)
    
    Returns:
        A: Basis matrix (num_bins × num_notes)
        note_names: List of note names corresponding to columns
    """
    basis_vectors = []
    note_names = []
    
    print("\n=== Building Basis Matrix ===")
    print(f"Trimming enabled: {use_trimming}")
    if use_trimming:
        print(f"Skip {skip_oscillations} samples, Keep {keep_oscillations} samples")
    
    for idx, note_file in enumerate(note_files):
        # Extract note name from filename
        stem = note_file.split('/')[-1].split('.')[0]
        note_name = filename_stem_to_pitchname(stem)   # e.g., 'dflat4' -> 'Db4'

        note_names.append(note_name)
        
        print(f"\nProcessing: {note_name}")
        
        # Load audio (with optional trimming)
        if use_trimming:
            # Load original signal first for comparison plotting
            original_signal, _ = librosa.load(note_file, sr=sr, mono=True)
            
            # Trim the signal
            signal = trim_to_fundamental(original_signal, sr, skip_oscillations, keep_oscillations)
            
            # Plot comparison for first N notes
            if plot_first_n > 0 and idx < plot_first_n:
                plot_trimming_comparison(original_signal, signal, sr, note_name)
        else:
            signal, _ = librosa.load(note_file, sr=sr, mono=True)
        
        # Compute FFT
        ft = np.fft.rfft(signal)
        magnitude = np.abs(ft)
        freqs = np.fft.rfftfreq(len(signal), 1/sr)
        
        # Quantize to bins
        quantized = quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width)
        
        # Normalize
        max_val = np.max(quantized)
        quantized = quantized / max_val if max_val > 0 else quantized
        
        basis_vectors.append(quantized)
        print(f"  Max quantized bin = {max_val:.2f}")
    
    # Stack as columns
    A = np.column_stack(basis_vectors)
    
    return A, note_names


# --- LINEAR ALGEBRA SOLVER ---

def detect_notes(A, b, note_names, threshold=0.3):
    """
    Solve Ax = b to detect which notes are present in the mixed signal.
    
    Args:
        A: Basis matrix (num_bins × num_notes)
        b: Quantized spectrum of mixed signal (num_bins,)
        note_names: List of note names corresponding to columns of A
        threshold: Minimum weight to consider a note "present"
    
    Returns:
        detected_notes: List of (note_name, weight) tuples for detected notes
        weights: Full weight vector (solution to Ax = b)
    """
    print("\n=== Solving Ax = b ===")
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    
    # Find weights such that A*weights ≈ b
    # This tells us which combination of notes (columns of A) creates signal b
    solution = np.linalg.lstsq(A, b, rcond=None)
    
    weights = abs(solution[0])       # The weight for each note
    error = solution[1]             # How far off the solution is (residual)
    
    # Print Results
    print(f"\nSolution (weights):")
    for note, weight in zip(note_names, weights):
        print(f"  {note}: {weight:.4f}")
    
    if len(error) > 0:
        print(f"\nResidual error: {error[0]:.6f}")
    
    # Determine which notes are "present"
    # Only notes with weight >= threshold are considered detected
    detected_notes = []
    for note, weight in zip(note_names, weights):
        if weight >= threshold:
            detected_notes.append((note, weight))
    
    return detected_notes, weights


def create_midi_from_detected_notes(detected_notes, output_file='detected_chord.mid', duration=2.0):
    """
    Create a MIDI file from detected notes.
    
    Args:
        detected_notes: List of (note_name, weight) tuples
        output_file: Path to save MIDI file
        duration: Note duration in quarter notes
    """
    if not detected_notes:
        print("No notes detected to create MIDI.")
        return
    
    s = stream.Stream()
    
    # Convert note names to MIDI pitches
    midi_pitches = []
    for note_name, weight in detected_notes:
        try:
            p = pitch.Pitch(note_name)
            midi_pitches.append(p.midi)
        except:
            print(f"Warning: Could not convert {note_name} to MIDI pitch")
    
    if midi_pitches:
        c = chord.Chord(midi_pitches)
        c.duration.quarterLength = duration
        s.append(c)
        
        s.write('midi', fp=output_file)
        print(f"\n✓ MIDI file saved: {output_file}")
    else:
        print("No valid MIDI notes to write.")


# --- Visualization ---

def plot_basis_matrix(A, note_names, bin_centers):
    """
    Visualize the basis matrix as a heatmap.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(A, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Normalized Magnitude')
    plt.xlabel('Note')
    plt.ylabel('Frequency Bin Index')
    plt.title('Basis Matrix A (Frequency Bins × Notes)')
    plt.xticks(range(len(note_names)), note_names, rotation=45)
    
    plt.yticks(range(len(bin_centers)), [f"{f:.0f} Hz" for f in bin_centers])
    
    plt.tight_layout()
    plt.show()


def plot_quantized_spectrum(bin_centers, quantized_magnitude, title="Quantized FFT"):
    """
    Plot the quantized frequency spectrum.
    """
    plt.figure(figsize=(12, 4))
    plt.stem(bin_centers, quantized_magnitude, basefmt=' ')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_detection_results(note_names, weights, detected_notes, threshold):
    """
    Visualize the weight vector and detected notes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Bar chart of all weights
    colors = ['green' if w >= threshold else 'gray' for w in weights]
    ax1.bar(range(len(note_names)), weights, color=colors, alpha=0.7)
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    ax1.set_xlabel('Note')
    ax1.set_ylabel('Weight')
    ax1.set_title('Solution Vector (Note Weights)')
    ax1.set_xticks(range(len(note_names)))
    ax1.set_xticklabels(note_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Detected notes only
    if detected_notes:
        detected_note_names = [n for n, w in detected_notes]
        detected_weights = [w for n, w in detected_notes]
        ax2.bar(range(len(detected_notes)), detected_weights, color='green', alpha=0.7)
        ax2.set_xlabel('Detected Note')
        ax2.set_ylabel('Weight')
        ax2.set_title(f'Detected Notes (≥ {threshold})')
        ax2.set_xticks(range(len(detected_notes)))
        ax2.set_xticklabels(detected_note_names, rotation=45)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No notes detected', ha='center', va='center', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def plot_mixed_signal(mixed_signal, sr, title="Mixed Audio Signal"):
    """
    Plot the time-domain waveform of the mixed signal.
    
    Args:
        mixed_signal: Mixed audio signal array
        sr: Sample rate
        title: Plot title
    """
    time = np.arange(len(mixed_signal)) / sr
    
    plt.figure(figsize=(14, 5))
    plt.plot(time, mixed_signal, linewidth=0.5, alpha=0.8)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_trimming_comparison(original_signal, trimmed_signal, sr, note_name):
    """
    Visualize the effect of trimming on the audio signal.
    Shows where the max amplitude was found and what was kept.
    
    Args:
        original_signal: Original audio signal
        trimmed_signal: Trimmed audio signal
        sr: Sample rate
        note_name: Name of the note
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Find max amplitude location for visualization
    max_idx = np.argmax(np.abs(original_signal))
    
    # Original signal
    time_orig = np.arange(len(original_signal)) / sr
    ax1.plot(time_orig, original_signal, linewidth=0.5, alpha=0.8, color='blue')
    
    # Mark the max amplitude point
    ax1.axvline(x=max_idx/sr, color='red', linestyle='--', linewidth=2, 
                label=f'Max Amplitude (sample {max_idx})')
    ax1.scatter([max_idx/sr], [original_signal[max_idx]], color='red', s=100, zorder=5)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'BEFORE Trimming: {note_name} ({len(original_signal)} samples, {len(original_signal)/sr:.3f}s)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Trimmed signal
    time_trim = np.arange(len(trimmed_signal)) / sr
    ax2.plot(time_trim, trimmed_signal, linewidth=0.5, alpha=0.8, color='green')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title(f'AFTER Trimming: {note_name} ({len(trimmed_signal)} samples, {len(trimmed_signal)/sr:.3f}s) - Clean Fundamental', 
                  fontsize=14, fontweight='bold', color='green')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations
    reduction_pct = (1 - len(trimmed_signal)/len(original_signal)) * 100
    ax2.text(0.02, 0.98, f'Removed {reduction_pct:.1f}% of signal\nKept clean fundamental frequency', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def trim_mixed_signal(mixed_signal, sr=44100, skip_oscillations=30000, keep_oscillations=50000):
    """
    Trim a mixed signal to focus on the steady-state portion.
    Useful for removing initial transients from chord recordings.
    
    Args:
        mixed_signal: Mixed audio signal array
        sr: Sample rate
        skip_oscillations: Samples to skip after max amplitude
        keep_oscillations: Samples to keep for analysis
    
    Returns:
        Trimmed signal array
    """
    print("\n=== Trimming Mixed Signal ===")
    trimmed = trim_to_fundamental(mixed_signal, sr, skip_oscillations, keep_oscillations)
    return trimmed

def detect_onsets(signal, sr=44100, hop_length=100, 
                  pre_max=20, post_max=20, pre_avg=100, post_avg=100,
                  delta=0.2, wait=0, backtrack=False, visualize=True):
    """
    Detect note onsets using advanced peak-picking parameters (from MIR tutorial).
    
    Args:
        signal: Audio signal array
        sr: Sample rate
        hop_length: Number of samples between successive frames (smaller = more precise)
        pre_max: Number of frames before peak to consider for max filtering
        post_max: Number of frames after peak to consider for max filtering
        pre_avg: Number of frames before peak for moving average
        post_avg: Number of frames after peak for moving average
        delta: Threshold for peak detection (lower = more sensitive)
        wait: Minimum number of frames between peaks
        backtrack: Whether to backtrack to find precise onset location
        visualize: Whether to plot the onset detection
    
    Returns:
        onset_boundaries: Sample indices including 0 and len(signal)
        onset_times: Time in seconds of detected onsets (including 0 and end)
        onset_env: Onset strength envelope
    """
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
    
    print(f"\n=== Advanced Onset Detection ===")
    print(f"Onset envelope shape: {onset_env.shape}")
    print(f"Hop length: {hop_length} samples ({hop_length/sr:.4f}s)")
    
    # Detect onsets using advanced peak picking
    onset_samples = librosa.onset.onset_detect(
        y=signal,
        sr=sr,
        units='samples',
        hop_length=hop_length,
        backtrack=backtrack,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait
    )
    
    # Pad with beginning and end of signal
    onset_boundaries = np.concatenate([[0], onset_samples, [len(signal)]])
    
    # Convert to time
    onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
    
    print(f"Detected {len(onset_samples)} onsets")
    print(f"Onset sample indices: {onset_samples}")
    print(f"Onset times (seconds): {onset_times}")
    
    if visualize:
        plot_onset_detection(signal, sr, onset_times, onset_env, hop_length)
    
    return onset_boundaries, onset_times, onset_env


def plot_onset_detection(signal, sr, onset_times, onset_env, hop_length):
    """
    Visualize the onset detection results.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    
    # Plot waveform with onset markers
    time = np.arange(len(signal)) / sr
    ax1.plot(time, signal, linewidth=0.5, alpha=0.7, color='blue')
    
    for onset_time in onset_times:
        ax1.axvline(x=onset_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Audio Waveform with Detected Onsets', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot onset strength envelope
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    ax2.plot(times, onset_env, linewidth=1.5, color='green', label='Onset Strength')
    
    for onset_time in onset_times:
        ax2.axvline(x=onset_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Onset Strength', fontsize=12)
    ax2.set_title('Onset Strength Envelope', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def segment_audio_by_onsets(signal, sr, onset_boundaries):
    """
    Split audio signal into segments based on detected onset boundaries.
    This version uses the onset_boundaries directly (which includes 0 and len(signal)).
    
    Args:
        signal: Audio signal array
        sr: Sample rate
        onset_boundaries: Array of onset sample indices (including start=0 and end=len(signal))
    
    Returns:
        segments: List of audio segments
        segment_info: List of (start_time, end_time) tuples for each segment
    """
    segments = []
    segment_info = []
    
    print(f"\n=== Segmenting Audio ===")
    print(f"Number of boundaries: {len(onset_boundaries)}")
    print(f"Will create {len(onset_boundaries) - 1} segments")
    
    for i in range(len(onset_boundaries) - 1):
        start_sample = onset_boundaries[i]
        end_sample = onset_boundaries[i + 1]
        
        # Extract segment
        segment = signal[start_sample:end_sample]
        segments.append(segment)
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        segment_info.append((start_time, end_time))
        
        print(f"Segment {i+1}: {start_time:.3f}s to {end_time:.3f}s "
              f"({len(segment)} samples, {len(segment)/sr:.3f}s)")
    
    return segments, segment_info


def analyze_sequential_notes(audio_file, A, note_names, bin_centers, 
                             sr=44100, bin_width=10.0, detection_threshold=0.35,
                             hop_length=100, pre_max=20, post_max=20, 
                             pre_avg=100, post_avg=100, delta=0.2, wait=0,
                             trim_segments=True, skip_oscillations=1000, 
                             keep_oscillations=20000):
    """
    Main function to analyze sequential notes in an audio file using advanced onset detection.
    
    Args:
        audio_file: Path to audio file with sequential notes
        A: Basis matrix from pure notes
        note_names: List of note names
        bin_centers: Frequency bins
        sr: Sample rate
        bin_width: Bin width for FFT quantization
        detection_threshold: Threshold for note detection
        hop_length: Hop length for onset detection (smaller = more precise)
        pre_max, post_max: Frames before/after for max filtering
        pre_avg, post_avg: Frames before/after for moving average
        delta: Peak detection threshold (lower = more sensitive)
        wait: Minimum frames between peaks
        trim_segments: Whether to trim each segment to fundamental
        skip_oscillations: Samples to skip when trimming
        keep_oscillations: Samples to keep when trimming
    
    Returns:
        results: List of dictionaries with detection results for each segment
        onset_times: Array of onset times
    """
    print("="*70)
    print("SEQUENTIAL NOTE ANALYSIS")
    print("="*70)
    
    # Load audio
    print(f"\nLoading audio: {audio_file}")
    signal, _ = librosa.load(audio_file, sr=sr, mono=True)
    print(f"Loaded {len(signal)} samples ({len(signal)/sr:.3f}s)")
    
    # Detect onsets with advanced parameters
    onset_boundaries, onset_times, onset_env = detect_onsets(
        signal, sr,
        hop_length=hop_length,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
        backtrack=False,
        visualize=True
    )
    
    # Segment audio using boundaries
    segments, segment_info = segment_audio_by_onsets(signal, sr, onset_boundaries)
    
    # Analyze each segment
    results = []
    
    print("\n" + "="*70)
    print("ANALYZING EACH SEGMENT")
    print("="*70)
    
    for i, (segment, (start_time, end_time)) in enumerate(zip(segments, segment_info)):
        print(f"\n{'='*70}")
        print(f"SEGMENT {i+1}/{len(segments)}: {start_time:.3f}s - {end_time:.3f}s")
        print(f"{'='*70}")
        
        # Optional: Trim segment to fundamental
        if trim_segments and len(segment) > skip_oscillations + keep_oscillations:
            print(f"Trimming segment {i+1}...")
            segment = trim_to_fundamental(segment, sr, skip_oscillations, keep_oscillations)
        
        # Skip if segment is too short
        if len(segment) < 512:
            print(f"Segment {i+1} is too short (<512 samples); skipping.")
            results.append({
                'segment_id': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'detected_notes': [],
                'weights': np.zeros(len(note_names))
            })
            continue

        # Compute FFT of the segment
        ft = np.fft.rfft(segment)
        magnitude = np.abs(ft)
        freqs = np.fft.rfftfreq(len(segment), 1/sr)

        # Quantize to the same bins used by the basis
        b = quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width)

        # Normalize b
        if np.max(b) > 0:
            b = b / np.max(b)

        # Detect notes for this segment
        detected_notes, weights = detect_notes(A, b, note_names, threshold=detection_threshold)

        # Store result
        results.append({
            'segment_id': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'detected_notes': detected_notes,   # list of (note_name, weight)
            'weights': weights                  # full weight vector
        })

    # Done
    return results, onset_times

def visualize_sequential_results(results, onset_times=None):
    """
    Simple timeline visualization: each segment as a block with the top detected note shown.
    """
    if not results:
        print("No results to visualize.")
        return

    # Build label per segment (top-1 note by weight if available)
    labels = []
    starts = []
    ends = []
    for r in results:
        starts.append(r['start_time'])
        ends.append(r['end_time'])
        if r['detected_notes']:
            top = max(r['detected_notes'], key=lambda t: t[1])
            labels.append(f"{top[0]} ({top[1]:.2f})")
        else:
            labels.append("—")

    fig, ax = plt.subplots(figsize=(14, 4))

    # Draw blocks
    for i, (s, e, lab) in enumerate(zip(starts, ends, labels)):
        ax.axvspan(s, e, alpha=0.2)
        ax.text((s + e) / 2.0, 0.5, lab, ha='center', va='center', transform=ax.get_xaxis_transform())

    # Optional: draw onset markers
    if onset_times is not None:
        for t in onset_times:
            ax.axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Sequential Note Detection (top-1 label per segment)")
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_midi_from_sequential_results(results, output_file='sequential_detected.mid', seconds_per_quarter=0.5):
    """
    Create a MIDI file from sequential detection results.
    Uses the *top-1 note* per segment, with duration proportional to the segment length.
    seconds_per_quarter maps real seconds to MIDI quarterLength (default assumes ~120 BPM).
    """
    if not results:
        print("No results to write to MIDI.")
        return

    s = stream.Stream()
    current_offset = 0.0  # in quarterLength units

    for r in results:
        seg_len_sec = max(0.05, r['end_time'] - r['start_time'])  # guard very short segments
        ql = max(0.25, seg_len_sec / seconds_per_quarter)         # min duration 1/4 note

        if r['detected_notes']:
            # pick top-1 note
            top_note = max(r['detected_notes'], key=lambda t: t[1])[0]
            try:
                n = note.Note(top_note)
                n.duration.quarterLength = ql
                s.insert(current_offset, n)
            except Exception as e:
                print(f"Skipping note '{top_note}' due to parse error: {e}")
        else:
            # no detection -> insert a rest to keep timing aligned
            r_obj = note.Rest()
            r_obj.duration.quarterLength = ql
            s.insert(current_offset, r_obj)

        current_offset += ql

    s.write('midi', fp=output_file)
    print(f"✓ Sequential MIDI saved: {output_file}")

# --- Main Execution ---
if __name__ == "__main__":
    # ===== Config =====
    num_harmonics = 6          # increase if you want more partials in the basis
    bin_width = 10.0           # Hz window around each bin center
    sr = 44100
    detection_threshold = 0.35

    # Basis trimming params (match your segment trimming where possible)
    use_trimming = True
    skip_oscillations = 10000
    keep_oscillations = 20000

    # ===== Basis note files (filenames may use 'dflat4', 'gsharp3', etc.) =====
    note_files = [
        "pure_notes/c3.mp3", 
        "pure_notes/db3.mp3", 
        "pure_notes/d3.mp3", 
        "pure_notes/eb3.mp3", 
        "pure_notes/e3.mp3", 
        "pure_notes/f3.mp3", 
        "pure_notes/gb3.mp3", 
        "pure_notes/g3.mp3", 
        "pure_notes/ab3.mp3", 
        "pure_notes/a3.mp3", 
        "pure_notes/bb3.mp3", 
        "pure_notes/b3.mp3", 

        "pure_notes/c4.mp3", 
        "pure_notes/db4.mp3", 
        "pure_notes/d4.mp3", 
        "pure_notes/eb4.mp3", 
        "pure_notes/e4.mp3",
        "pure_notes/f4.mp3", 
        "pure_notes/gb4.mp3", 
        "pure_notes/g4.mp3", 
        "pure_notes/ab4.mp3", 
        "pure_notes/a4.mp3",
        "pure_notes/bb4.mp3", 
        "pure_notes/b4.mp3", 

        "pure_notes/c5.mp3",
        "pure_notes/db5.mp3", 
        "pure_notes/d5.mp3",
        "pure_notes/eb5.mp3", 
        "pure_notes/e5.mp3",
        "pure_notes/f5.mp3",
        "pure_notes/gb5.mp3", 
        "pure_notes/g5.mp3",
        "pure_notes/ab5.mp3", 
        "pure_notes/a5.mp3",
        "pure_notes/bb5.mp3", 
        "pure_notes/b5.mp3",
        "pure_notes/c6.mp3",
    ]

    # Sequential notes audio (update this to your file)
    sequential_audio_file = "sequential/hotcross.mp3"

    try:
        # ===== Derive note_range directly from the filenames (canonicalized) =====
        stems = [p.split('/')[-1].split('.')[0] for p in note_files]
        canonical_names = [filename_stem_to_pitchname(s) for s in stems]  # e.g., 'dflat4' -> 'Db4'

        # Sort by MIDI so bins are in musical order; also unique them
        unique_names = sorted(set(canonical_names), key=lambda n: pitch.Pitch(n).midi)
        note_range = unique_names

        # ===== Create frequency bins =====
        print("=== Creating Frequency Bins ===")
        bin_centers = create_frequency_bins(note_range, num_harmonics=num_harmonics)
        print(f"Created {len(bin_centers)} frequency bins")
        print(f"Frequency range: {bin_centers[0]:.2f} Hz to {bin_centers[-1]:.2f} Hz")
        print(f"Bin width: ±{bin_width} Hz")

        # ===== Build basis matrix =====
        A, note_names = build_basis_matrix(
            note_files, bin_centers, sr=sr, bin_width=bin_width,
            use_trimming=use_trimming,
            skip_oscillations=skip_oscillations,
            keep_oscillations=keep_oscillations
        )
        print(f"\n=== Basis Matrix Built ===")
        print(f"Shape: {A.shape} ({A.shape[0]} bins × {A.shape[1]} notes)")
        print(f"Notes: {note_names}")

        # (Optional) visualize basis
        # plot_basis_matrix(A, note_names, bin_centers)

        # ===== Analyze sequential notes =====
        print("\n" + "="*70)
        print("TESTING SEQUENTIAL NOTE DETECTION")
        print("="*70)

        results, onset_times = analyze_sequential_notes(
            audio_file=sequential_audio_file,
            A=A,
            note_names=note_names,
            bin_centers=bin_centers,
            sr=sr,
            bin_width=bin_width,
            detection_threshold=detection_threshold,
            # onset detection params:
            hop_length=100,
            pre_max=20, post_max=20, pre_avg=100, post_avg=100,
            delta=0.2, wait=0,
            # per-segment trimming params:
            trim_segments=True,
            skip_oscillations=1000,
            keep_oscillations=6000
        )

        # ===== Visualize + MIDI =====
        visualize_sequential_results(results, onset_times)
        create_midi_from_sequential_results(results, output_file='sequential_detected.mid')

        # ===== Summary =====
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Total segments analyzed: {len(results)}")
        detected_count = sum(1 for r in results if r['detected_notes'])
        print(f"Segments with detected notes: {detected_count}")

        print("\nDetected sequence:")
        for r in results:
            if r['detected_notes']:
                notes_str = ', '.join([f"{n} ({w:.3f})" for n, w in r['detected_notes']])
                print(f"  Segment {r['segment_id']} [{r['start_time']:.2f}s–{r['end_time']:.2f}s]: {notes_str}")
            else:
                print(f"  Segment {r['segment_id']} [{r['start_time']:.2f}s–{r['end_time']:.2f}s]: (no detection)")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure the sequential audio file and note files exist in the specified folders!")
        for nf in note_files:
            print(f"  - {nf}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

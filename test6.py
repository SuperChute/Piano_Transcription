import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, chord, midi, pitch, tempo

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

def build_basis_matrix(note_files, bin_centers, sr=44100, bin_width=10.0, 
                      use_trimming=True, skip_oscillations=10000, keep_oscillations=20000,
                      plot_first_n=1):
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
        note_name = note_file.split('/')[-1].split('.')[0].upper()
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
        
        signal = signal * np.hanning(len(signal))
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


def chromatic_range(start='C3', end='C6', use_flats=True):
    """
    Build a chromatic list of names from start..end inclusive.
    use_flats=True forces Db/Eb/Gb/Ab/Bb instead of C#/D#/F#/G#/A#.
    """
    from music21 import pitch
    start_m = pitch.Pitch(start).midi
    end_m   = pitch.Pitch(end).midi
    repl = {'C#':'Db','D#':'Eb','F#':'Gb','G#':'Ab','A#':'Bb'}

    names = []
    for m in range(start_m, end_m + 1):
        p = pitch.Pitch()
        p.midi = m
        name = p.nameWithOctave  # usually uses sharps
        if use_flats:
            for sh, fl in repl.items():
                if name.startswith(sh):
                    name = name.replace(sh, fl, 1)
                    break
        names.append(name)
    return names

# --- ONSET DETECTION ---

def detect_onsets_simple(signal, sr=44100, hop_length=512, threshold=0.15):
    """
    Detect note onsets using librosa's built-in onset detection.
    
    Args:
        signal: Audio signal array
        sr: Sample rate
        hop_length: Number of samples between analysis frames
        threshold: Sensitivity (lower = more sensitive, range: 0.0-1.0)
    
    Returns:
        onset_samples: Array of sample indices where onsets occur
        onset_times: Array of onset times in seconds
    """
    # Use librosa's onset detection (combines multiple methods)
    onset_frames = librosa.onset.onset_detect(
        y=signal,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,  # More accurate onset positions
        units='frames'
    )
    
    # Convert frames to sample indices
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    onset_times = onset_samples / sr
    
    print(f"\n=== Onset Detection ===")
    print(f"Detected {len(onset_samples)} onsets")
    print(f"Onset times (seconds): {onset_times}")
    
    return onset_samples, onset_times


def segment_and_detect_notes(signal, sr, onset_samples, A, note_names, bin_centers,
                             bin_width=10.0, threshold=0.35,
                             min_segment_samples=2048,
                             trim_skip=5000, trim_keep=15000):
    """
    Segment audio by onsets and detect notes in each segment.
    Uses your existing trimming approach to handle decaying notes.
    
    Args:
        signal: Full audio signal
        sr: Sample rate
        onset_samples: Array of onset positions (sample indices)
        A: Basis matrix for note detection
        note_names: List of note names
        bin_centers: Frequency bins
        bin_width: Bin width for quantization
        threshold: Detection threshold
        min_segment_samples: Minimum segment length (skip shorter ones)
        trim_skip: Samples to skip after segment start (remove attack transient)
        trim_keep: Samples to keep for analysis (clean sustain portion)
    
    Returns:
        detected_timeline: List of dicts with note, start_time, duration, weight
    """
    print("\n=== Segmenting and Detecting Notes ===")
    
    # Add end of signal as final boundary
    segment_boundaries = np.append(onset_samples, len(signal))
    
    detected_timeline = []
    
    for i in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i + 1]
        segment_length = end_idx - start_idx
        
        print(f"\n--- Segment {i+1} ---")
        print(f"  Range: samples {start_idx} to {end_idx} (length: {segment_length})")
        
        # Skip very short segments (likely noise or errors)
        if segment_length < min_segment_samples:
            print(f"  ⚠️  Segment too short ({segment_length} < {min_segment_samples}), skipping")
            continue
        
        # Extract segment
        segment = signal[start_idx:end_idx]
        
        # YOUR TRIMMING MAGIC: Remove attack, keep clean sustain
        # This handles the decaying note problem!
        if len(segment) > trim_skip + trim_keep:
            # Find max amplitude in segment (the attack)
            max_idx = np.argmax(np.abs(segment))
            
            # Skip past the attack transient
            trim_start = max_idx + trim_skip
            trim_end = min(trim_start + trim_keep, len(segment))
            
            if trim_start < len(segment):
                segment_trimmed = segment[trim_start:trim_end]
                print(f"  ✂️  Trimmed: {len(segment)} → {len(segment_trimmed)} samples")
            else:
                segment_trimmed = segment
                print(f"  ⚠️  Segment too short to trim, using full segment")
        else:
            segment_trimmed = segment
            print(f"  ℹ️  Segment too short for trimming ({len(segment)} samples)")
        
        # Apply window
        segment_windowed = segment_trimmed * np.hanning(len(segment_trimmed))
        
        # Compute FFT
        ft = np.fft.rfft(segment_windowed)
        magnitude = np.abs(ft)
        freqs = np.fft.rfftfreq(len(segment_windowed), 1/sr)
        
        # Quantize to bins
        b = quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width)
        b = b / np.max(b) if np.max(b) > 0 else b
        
        # YOUR EXISTING DETECTION
        detected_notes, weights = detect_notes(A, b, note_names, threshold)
        
        # Calculate timing
        start_time = start_idx / sr
        duration = (end_idx - start_idx) / sr
        
        # Store results
        if detected_notes:
            for note, weight in detected_notes:
                detected_timeline.append({
                    'note': note,
                    'start_time': start_time,
                    'duration': duration,
                    'weight': weight,
                    'velocity': int(min(weight * 127, 127))  # MIDI velocity
                })
                print(f"  ✅ Detected: {note} (weight: {weight:.3f})")
        else:
            print(f"  ❌ No notes detected")
    
    return detected_timeline


def create_midi_sequence(timeline, output_file='transcription.mid', bpm=120):
    """
    Create MIDI file from detected note timeline.
    
    Args:
        timeline: List of note events from segment_and_detect_notes
        output_file: Path to save MIDI file
        bpm: BPM for tempo reference (renamed from 'tempo' to avoid conflict)
    """
    if not timeline:
        print("No notes in timeline to create MIDI.")
        return
    
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=bpm))
    
    for event in timeline:
        try:
            n = note.Note(event['note'])
            n.offset = event['start_time']  # Absolute time in seconds
            n.quarterLength = event['duration']
            n.volume.velocity = event['velocity']
            s.append(n)
        except Exception as e:
            print(f"Warning: Could not add note {event['note']}: {e}")
    
    s.write('midi', fp=output_file)
    print(f"\n✅ MIDI sequence saved: {output_file}")
    print(f"   Total notes: {len(timeline)}")
    if timeline:
        print(f"   Duration: {timeline[-1]['start_time'] + timeline[-1]['duration']:.2f} seconds")


def plot_onsets_on_waveform(signal, sr, onset_samples, title="Detected Onsets"):
    """
    Visualize onsets overlaid on the audio waveform.
    """
    time = np.arange(len(signal)) / sr
    onset_times = onset_samples / sr
    
    plt.figure(figsize=(16, 6))
    plt.plot(time, signal, linewidth=0.5, alpha=0.7, label='Audio Signal')
    
    # Mark onsets with vertical lines
    for onset_time in onset_times:
        plt.axvline(x=onset_time, color='red', linestyle='--', 
                   linewidth=2, alpha=0.8)
    
    # Mark first onset more prominently
    if len(onset_times) > 0:
        plt.axvline(x=onset_times[0], color='red', linestyle='--', 
                   linewidth=2, alpha=0.8, label='Detected Onsets')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()







# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    #note_range = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4' ]
    note_range = chromatic_range('C3', 'C6', use_flats=True)
    num_harmonics = 5
    bin_width = 10.0  # Hz
    sr = 44100
    detection_threshold = 0.3  # Adjust this to tune sensitivity
    
    # Trimming parameters
    use_trimming = True  # Set to False to disable trimming
    skip_oscillations = 0  # Samples to skip after max amplitude
    keep_oscillations = 50000  # Samples to keep for analysis
    
    # Create frequency bins
    print("=== Creating Frequency Bins ===")
    bin_centers = create_frequency_bins(note_range, num_harmonics=num_harmonics)
    print(f"Created {len(bin_centers)} frequency bins")
    print(f"Frequency range: {bin_centers[0]:.2f} Hz to {bin_centers[-1]:.2f} Hz")
    print(f"Bin width: ±{bin_width} Hz")
    
    # Build basis matrix from synthetic notes
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
    try:
        A, note_names = build_basis_matrix(note_files, bin_centers, sr=sr, bin_width=bin_width,
                                          use_trimming=use_trimming, 
                                          skip_oscillations=skip_oscillations,
                                          keep_oscillations=keep_oscillations)
        print(f"\n=== Basis Matrix Built ===")
        print(f"Shape: {A.shape} ({A.shape[0]} bins × {A.shape[1]} notes)")
        print(f"Notes: {note_names}")
        
        # Visualize basis matrix
        plot_basis_matrix(A, note_names, bin_centers)
        
        # === NEW: TEST SEQUENTIAL NOTE DETECTION ===
        print("\n" + "="*60)
        print("=== SEQUENTIAL NOTE DETECTION TEST ===")
        print("="*60)
        
        # Load your sequential audio file 
        test_audio_file = "sequential/hot_cross_buns.mp3"  # YOUR FILE HERE
        
        print(f"\nLoading: {test_audio_file}")
        signal, _ = librosa.load(test_audio_file, sr=sr, mono=True)
        
        print(f"Audio length: {len(signal)} samples ({len(signal)/sr:.2f} seconds)")
        
        # Plot original waveform
        plot_mixed_signal(signal, sr, title="Input Audio - Sequential Notes")
        
        # 1. DETECT ONSETS
        onset_samples, onset_times = detect_onsets_simple(
            signal, sr=sr, 
            hop_length=512,
            threshold=detection_threshold 
        )
        
        # Visualize onsets
        plot_onsets_on_waveform(signal, sr, onset_samples, 
                                title="Onset Detection Results")
        
        # 2. SEGMENT AND DETECT NOTES
        timeline = segment_and_detect_notes(
            signal=signal,
            sr=sr,
            onset_samples=onset_samples,
            A=A,
            note_names=note_names,
            bin_centers=bin_centers,
            bin_width=bin_width,
            threshold=detection_threshold,
            min_segment_samples=2048, # Window size
            trim_skip=5000,     # Skip samples after attack (adjust for piano)
            trim_keep=10000     # Keep samples of clean sustain
        )
        
        # 3. DISPLAY RESULTS
        print("\n" + "="*60)
        print("TRANSCRIPTION RESULTS")
        print("="*60)
        
        if timeline:
            print(f"\n✅ Detected {len(timeline)} notes:")
            for i, event in enumerate(timeline, 1):
                print(f"  {i}. {event['note']:4s} @ {event['start_time']:.3f}s "
                      f"(duration: {event['duration']:.3f}s, weight: {event['weight']:.3f})")
        else:
            print("\n❌ No notes detected")
        
        # 4. CREATE MIDI OUTPUT
        if timeline:
            create_midi_sequence(timeline, output_file='transcription.mid', bpm=120)
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("Make sure all note files exist in the 'pure_notes/' directory:")
        for nf in note_files:
            print(f"  - {nf}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, chord, midi, pitch

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
                      plot_first_n=8):
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


# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    note_range = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5' ] #, 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6']
    num_harmonics = 2
    bin_width = 10.0  # Hz
    sr = 44100
    detection_threshold = 0.3  # Adjust this to tune sensitivity
    
    # Trimming parameters
    use_trimming = True  # Set to False to disable trimming
    skip_oscillations = 60000  # Samples to skip after max amplitude
    keep_oscillations = 20000  # Samples to keep for analysis
    plot_first_n_notes = 8  # Number of notes to show before/after trimming plots (set to 0 to disable)
    
    # Create frequency bins
    print("=== Creating Frequency Bins ===")
    bin_centers = create_frequency_bins(note_range, num_harmonics=num_harmonics)
    print(f"Created {len(bin_centers)} frequency bins")
    print(f"Frequency range: {bin_centers[0]:.2f} Hz to {bin_centers[-1]:.2f} Hz")
    print(f"Bin width: ±{bin_width} Hz")
    
    # Build basis matrix from pure notes
    """
        Real Notes
        "pure_notes/C4_real.m4a",
        "pure_notes/D4_real.m4a",
        "pure_notes/E4_real.m4a",
        "pure_notes/F4_real.m4a",
        "pure_notes/G4_real.m4a",
        "pure_notes/A4_real.m4a",
        "pure_notes/B4_real.m4a",
        "pure_notes/C5_real2.m4a",


        Synthetic Notes
        "pure_notes/c4.mp3",
        "pure_notes/d4.mp3",
        "pure_notes/e4.mp3",
        "pure_notes/f4.mp3",
        "pure_notes/g4.mp3",
        "pure_notes/a4.mp3",
        "pure_notes/b4.mp3",
        "pure_notes/c5.mp3",
        "pure_notes/d5.mp3",
        "pure_notes/e5.mp3",
        
        Sound Waves
        "soundwave/C4.webm",
        "soundwave/D4.webm",
        "soundwave/E4.webm",
        "soundwave/F4.webm",
        "soundwave/G4.webm",
        "soundwave/A4.webm",
        "soundwave/B4.webm",
        "soundwave/C5.webm",
    """ 
    note_files = [
        "pure_notes/C4_real.m4a",
        "pure_notes/D4_real.m4a",
        "pure_notes/E4_real.m4a",
        "pure_notes/F4_real.m4a",
        "pure_notes/G4_real.m4a",
        "pure_notes/A4_real.m4a",
        "pure_notes/B4_real.m4a",
        "pure_notes/C5_real2.m4a",
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
        
        # === TEST: Detect notes in a mixed signal ===
        print("\n" + "="*60)
        print("=== TESTING NOTE DETECTION ===")
        print("="*60)
        
        test_files = [
        "pure_notes/C4_real.m4a",
        "pure_notes/D4_real.m4a",
        "pure_notes/E4_real.m4a",
        "pure_notes/F4_real.m4a",
        ]
        
        print(f"\nNotes Inputted: {[f.split('/')[-1].split('.')[0].upper() for f in test_files]}")
        
        # Load and mix test signal
        mixed_signal, _ = load_and_mix_signals(test_files, sr=sr)
        plot_mixed_signal(mixed_signal, sr, 
                  title=f"Mixed Signal: {[f.split('/')[-1] for f in test_files]}")
        
        # Compute FFT of mixed signal
        ft = np.fft.rfft(mixed_signal)
        magnitude = np.abs(ft)
        freqs = np.fft.rfftfreq(len(mixed_signal), 1/sr)
        
        # Quantize mixed signal to same bins
        b = quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width)
        
        # Normalize
        b = b / np.max(b) if np.max(b) > 0 else b
        
        print(f"Mixed signal quantized to vector b of length {len(b)}")
        
        # Visualize quantized mixed signal
        plot_quantized_spectrum(bin_centers, b, title="Quantized Mixed Signal (Input)")
        
        # Solve notes
        detected_notes, x = detect_notes(A, b, note_names, threshold=detection_threshold)
        
        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)
        if detected_notes:
            print(f"\nDetected {len(detected_notes)} note(s):")
            for note_name, weight in detected_notes:
                print(f"  ✓ {note_name} (weight: {weight:.4f})")
        else:
            print("\n✗ No notes detected (try lowering the threshold)")
        
        # Visualize results
        plot_detection_results(note_names, x, detected_notes, detection_threshold)
        
        # Create MIDI output
        if detected_notes:
            create_midi_from_detected_notes(detected_notes, output_file='detected_chord.mid')
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("Make sure all note files exist in the 'pure_notes/' directory:")
        for nf in note_files:
            print(f"  - {nf}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
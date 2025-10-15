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
    note_name = notes[midi_name % 12]
    return f"{note_name}{octave}"


# --- Audio Preprocessing ---

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

def create_frequency_bins(note_range, num_harmonics):
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

def build_basis_matrix(note_files, bin_centers, sr=44100, bin_width=10.0):
    """
    Load each pure note, compute FFT, quantize to bins, and stack into matrix A.
    
    Args:
        note_files: List of paths to pure note audio files
        bin_centers: Frequency bins to use
        sr: Sample rate
        bin_width: Bin width for quantization
    
    Returns:
        A: Basis matrix (num_bins × num_notes)
        note_names: List of note names corresponding to columns
    """
    basis_vectors = []
    note_names = []
    
    print("\n=== Building Basis Matrix ===")
    for note_file in note_files:
        # Extract note name from filename
        note_name = note_file.split('/')[-1].split('.')[0].upper()
        note_names.append(note_name)
        
        # Load audio
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
        print(f"  {note_name}: {len(signal)} samples, max quantized bin = {max_val:.2f}")
    
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
        x: Full weight vector (solution to Ax = b)
    """
    print("\n=== Solving Ax = b ===")
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    
    # Solve the least squares problem: minimize ||Ax - b||^2
    # Solve: minimize ||A×x - b||²
    solution = np.linalg.lstsq(A, b, rcond=None)

    x = solution[0]          # The weight vector (what we care about!)
    residuals = solution[1]  # How far off the solution is (error)
    rank = solution[2]       # Matrix rank (for debugging)
    s = solution[3]          # Singular values (rarely used)
    
    print(f"\nSolution x (weights):")
    for i, (note, weight) in enumerate(zip(note_names, x)):
        print(f"  {note}: {weight:.4f}")
    
    if len(residuals) > 0:
        print(f"\nResidual error: {residuals[0]:.6f}")
    print(f"Matrix rank: {rank}/{A.shape[1]}")
    
    # Threshold to detect which notes are present
    detected_notes = []
    for note, weight in zip(note_names, x):
        if weight >= threshold:
            detected_notes.append((note, weight))
    
    return detected_notes, x


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
    
    # Show some frequency values on y-axis
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
    ax1.set_title('Solution Vector x (Note Weights)')
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


# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    note_range = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
    num_harmonics = 2
    bin_width = 10.0  # Hz
    sr = 44100
    detection_threshold = 0.3  # Adjust this to tune sensitivity
    
    # Create frequency bins
    print("=== Creating Frequency Bins ===")
    bin_centers = create_frequency_bins(note_range, num_harmonics=num_harmonics)
    print(f"Created {len(bin_centers)} frequency bins")
    print(f"Frequency range: {bin_centers[0]:.2f} Hz to {bin_centers[-1]:.2f} Hz")
    print(f"Bin width: ±{bin_width} Hz")
    
    # Build basis matrix from pure notes
    note_files = [
        "pure_notes/c4.mp3",
        "pure_notes/d4.mp3",
        "pure_notes/e4.mp3",
        "pure_notes/f4.mp3",
        "pure_notes/g4.mp3",
        "pure_notes/a4.mp3",
        "pure_notes/b4.mp3",
        "pure_notes/c5.mp3",
    ]
    
    try:
        A, note_names = build_basis_matrix(note_files, bin_centers, sr=sr, bin_width=bin_width)
        print(f"\n=== Basis Matrix Built ===")
        print(f"Shape: {A.shape} ({A.shape[0]} bins × {A.shape[1]} notes)")
        print(f"Notes: {note_names}")
        
        # Visualize basis matrix
        plot_basis_matrix(A, note_names, bin_centers)
        
        # === TEST: Detect notes in a mixed signal ===
        print("=== TESTING NOTE DETECTION ===")      

        test_files = [
          "pure_notes/d4.mp3",
          "pure_notes/f4.mp3",
          "pure_notes/a4.mp3",
          "pure_notes/d5.mp3",
        ]
        
        print(f"\nGround truth: {[f.split('/')[-1].split('.')[0].upper() for f in test_files]}")
        
        # Load and mix test signal
        mixed_signal, _ = load_and_mix_signals(test_files, sr=sr)
        
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
        
        # === SOLVE FOR NOTES ===
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
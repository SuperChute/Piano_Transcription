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
        # This gives the frequency values from the note name
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
    Essentialy this function takes an audio input fft and compares it to all the notes 
    in the bin_centers to find which notes have the most energy
    
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
    plt.figure(figsize=(12, 5))
    plt.stem(bin_centers, quantized_magnitude, basefmt=' ')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    note_range = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
    num_harmonics = 2
    bin_width = 10.0  # Hz
    sr = 44100
    
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
        
        # Test: Load and quantize a mixed signal
        print("\n=== Testing Mixed Signal ===")
        test_files = [
            "pure_notes/d4.mp3",
            "pure_notes/d5.mp3",
        ]
        
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
        plot_quantized_spectrum(bin_centers, b, title="Quantized Mixed Signal")
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("Make sure all note files exist in the 'pure_notes/' directory:")
        for nf in note_files:
            print(f"  - {nf}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
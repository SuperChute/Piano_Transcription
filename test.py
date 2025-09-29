import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, midi

# Converts frequency in Hz to MIDI note number
def frequency_to_midi(freq):
    """Convert frequency in Hz to MIDI note number"""
    if freq <= 0:
        return None
    midi_note = (12 * np.log2(freq / 440)) + 69 
    return round(midi_note)

def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note_name = notes[midi_note % 12]
    return f"{note_name}{octave}"

def get_magnitude_at_freq(target_freq, freqs, magnitude, tolerance=5.0):
    """
    Get the magnitude at a target frequency within a tolerance.
    Returns the max magnitude found in the range [target_freq - tolerance, target_freq + tolerance]
    """
    # Find indices within tolerance
    mask = np.abs(freqs - target_freq) <= tolerance
    if np.any(mask):
        return np.max(magnitude[mask])
    return 0.0

def create_candidate_notes(min_midi, max_midi):
    """
    Create a dictionary of candidate notes to check.
    Within the octave C4 - C5
    """
    candidates = {}
    for midi_num in range(min_midi, max_midi + 1):
        note_name = midi_to_note_name(midi_num)
        # Calculate frequency from MIDI number
        freq = 440 * (2 ** ((midi_num - 69) / 12))
        candidates[note_name] = freq
    return candidates

def harmonic_summation(freqs, magnitude, candidate_notes, num_harmonics=6, tolerance=5.0):
    """
    Perform harmonic summation to detect which notes are present.
    
    Args:
        freqs: Array of FFT frequency bins
        magnitude: Array of FFT magnitudes
        candidate_notes: Dict of note_name -> fundamental_frequency
        num_harmonics: How many harmonics to check (including fundamental)
        tolerance: Hz tolerance when looking up harmonic magnitudes
    
    Returns:
        Dict of note_name -> confidence_score
    """
    pitch_confidence = {}
    
    print("\n=== HARMONIC SUMMATION ANALYSIS ===\n")
    
    for note_name, fundamental_freq in candidate_notes.items():
        score = 0.0
        harmonic_details = []
        
        # Sum energy from fundamental and its harmonics
        for harmonic_num in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * harmonic_num
            
            # Don't look beyond Nyquist or reasonable piano range
            if harmonic_freq > freqs[-1] or harmonic_freq > 4000:
                break
            
            # Get magnitude at this harmonic frequency
            harm_magnitude = get_magnitude_at_freq(harmonic_freq, freqs, magnitude, tolerance)
            score += harm_magnitude
            
            harmonic_details.append(f"  {harmonic_num}x: {harmonic_freq:.1f}Hz → mag={harm_magnitude:.1f}")
        
        pitch_confidence[note_name] = score
        
        # Print details for notes with significant scores
        if score > 100:
            print(f"{note_name} (f0={fundamental_freq:.2f}Hz) | Score: {score:.1f}")
            for detail in harmonic_details:
                print(detail)
            print()
    
    return pitch_confidence

def detect_notes(pitch_confidence, threshold=500):
    """
    Detect which notes are present based on confidence scores.
    
    Args:
        pitch_confidence: Dict of note_name -> score
        threshold: Minimum score to consider a note as "detected"
    
    Returns:
        List of detected note names, sorted by score (highest first)
    """
    detected = [(note, score) for note, score in pitch_confidence.items() if score >= threshold]
    detected.sort(key=lambda x: x[1], reverse=True)
    return detected

#---------------------------------------------------------

def load_and_mix_signals(audio_files, sr=44100):
    """Load multiple mono files, align lengths, and mix into one signal."""
    signals = []
    for path in audio_files:
        sig, sr_loaded = librosa.load(path, sr=sr, mono=True)
        signals.append(sig)

    max_len = max(len(s) for s in signals)
    padded = [np.pad(s, (0, max_len - len(s))) for s in signals]
    mixed = np.sum(np.stack(padded, axis=0), axis=0)
    return mixed, sr

def plot_wave_and_fft(signal, sr, max_freq_to_show, pitch_confidence, detected_notes):
    """Plot waveform, FFT, and harmonic summation results."""
    # FFT
    ft = np.fft.rfft(signal)
    magnitude = np.abs(ft)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    
    mask = freqs <= max_freq_to_show
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Time-domain waveform
    plt.subplot(3, 1, 1)
    plt.plot(signal)
    plt.title('Mixed Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Frequency-domain magnitude
    plt.subplot(3, 1, 2)
    plt.plot(freqs[mask], magnitude[mask])
    plt.title(f'FFT Magnitude (up to {max_freq_to_show} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    # Pitch confidence scores
    plt.subplot(3, 1, 3)
    notes = list(pitch_confidence.keys())
    scores = list(pitch_confidence.values())
    colors = ['green' if note in [n for n, s in detected_notes] else 'lightgray' for note in notes]
    
    plt.bar(notes, scores, color=colors)
    plt.title('Harmonic Summation Confidence Scores')
    plt.xlabel('Note')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Multiple note files
    audio_files = [
        #"pure_notes/c4.mp3",
        "pure_notes/e4.mp3",
        #"pure_notes/c5.mp3"
        "pure_notes/e5.mp3"
    ]   

    freq_to_show = 1000
    
    try:
        # Load and mix audio
        mixed_signal, sr = load_and_mix_signals(audio_files, sr=44100)
        
        # Compute FFT
        ft = np.fft.rfft(mixed_signal)
        magnitude = np.abs(ft)
        freqs = np.fft.rfftfreq(len(mixed_signal), 1/sr)
        
        # Create candidate notes (C4 to C5)
        candidate_notes = create_candidate_notes(min_midi=60, max_midi=72)
        
        # Perform harmonic summation
        pitch_confidence = harmonic_summation(freqs, magnitude, candidate_notes, 
                                             num_harmonics=6, tolerance=5.0)
        
        # Detect notes above threshold
        detected_notes = detect_notes(pitch_confidence, threshold=500)
        
        # Print results
        print("\n=== DETECTED NOTES ===")
        if detected_notes:
            for note, score in detected_notes:
                print(f"✓ {note}: {score:.1f}")
        else:
            print("No notes detected above threshold")
        
        # Plot everything
        plot_wave_and_fft(mixed_signal, sr, freq_to_show, pitch_confidence, detected_notes)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Check your file paths.")
    except Exception as e:
        print(f"Error: {e}")
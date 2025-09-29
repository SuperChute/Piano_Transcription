import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, chord, midi

# --- Configuration and Utility Functions ---

def frequency_to_midi(freq):
    """
    Convert frequency in Hz to MIDI note number using the A440 standard.
    """
    if freq <= 0:
        return None
    # Formula for converting frequency to MIDI note number
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
    to simulate a chord. This serves as the basic audio preprocessing step.
    
    Returns: mixed signal (np.array), sample rate (int)
    """
    signals = []
    for path in audio_files:
        # Load and resample to target sr if necessary
        sig, sr_loaded = librosa.load(path, sr=sr, mono=True)
        signals.append(sig)

    # Pad all to the same length (max of all)
    max_len = max(len(s) for s in signals)
    padded = [np.pad(s, (0, max_len - len(s))) for s in signals]

    # Mix by summing
    mixed = np.sum(np.stack(padded, axis=0), axis=0)
    return mixed, sr


# --- MIDI Output ---

def create_midi_chord(detected_notes, output_file='chord_output.mid', duration=2.0):
    """
    Creates a music21 stream and exports a MIDI file containing the detected notes 
    as a single chord block.
    
    Args:
        detected_notes (list): List of (note_name, frequency, magnitude) tuples.
        output_file (str): The path to save the MIDI file.
        duration (float): The length of the chord in quarter notes (since FFT provides no duration).
    """
    if not detected_notes:
        print("No notes detected to create MIDI.")
        return

    s = stream.Stream()
    
    # Extract MIDI pitches from the detected fundamentals
    midi_pitches = []
    for _, freq, _ in detected_notes:
        midi_num = frequency_to_midi(freq)
        if midi_num is not None:
            midi_pitches.append(midi_num)

    # Create a chord object from all detected pitches
    if midi_pitches:
        c = chord.Chord(midi_pitches)
        c.duration.quarterLength = duration
        s.append(c)

        # Write to MIDI file
        s.write('midi', fp=output_file)
        print(f"\nSUCCESS: MIDI file saved as: {output_file}")
    else:
        print("No valid MIDI notes could be created.")


# --- Multi-Pitch Estimation (Current Heuristic) ---

def harmonic_suppression_filter(note_peaks, freq_tolerance=3.0, max_harmonic=5):
    """
    APPLIED METHOD: Harmonic Suppression Filter (Heuristic)
    This filter attempts to remove notes that are likely harmonics by checking if a 
    higher-frequency peak (f_high) is a harmonic of a lower-frequency peak (f_low) 
    AND is weaker than it (Mag(f_high) <= Mag(f_low)).

    NOTE: This method has been proven to be unstable, especially for higher piano notes 
    where harmonics can be stronger than the fundamental, necessitating the move to HPS.
    
    Args:
        note_peaks (dict): {note_name: (frequency, magnitude)}
        freq_tolerance (float): tolerance in Hz for harmonic matching (e.g., for 2x, 3x, etc.)
        max_harmonic (int): highest harmonic multiplier to check (e.g. 5 = check up to 5x multiple)

    Returns:
        filtered_notes (list): list of (note_name, frequency, magnitude) for fundamentals only.
    """
    note_items = list(note_peaks.items())
    notes_to_remove = set()

    # Sort by frequency (low → high) to ensure f_low is always checked before f_high
    sorted_items = sorted(note_items, key=lambda item: item[1][0])

    for i in range(len(sorted_items)):
        low_name, (low_freq, low_mag) = sorted_items[i]

        for j in range(i + 1, len(sorted_items)):
            high_name, (high_freq, high_mag) = sorted_items[j]

            # Check for harmonic relationship: 2×, 3×, … up to max_harmonic
            for k in range(2, max_harmonic + 1):
                # If high_freq is within tolerance of k * low_freq
                if abs(high_freq - k * low_freq) <= freq_tolerance:
                    
                    # Apply the Heuristic: If the potential harmonic is weaker than the fundamental
                    if high_mag <= low_mag:
                        # High note is weaker → treat it as harmonic and remove it
                        notes_to_remove.add(high_name)
                    # If high_mag > low_mag, both are kept, as high_freq is treated as a new fundamental
                    # and the heuristic is considered broken for this pair.

    # Build filtered list
    filtered_notes = []
    for note_name, (freq, mag) in sorted_items:
        if note_name not in notes_to_remove:
            filtered_notes.append((note_name, freq, mag))

    return filtered_notes


def plot_wave_and_fft(signal, sr, max_freq_to_show):
    """
    Calculates the FFT, identifies peak frequencies (Multi-Pitch Estimation - Initial), 
    applies the Harmonic Suppression Filter, and plots the results.
    
    Returns: detected_notes (list)
    """
    # 1. FFT
    ft = np.fft.rfft(signal)
    magnitude = np.abs(ft)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)

    threshold = 200.0
    note_peaks = {}
 
    # 2. Initial Peak Picking (Local Maxima + Threshold)
    print(f"Peaks ≥ {threshold} (up to {max_freq_to_show} Hz):")
    for i in range(1, len(magnitude) - 1):
        f = freqs[i]
        if f > max_freq_to_show:
            break
        # local maximum + threshold check
        if magnitude[i] >= threshold and magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
            midi_num = frequency_to_midi(f)
            if midi_num is None:
                continue
            note_name = midi_to_note_name(midi_num)
            
            # Keep the single strongest peak for a given note name (e.g., only one C4 is kept)
            if note_name not in note_peaks or magnitude[i] > note_peaks[note_name][1]:
                note_peaks[note_name] = (f, magnitude[i])

    # Print initial peaks
    for note_name, (f, m) in note_peaks.items():
        print(f"{f:7.2f} Hz  |  {m:8.2f}  |  {note_name}")
        
    # 3. Apply Harmonic Suppression Filter
    detected_notes = harmonic_suppression_filter(note_peaks, freq_tolerance=3.0)
    for note_name, f, m in detected_notes:
        print(f"Detected: {f:7.2f} Hz  |  {m:8.2f}  |  {note_name}")

    # 4. Plot Results
    # Limit spectrum to a readable range
    mask = freqs <= max_freq_to_show

    plt.figure(figsize=(12, 4))

    # Time-domain waveform
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title('Mixed Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Frequency-domain magnitude
    plt.subplot(1, 2, 2)
    plt.plot(freqs[mask], magnitude[mask])
    plt.title(f'FFT Magnitude (up to {max_freq_to_show} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()
    
    return detected_notes


# --- Main Execution ---

if __name__ == "__main__":
    audio_files = [
        "pure_notes/c4.mp3",
        #"pure_notes/d4.mp3",
        #"pure_notes/e4.mp3",
        "pure_notes/c5.mp3",

    ]   

    freq_to_show = 1000 # Max frequency to plot

    try:
        mixed_signal, sr = load_and_mix_signals(audio_files, sr=44100)
        
        # 1. Run analysis and get the list of detected fundamental notes
        detected_notes = plot_wave_and_fft(mixed_signal, sr, freq_to_show)
        
        # 2. Convert detected notes to a MIDI file (the requested output)
        create_midi_chord(detected_notes, output_file='chord_output_heuristic.mid')

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Check your file paths. If needed, list available files in 'pure_notes/'.")
    except Exception as e:
        print(f"Error: {e}")
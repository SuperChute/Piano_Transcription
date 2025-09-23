import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, midi
from scipy.signal import find_peaks

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

def detect_chord_frequencies(audio_file, num_notes=3, min_freq=80, max_freq=800):
    """Find multiple dominant frequencies in an audio file (for chords) - harmonic aware"""
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    print(f"Loaded audio: {len(y)} samples at {sr} Hz")
    
    # Apply FFT to the entire audio
    fft = np.fft.rfft(y)
    magnitude = np.abs(fft)
    
    # Create frequency bins
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    
    # Filter to piano frequency range (focus on fundamentals, not high harmonics)
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    filtered_freqs = freqs[freq_mask]
    filtered_magnitude = magnitude[freq_mask]
    
    # Find all significant peaks first
    min_height = np.max(filtered_magnitude) * 0.05  # Lower threshold to catch fundamentals
    min_distance = int(len(filtered_freqs) * 0.01)  # Closer peaks allowed
    
    peaks, properties = find_peaks(filtered_magnitude, 
                                 height=min_height, 
                                 distance=min_distance)
    
    peak_freqs = filtered_freqs[peaks]
    peak_magnitudes = filtered_magnitude[peaks]
    
    # Harmonic suppression: for each peak, check if it's likely a harmonic of a lower peak
    fundamental_candidates = []
    
    for i, (freq, mag) in enumerate(zip(peak_freqs, peak_magnitudes)):
        is_fundamental = True
        
        # Check if this frequency is a harmonic of any lower frequency peak
        for lower_freq, lower_mag in zip(peak_freqs, peak_magnitudes):
            if lower_freq >= freq:
                continue
                
            # Check if freq is approximately 2x, 3x, 4x, etc. of lower_freq
            for harmonic in [2, 3, 4, 5]:
                expected_harmonic = lower_freq * harmonic
                # Allow 5% tolerance for harmonic detection
                if abs(freq - expected_harmonic) / expected_harmonic < 0.05:
                    # This peak is likely a harmonic
                    is_fundamental = False
                    print(f"  {freq:.1f} Hz likely harmonic {harmonic} of {lower_freq:.1f} Hz")
                    break
            
            if not is_fundamental:
                break
        
        if is_fundamental:
            fundamental_candidates.append((freq, mag, i))
    
    print(f"Found {len(fundamental_candidates)} fundamental candidates")
    
    # Sort fundamental candidates by magnitude and take top N
    fundamental_candidates.sort(key=lambda x: x[1], reverse=True)
    detected_freqs = []
    detected_magnitudes = []
    
    for freq, mag, idx in fundamental_candidates[:num_notes]:
        detected_freqs.append(freq)
        detected_magnitudes.append(mag)
        midi_note = frequency_to_midi(freq)
        note_name = midi_to_note_name(midi_note) if midi_note else "Unknown"
        print(f"  Fundamental: {freq:.1f} Hz -> {note_name} (magnitude: {mag:.1f})")
    
    # Convert to numpy arrays and sort by frequency
    detected_freqs = np.array(detected_freqs)
    detected_magnitudes = np.array(detected_magnitudes)
    
    if len(detected_freqs) > 0:
        freq_sort_idx = np.argsort(detected_freqs)
        detected_freqs = detected_freqs[freq_sort_idx]
        detected_magnitudes = detected_magnitudes[freq_sort_idx]
    
    # Plot the spectrum with detected peaks
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(y)
    plt.title('Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    # Show full spectrum in piano range
    piano_mask = (freqs >= 80) & (freqs <= 1200)
    plt.plot(freqs[piano_mask], magnitude[piano_mask])
    
    # Mark ALL peaks (in light gray)
    for freq in peak_freqs:
        if freq <= 1200:
            plt.axvline(x=freq, color='lightgray', linestyle=':', alpha=0.5)
    
    # Mark detected fundamentals (in bright colors)
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, freq in enumerate(detected_freqs):
        color = colors[i % len(colors)]
        midi_note = frequency_to_midi(freq)
        note_name = midi_to_note_name(midi_note) if midi_note else "Unknown"
        plt.axvline(x=freq, color=color, linestyle='--', linewidth=2,
                   label=f'{note_name}: {freq:.1f} Hz')
    
    plt.title('Frequency Spectrum - Chord Detection (Harmonic Aware)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return detected_freqs

def create_midi_from_chord(frequencies, duration=2.0, output_file='chord_output.mid'):
    """Create a MIDI file from multiple detected frequencies (chord)"""
    if len(frequencies) == 0:
        print("No frequencies detected")
        return
    
    print(f"\nCreating MIDI chord with {len(frequencies)} notes:")
    
    # Create a music21 stream
    s = stream.Stream()
    
    # Add each note to the chord
    chord_notes = []
    for freq in frequencies:
        midi_note = frequency_to_midi(freq)
        if midi_note is not None:
            note_name = midi_to_note_name(midi_note)
            print(f"  Adding: {freq:.1f} Hz -> {note_name}")
            chord_notes.append(note.Note(midi_note))
    
    # Create a chord object (simultaneous notes)
    if len(chord_notes) > 1:
        from music21 import chord
        c = chord.Chord(chord_notes)
        c.duration.quarterLength = duration
        s.append(c)
    elif len(chord_notes) == 1:
        chord_notes[0].duration.quarterLength = duration
        s.append(chord_notes[0])
    
    # Write to MIDI file
    s.write('midi', fp=output_file)
    print(f"MIDI chord saved as: {output_file}")

def combine_notes_to_chord(note_files, output_file="combined_chord.wav"):
    """Combine multiple single note files into one chord audio file"""
    combined_audio = None
    sr = None
    
    print(f"Combining {len(note_files)} notes into a chord:")
    
    for note_file in note_files:
        filepath = f"pure_notes/{note_file}"
        print(f"  Loading: {note_file}")
        
        # Load each note
        y, current_sr = librosa.load(filepath, sr=None, mono=True)
        
        if combined_audio is None:
            combined_audio = y
            sr = current_sr
        else:
            # Make sure all files have same sample rate and length
            if current_sr != sr:
                print(f"Warning: Sample rate mismatch for {note_file}")
            
            # Match lengths (use shorter one)
            min_len = min(len(combined_audio), len(y))
            combined_audio = combined_audio[:min_len]
            y = y[:min_len]
            
            # Add the audio signals together
            combined_audio += y
    
    # Normalize to prevent clipping
    combined_audio = combined_audio / len(note_files)
    
    # Save combined audio
    import soundfile as sf
    sf.write(output_file, combined_audio, sr)
    print(f"Combined chord saved as: {output_file}")
    
    return output_file

# Main execution
if __name__ == "__main__":
    # Test with C Major chord: C4 + E4 + G4
    chord_notes = ["f4.mp3", "a4.mp3", "c5.mp3"]
    
    try:
        # First, combine the individual notes into a chord
        chord_file = combine_notes_to_chord(chord_notes)
        
        # Then detect multiple frequencies in the combined chord
        detected_frequencies = detect_chord_frequencies(chord_file, num_notes=3)
        
        # Convert to MIDI chord
        create_midi_from_chord(detected_frequencies)
        
        print(f"\nExpected notes: C4 (~261Hz), E4 (~329Hz), G4 (~392Hz)")
        
    except FileNotFoundError as e:
        print(f"Audio file not found: {e}")
        print("Make sure you have c4.mp3, e4.mp3, and g4.mp3 in your pure_notes folder")
    except ImportError:
        print("Need to install soundfile: pip install soundfile")
    except Exception as e:
        print(f"Error: {e}")
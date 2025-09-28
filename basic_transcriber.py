import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, midi

# Converts frequency in Hz to MIDI note number
# Using 440Hz as the reference frequency
def frequency_to_midi(freq):
    """Convert frequency in Hz to MIDI note number"""
    if freq <= 0:
        return None
    #Formula for converting frequency to MIDI note number via A440 standard
    midi_note = (12 * np.log2(freq / 440)) + 69 
    return round(midi_note)

def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note_name = notes[midi_note % 12]
    return f"{note_name}{octave}"

# Analyze audio to find dominant frequency
def detect_dominant_frequency(audio_file):
    """Find the dominant frequency in an audio file"""
    # Load audio file
    # y is the audio time series, sr is the sampling rate
    signal, sr = librosa.load(audio_file, sr=None, mono=True)
    print(f"Loaded audio: {len(signal)} samples at {sr} Hz")
    
    # Apply FFT to the entire audio
    ft = np.fft.rfft(signal)
    magnitude = np.abs(ft)
    
    # Create frequency bins
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    
    # Find the frequency with highest magnitude
    dominant_mag = np.argmax(magnitude)
    dominant_freq = freqs[dominant_mag]
    
    # Find the index where frequency reaches desired limit
    max_freq_to_show = 1000  # Hz
    max_idx = np.where(freqs <= max_freq_to_show)[0][-1]
    
    # Plot the spectrum
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title('Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(freqs[:max_idx], magnitude[:max_idx])
    # plt.axvline(x=dominant_freq, color='r', linestyle='--', label=f'Dominant: {dominant_freq:.1f} Hz')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Dominant magnitude: {magnitude[dominant_mag]:.1f}")
    return dominant_freq

def create_midi_from_frequency(freq, duration=2.0, output_file='output.mid'):
    """Create a MIDI file from a detected frequency"""
    midi_note = frequency_to_midi(freq)
    
    if midi_note is None:
        print("Invalid frequency detected")
        return
    
    note_name = midi_to_note_name(midi_note)
    print(f"Detected frequency: {freq:.1f} Hz")
    print(f"MIDI note: {midi_note} ({note_name})")
    
    # Create a music21 stream
    s = stream.Stream()
    n = note.Note(midi_note)
    n.duration.quarterLength = duration
    s.append(n)
    
    # Write to MIDI file
    s.write('midi', fp=output_file)
    print(f"MIDI file saved as: {output_file}")

# Main execution
if __name__ == "__main__":
    # Path to the audio file
    audio_file = "pure_notes/c4.mp3"
    
    try:
        # Detect the dominant frequency
        freq = detect_dominant_frequency(audio_file)
        
        # Convert to MIDI
        create_midi_from_frequency(freq)
        
    except FileNotFoundError:
        print(f"Audio file '{audio_file}' not found!")
        print("Available files in pure_notes folder:")
        import os
        if os.path.exists("pure_notes"):
            files = os.listdir("pure_notes")
            for file in files:
                if file.endswith(('.mp3', '.wav')):
                    print(f"  {file}")
        else:
            print("  pure_notes folder not found!")
    except Exception as e:
        print(f"Error: {e}")
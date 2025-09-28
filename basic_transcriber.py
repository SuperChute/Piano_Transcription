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


#---------------------------------------------------------

def load_and_mix_signals(audio_files, sr=44100):
    """Load multiple mono files, align lengths, and mix into one signal."""
    signals = []
    for path in audio_files:
        sig, sr_loaded = librosa.load(path, sr=sr, mono=True)
        if sr_loaded != sr:
            # librosa.load already resampled to sr when sr=... is passed
            pass
        signals.append(sig)

    # Pad all to the same length (max of all)
    max_len = max(len(s) for s in signals)
    padded = [np.pad(s, (0, max_len - len(s))) for s in signals]

    # Mix by summing
    mixed = np.sum(np.stack(padded, axis=0), axis=0)
    return mixed, sr


def plot_wave_and_fft(signal, sr, max_freq_to_show):
    """Plot time-domain waveform and FFT magnitude of a signal."""
    # FFT
    ft = np.fft.rfft(signal)
    magnitude = np.abs(ft)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)

    threshold = 100.0
    note_peaks = {}

    print(f"Peaks ≥ {threshold} (up to {max_freq_to_show} Hz):")
    for i in range(1, len(magnitude) - 1):
        f = freqs[i]
        if f > max_freq_to_show:
            break
        # local maximum + threshold
        if magnitude[i] >= threshold and magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
            midi_num = frequency_to_midi(f)
            if midi_num is None:
                continue
            note_name = midi_to_note_name(midi_num)
            
            # keep the strongest magnitude for this note
            if note_name not in note_peaks or magnitude[i] > note_peaks[note_name][1]:
                note_peaks[note_name] = (f, magnitude[i])

    # print results
    for note_name, (f, m) in note_peaks.items():
        print(f"{f:7.2f} Hz  |  {m:8.2f}  |  {note_name}")

    # Limit spectrum to a readable range
    mask = freqs <= max_freq_to_show

    # Plot
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



if __name__ == "__main__":
    # Multiple note files (example)
    audio_files = [
        "pure_notes/C4_real.m4a",
        "pure_notes/e4.mp3",
        "pure_notes/g4.mp3",
        #"pure_notes/c4.mp3",
        #"pure_notes/c5.mp3"
        
    ]   

    freq_to_show = 1000

    try:
        mixed_signal, sr = load_and_mix_signals(audio_files, sr=44100)
        plot_wave_and_fft(mixed_signal, sr, freq_to_show)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Check your file paths. If needed, list available files in 'pure_notes/'.")
    except Exception as e:
        print(f"Error: {e}")

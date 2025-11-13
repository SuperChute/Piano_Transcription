import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, chord, midi, pitch, tempo 
from scipy.optimize import nnls
import librosa.display

# --- Configuration and Utility Functions ---

def frequency_to_midi(freq):
    """Convert frequency in Hz to MIDI note number using the A440 standard."""
    if freq <= 0:
        return None
    midi_note = (12 * np.log2(freq / 440)) + 69 
    return round(midi_note)

def midi_to_note_name(midi_note):
    """Convert MIDI note number to musical note name (e.g., 60 -> C4)."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note_name = notes[midi_note % 12]
    return f"{note_name}{octave}"

def chromatic_range(start='C3', end='C6', use_flats=True):
    """Build a chromatic list of names from start..end inclusive."""
    start_m = pitch.Pitch(start).midi
    end_m = pitch.Pitch(end).midi
    repl = {'C#':'Db','D#':'Eb','F#':'Gb','G#':'Ab','A#':'Bb'}

    names = []
    for m in range(start_m, end_m + 1):
        p = pitch.Pitch()
        p.midi = m
        name = p.nameWithOctave
        if use_flats:
            for sh, fl in repl.items():
                if name.startswith(sh):
                    name = name.replace(sh, fl, 1)
                    break
        names.append(name)
    return names


# --- Audio Preprocessing ---

def trim_to_fundamental(signal, sr=44100, skip_oscillations=10000, keep_oscillations=20000):
    """Trim audio to capture only the fundamental frequency."""
    max_idx = np.argmax(np.abs(signal))
    
    print(f"  Max amplitude at sample {max_idx} ({max_idx/sr:.4f} seconds)")
    
    start_idx = max_idx + skip_oscillations
    end_idx = start_idx + keep_oscillations
    
    if start_idx >= len(signal):
        print(f"  Warning: Start index {start_idx} exceeds signal length {len(signal)}")
        start_idx = max(0, len(signal) - keep_oscillations)
        end_idx = len(signal)
    elif end_idx > len(signal):
        print(f"  Warning: End index {end_idx} exceeds signal length {len(signal)}")
        end_idx = len(signal)
    
    trimmed = signal[start_idx:end_idx]
    
    print(f"  Trimmed: {len(trimmed)} samples ({len(trimmed)/sr:.4f} seconds)")
    
    return trimmed


# --- Frequency Binning ---

def create_frequency_bins(note_range, num_harmonics=10):
    """Create frequency bins centered around each note's fundamental and harmonics."""
    bin_centers = []
    
    for note_name in note_range:
        p = pitch.Pitch(note_name)
        fundamental = p.frequency
        
        for harmonic in range(1, num_harmonics + 1):
            bin_centers.append(fundamental * harmonic)
    
    bin_centers = sorted(set(bin_centers))
    return np.array(bin_centers)


def quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width=10.0):
    """Quantize full FFT spectrum into predefined frequency bins."""
    quantized_magnitude = np.zeros(len(bin_centers))
    
    for i, fc in enumerate(bin_centers):
        mask = (freqs >= fc - bin_width) & (freqs <= fc + bin_width)
        if np.any(mask):
            quantized_magnitude[i] = np.sum(magnitude[mask])
    
    return quantized_magnitude


# --- Basis Matrix Construction ---

def build_basis_matrix(note_files, bin_centers, sr=44100, bin_width=10.0, 
                      use_trimming=True, skip_oscillations=8000, keep_oscillations=20000,
                      plot_first_n=0):
    """Load each pure note, compute FFT, quantize to bins, and stack into matrix A."""
    basis_vectors = []
    note_names = []
    
    print("\n=== Building Basis Matrix ===")
    print(f"Trimming enabled: {use_trimming}")
    if use_trimming:
        print(f"Skip {skip_oscillations} samples, Keep {keep_oscillations} samples")
    
    for idx, note_file in enumerate(note_files):
        note_name = note_file.split('/')[-1].split('.')[0].upper()
        note_names.append(note_name)
        
        print(f"\nProcessing: {note_name}")
        
        if use_trimming:
            original_signal, _ = librosa.load(note_file, sr=sr, mono=True)
            signal = trim_to_fundamental(original_signal, sr, skip_oscillations, keep_oscillations)
            
            if plot_first_n > 0 and idx < plot_first_n:
                plot_trimming_comparison(original_signal, signal, sr, note_name)
        else:
            signal, _ = librosa.load(note_file, sr=sr, mono=True)
        
        signal = signal * np.hanning(len(signal))
        
        ft = np.fft.rfft(signal)
        magnitude = np.abs(ft)
        freqs = np.fft.rfftfreq(len(signal), 1/sr)
        
        quantized = quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width)
        max_val = np.max(quantized)
        quantized = quantized / max_val if max_val > 0 else quantized
        
        basis_vectors.append(quantized)
        print(f"  Max quantized bin = {np.max(quantized):.2f}")
    
    A = np.column_stack(basis_vectors)
    return A, note_names


# --- NNLS Solver ---

def detect_notes_nnls(A, b, note_names, threshold=0.3, top_k=None):
    """Solve Ax = b using NNLS (Non-Negative Least Squares) to detect notes."""
    weights, residual = nnls(A, b)
    
    print(f"\nSolution (weights):")
    for note, weight in zip(note_names, weights):
        if weight > 0.01:
            print(f"  {note}: {weight:.4f}")
    
    print(f"\nResidual error: {residual:.6f}")
    
    order = np.argsort(weights)[::-1]
    
    detected_notes = []
    for idx in order:
        if weights[idx] >= threshold:
            detected_notes.append((note_names[idx], (weights[idx])))
            if top_k is not None and len(detected_notes) == top_k:
                break
        else:
            break
    
    return detected_notes, weights


# --- Onset Detection ---

def detect_onsets(signal, sr=44100, hop_length=512):
    """Detect note onsets using librosa's built-in onset detection."""
    onset_frames = librosa.onset.onset_detect(
        y=signal,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units='frames'
    )
    
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    onset_times = onset_samples / sr
    
    print(f"\n=== Onset Detection ===")
    print(f"Detected {len(onset_samples)} onsets")
    print(f"Onset times (seconds): {onset_times}")
    
    return onset_samples, onset_times


def segment_and_detect_notes(signal, sr, onset_samples, A, note_names, bin_centers,
                             bin_width=10.0, threshold=0.35,
                             min_segment_samples=2048,
                             pre_roll_ms=25, analysis_window_ms=120,
                             top_k=None, polyphonic_threshold=0.25):
    """
    Segment audio by onsets and detect notes with NNLS.
    Supports both monophonic (top_k=1) and polyphonic (top_k=None) detection.
    
    Args:
        polyphonic_threshold: Lower threshold for polyphonic detection
    """
    print("\n=== Segmenting and Detecting Notes ===")
    print(f"Mode: {'MONOPHONIC (top note only)' if top_k == 1 else 'POLYPHONIC (all notes)'}")
    
    def ms_to_samples(ms):
        return int((ms / 1000.0) * sr)
    
    pre_samples = ms_to_samples(pre_roll_ms)
    window_samples = ms_to_samples(analysis_window_ms)
    NFFT = 8192
    
    segment_boundaries = np.append(onset_samples, len(signal))
    detected_timeline = []
    
    for i in range(len(segment_boundaries) - 1):
        onset_idx = segment_boundaries[i]
        next_onset_idx = segment_boundaries[i + 1]
        segment_duration = next_onset_idx - onset_idx
        
        print(f"\n--- Segment {i+1} ---")
        print(f"  Onset at: {onset_idx} ({onset_idx/sr:.3f}s)")
        
        if segment_duration < min_segment_samples:
            print(f"  ⚠️  Segment too short ({segment_duration} samples), skipping")
            continue
        
        start_ana = max(0, onset_idx - pre_samples)
        end_ana = min(len(signal), start_ana + window_samples)
        segment_trimmed = signal[start_ana:end_ana]
        
        print(f"  Analysis window: {start_ana} to {end_ana} ({len(segment_trimmed)} samples, {len(segment_trimmed)/sr*1000:.1f}ms)")
        
        if len(segment_trimmed) < NFFT:
            segment_padded = np.pad(segment_trimmed, (0, NFFT - len(segment_trimmed)))
        else:
            segment_padded = segment_trimmed[:NFFT]
        
        segment_windowed = segment_padded * np.hanning(NFFT)
        
        ft = np.fft.rfft(segment_windowed)
        magnitude = np.abs(ft)
        freqs = np.fft.rfftfreq(NFFT, 1/sr)
        
        b = quantize_fft_to_bins(freqs, magnitude, bin_centers, bin_width)
        b = b / np.max(b) if np.max(b) > 0 else b
        
        # Use polyphonic_threshold for polyphonic detection
        detection_threshold = threshold if top_k == 1 else polyphonic_threshold
        detected_notes, weights = detect_notes_nnls(A, b, note_names, detection_threshold, top_k=top_k)
        
        start_time = onset_idx / sr
        duration = segment_duration / sr
        
        if detected_notes:
            for note, weight in detected_notes:
                detected_timeline.append({
                    'note': note,
                    'start_time': start_time,
                    'duration': duration,
                    'weight': weight,
                    'velocity': int(min(weight * 127, 127))
                })
                print(f"  ✅ Detected: {note} (weight: {weight:.3f})")
        else:
            print(f"  ❌ No notes detected")
    
    return detected_timeline


def compute_tuning_compensation(signal, sr):
    """Compute global tuning offset in cents and return tuning ratio."""
    print("\n=== Computing Global Tuning ===")
    
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sr)
    tuning_cents = librosa.pitch_tuning(chroma)
    tuning_ratio = 2.0 ** (tuning_cents / 1200.0)
    
    print(f"Detected tuning: {tuning_cents:+.1f} cents")
    print(f"Tuning ratio: {tuning_ratio:.6f}")
    
    return tuning_ratio, tuning_cents


def create_midi_sequence(timeline, signal, sr, output_file='transcription.mid', fixed_bpm=None):
    """Create MIDI file with robust tempo detection or fixed BPM."""
    if not timeline:
        print("No notes in timeline to create MIDI.")
        return
    
    if fixed_bpm is not None:
        bpm = fixed_bpm
        print(f"\n🎵 Using fixed tempo: {bpm} BPM")
    else:
        oenv = librosa.onset.onset_strength(y=signal, sr=sr)
        tempo_est, _ = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
        bpm = int(round(float(tempo_est[0])))
        print(f"\n🎵 Detected tempo: {bpm} BPM")
    
    s = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=bpm))
    
    seconds_per_quarter = 60.0 / bpm
    
    # Group simultaneous notes into chords
    timeline_sorted = sorted(timeline, key=lambda x: x['start_time'])
    
    i = 0
    while i < len(timeline_sorted):
        current_time = timeline_sorted[i]['start_time']
        current_notes = []
        
        # Collect all notes that start at approximately the same time
        while i < len(timeline_sorted) and abs(timeline_sorted[i]['start_time'] - current_time) < 0.01:
            current_notes.append(timeline_sorted[i])
            i += 1
        
        try:
            if len(current_notes) == 1:
                # Single note
                n = note.Note(current_notes[0]['note'])
                n.offset = current_notes[0]['start_time'] / seconds_per_quarter
                n.quarterLength = current_notes[0]['duration'] / seconds_per_quarter
                n.volume.velocity = current_notes[0]['velocity']
                part.append(n)
            else:
                # Chord
                midi_pitches = [pitch.Pitch(event['note']).midi for event in current_notes]
                c = chord.Chord(midi_pitches)
                c.offset = current_time / seconds_per_quarter
                avg_duration = np.mean([event['duration'] for event in current_notes])
                c.quarterLength = avg_duration / seconds_per_quarter
                avg_velocity = int(np.mean([event['velocity'] for event in current_notes]))
                c.volume.velocity = avg_velocity
                part.append(c)
        except Exception as e:
            print(f"Warning: Could not add note(s): {e}")
    
    s.append(part)
    s.write('midi', fp=output_file)
    
    print(f"✅ MIDI sequence saved: {output_file}")
    print(f"   Total events: {len(timeline)}")
    if timeline:
        print(f"   Duration: {timeline[-1]['start_time'] + timeline[-1]['duration']:.2f} seconds")


# --- Visualization ---

def plot_basis_matrix(A, note_names, bin_centers):
    """Visualize the basis matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(A, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Normalized Magnitude')
    plt.xlabel('Note')
    plt.ylabel('Frequency Bin Index')
    plt.title('Basis Matrix A (Frequency Bins × Notes)')
    plt.xticks(range(len(note_names)), note_names, rotation=45)
    plt.yticks(range(0, len(bin_centers), max(1, len(bin_centers)//20)), 
               [f"{bin_centers[i]:.0f} Hz" for i in range(0, len(bin_centers), max(1, len(bin_centers)//20))])
    plt.tight_layout()
    plt.show()


def plot_mixed_signal(mixed_signal, sr, title="Audio Signal"):
    """Plot the time-domain waveform."""
    time = np.arange(len(mixed_signal)) / sr
    
    plt.figure(figsize=(14, 5))
    plt.plot(time, mixed_signal, linewidth=0.5, alpha=0.8)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_onsets_on_waveform(signal, sr, onset_samples, title="Detected Onsets"):
    """Visualize onsets overlaid on the audio waveform."""
    time = np.arange(len(signal)) / sr
    onset_times = onset_samples / sr
    
    plt.figure(figsize=(16, 6))
    plt.plot(time, signal, linewidth=0.5, alpha=0.7, label='Audio Signal')
    
    for onset_time in onset_times:
        plt.axvline(x=onset_time, color='red', linestyle='--', 
                   linewidth=2, alpha=0.8)
    
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


def plot_trimming_comparison(original_signal, trimmed_signal, sr, note_name):
    """Visualize the effect of trimming on the audio signal."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    max_idx = np.argmax(np.abs(original_signal))
    
    time_orig = np.arange(len(original_signal)) / sr
    ax1.plot(time_orig, original_signal, linewidth=0.5, alpha=0.8, color='blue')
    
    ax1.axvline(x=max_idx/sr, color='red', linestyle='--', linewidth=2, 
                label=f'Max Amplitude (sample {max_idx})')
    ax1.scatter([max_idx/sr], [original_signal[max_idx]], color='red', s=100, zorder=5)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'BEFORE Trimming: {note_name} ({len(original_signal)} samples, {len(original_signal)/sr:.3f}s)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    time_trim = np.arange(len(trimmed_signal)) / sr
    ax2.plot(time_trim, trimmed_signal, linewidth=0.5, alpha=0.8, color='green')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title(f'AFTER Trimming: {note_name} ({len(trimmed_signal)} samples, {len(trimmed_signal)/sr:.3f}s) - Clean Fundamental', 
                  fontsize=14, fontweight='bold', color='green')
    ax2.grid(True, alpha=0.3)
    
    reduction_pct = (1 - len(trimmed_signal)/len(original_signal)) * 100
    ax2.text(0.02, 0.98, f'Removed {reduction_pct:.1f}% of signal\nKept clean fundamental frequency', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    note_range = chromatic_range('C2', 'C7', use_flats=True)
    num_harmonics = 8
    bin_width = 7.5
    sr = 44100
    
    # Trimming parameters for basis matrix
    use_trimming = True
    skip_oscillations = 0
    keep_oscillations = 20000
    
    # Detection parameters
    MONOPHONIC_MODE = False  # Set to True for melody, False for chords
    monophonic_threshold = 0.35
    polyphonic_threshold = 0.24
    
    print("=== Creating Frequency Bins ===")
    bin_centers = create_frequency_bins(note_range, num_harmonics=num_harmonics)
    print(f"Created {len(bin_centers)} frequency bins")
    print(f"Frequency range: {bin_centers[0]:.2f} Hz to {bin_centers[-1]:.2f} Hz") 

    """ 
            #"pure_notes/c1.mp3", "pure_notes/db1.mp3", "pure_notes/d1.mp3", 
        #"pure_notes/eb1.mp3", "pure_notes/e1.mp3", "pure_notes/f1.mp3", 
        #"pure_notes/gb1.mp3", "pure_notes/g1.mp3", "pure_notes/ab1.mp3", 
        #"pure_notes/a1.mp3", "pure_notes/bb1.mp3", "pure_notes/b1.mp3",

        "pure_notes/c2.mp3", "pure_notes/db2.mp3", "pure_notes/d2.mp3", 
        "pure_notes/eb2.mp3", "pure_notes/e2.mp3", "pure_notes/f2.mp3", 
        "pure_notes/gb2.mp3", "pure_notes/g2.mp3", "pure_notes/ab2.mp3", 
        "pure_notes/a2.mp3", "pure_notes/bb2.mp3", "pure_notes/b2.mp3",

        "pure_notes/c3.mp3", "pure_notes/db3.mp3", "pure_notes/d3.mp3", 
        "pure_notes/eb3.mp3", "pure_notes/e3.mp3", "pure_notes/f3.mp3", 
        "pure_notes/gb3.mp3", "pure_notes/g3.mp3", "pure_notes/ab3.mp3", 
        "pure_notes/a3.mp3", "pure_notes/bb3.mp3", "pure_notes/b3.mp3",

        "pure_notes/c4.mp3", "pure_notes/db4.mp3", "pure_notes/d4.mp3", 
        "pure_notes/eb4.mp3", "pure_notes/e4.mp3","pure_notes/f4.mp3", 
        "pure_notes/gb4.mp3", "pure_notes/g4.mp3", "pure_notes/ab4.mp3", 
        "pure_notes/a4.mp3","pure_notes/bb4.mp3", "pure_notes/b4.mp3",

        "pure_notes/c5.mp3", "pure_notes/db5.mp3", "pure_notes/d5.mp3",
        "pure_notes/eb5.mp3", "pure_notes/e5.mp3","pure_notes/f5.mp3",
        "pure_notes/gb5.mp3", "pure_notes/g5.mp3","pure_notes/ab5.mp3", 
        "pure_notes/a5.mp3","pure_notes/bb5.mp3", "pure_notes/b5.mp3", 

        "pure_notes/c6.mp3", "pure_notes/db6.mp3", "pure_notes/d6.mp3",
        "pure_notes/eb6.mp3", "pure_notes/e6.mp3","pure_notes/f6.mp3",
        "pure_notes/gb6.mp3", "pure_notes/g6.mp3","pure_notes/ab6.mp3", 
        "pure_notes/a6.mp3","pure_notes/bb6.mp3", "pure_notes/b6.mp3",  

        "pure_notes/c7.mp3",
    ]
    """
    
    # Build basis matrix
    note_files = [
        "pure_notes/c2.mp3", "pure_notes/db2.mp3", "pure_notes/d2.mp3", 
        "pure_notes/eb2.mp3", "pure_notes/e2.mp3", "pure_notes/f2.mp3", 
        "pure_notes/gb2.mp3", "pure_notes/g2.mp3", "pure_notes/ab2.mp3", 
        "pure_notes/a2.mp3", "pure_notes/bb2.mp3", "pure_notes/b2.mp3",

        "pure_notes/c3.mp3", "pure_notes/db3.mp3", "pure_notes/d3.mp3", 
        "pure_notes/eb3.mp3", "pure_notes/e3.mp3", "pure_notes/f3.mp3", 
        "pure_notes/gb3.mp3", "pure_notes/g3.mp3", "pure_notes/ab3.mp3", 
        "pure_notes/a3.mp3", "pure_notes/bb3.mp3", "pure_notes/b3.mp3",

        "pure_notes/c4.mp3", "pure_notes/db4.mp3", "pure_notes/d4.mp3", 
        "pure_notes/eb4.mp3", "pure_notes/e4.mp3","pure_notes/f4.mp3", 
        "pure_notes/gb4.mp3", "pure_notes/g4.mp3", "pure_notes/ab4.mp3", 
        "pure_notes/a4.mp3","pure_notes/bb4.mp3", "pure_notes/b4.mp3",

        "pure_notes/c5.mp3", "pure_notes/db5.mp3", "pure_notes/d5.mp3",
        "pure_notes/eb5.mp3", "pure_notes/e5.mp3","pure_notes/f5.mp3",
        "pure_notes/gb5.mp3", "pure_notes/g5.mp3","pure_notes/ab5.mp3", 
        "pure_notes/a5.mp3","pure_notes/bb5.mp3", "pure_notes/b5.mp3", 

        "pure_notes/c6.mp3", "pure_notes/db6.mp3", "pure_notes/d6.mp3",
        "pure_notes/eb6.mp3", "pure_notes/e6.mp3","pure_notes/f6.mp3",
        "pure_notes/gb6.mp3", "pure_notes/g6.mp3","pure_notes/ab6.mp3", 
        "pure_notes/a6.mp3","pure_notes/bb6.mp3", "pure_notes/b6.mp3",  

        "pure_notes/c7.mp3",
    ]
    
    try:
        A, note_names = build_basis_matrix(
            note_files, bin_centers, sr=sr, bin_width=bin_width,
            use_trimming=use_trimming, 
            skip_oscillations=skip_oscillations,
            keep_oscillations=keep_oscillations,
            plot_first_n=0
        )
        print(f"\n=== Basis Matrix Built ===")
        print(f"Shape: {A.shape} ({A.shape[0]} bins × {A.shape[1]} notes)")
        
        plot_basis_matrix(A, note_names, bin_centers)
        
        # Load test audio
        print("\n" + "="*60)
        print("=== UNIFIED NOTE DETECTION ===")
        print(f"Mode: {'MONOPHONIC' if MONOPHONIC_MODE else 'POLYPHONIC'}")
        print("="*60)
        
        test_audio_file = "sequential/chopsticks.mp3" 
        
        print(f"\nLoading: {test_audio_file}")
        signal, _ = librosa.load(test_audio_file, sr=sr, mono=True)
        print(f"Audio length: {len(signal)} samples ({len(signal)/sr:.2f} seconds)")
        
        plot_mixed_signal(signal, sr, title="Input Audio")
        
        # Optional: Tuning compensation
        #tuning_ratio, tuning_cents = compute_tuning_compensation(signal, sr)
        #bin_centers_tuned = bin_centers * tuning_ratio
        
        # Detect onsets
        onset_samples, onset_times = detect_onsets(
            signal, sr=sr, 
            hop_length=512, # Can be 256, 128
        )
        
        #plot_onsets_on_waveform(signal, sr, onset_samples, 
                                #title="Onset Detection Results")
        
        # Segment and detect notes
        timeline = segment_and_detect_notes(
            signal=signal,
            sr=sr,
            onset_samples=onset_samples,
            A=A,
            note_names=note_names,
            bin_centers=bin_centers,
            bin_width=bin_width,
            threshold=monophonic_threshold,
            min_segment_samples=(1024 * 2), # Can be 1024, 256, 128
            pre_roll_ms=-25,
            analysis_window_ms=200, #Window of analysis to capture after an onset in ms
            top_k=None if MONOPHONIC_MODE else None,
            polyphonic_threshold=polyphonic_threshold
        )
        
        # Display results
        print("\n" + "="*60)
        print("TRANSCRIPTION RESULTS")
        print("="*60)
        
        if timeline:
            print(f"\n✅ Detected {len(timeline)} note events:")
            for i, event in enumerate(timeline, 1):
                print(f"  {i}. {event['note']:4s} @ {event['start_time']:.3f}s "
                      f"(duration: {event['duration']:.3f}s, weight: {event['weight']:.3f})")
        else:
            print("\n❌ No notes detected")
        
        # Create MIDI
        if timeline:
            create_midi_sequence(timeline, signal, sr, 
                              output_file='unified_transcription.mid',
                              fixed_bpm=None)
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
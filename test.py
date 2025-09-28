import numpy as np

def create_piano_templates(max_freq=2000, num_harmonics=8):
    """
    Create harmonic templates for piano notes from C4 to C5.
    Each template contains the fundamental frequency and its harmonics.
    
    Args:
        max_freq: Maximum frequency to include harmonics up to
        num_harmonics: Maximum number of harmonics to include per note
    
    Returns:
        dict: Note name -> list of harmonic frequencies
    """
    
    # Base frequencies for notes C4 to C5 (one octave)
    base_frequencies = {
        'C4': 261.63,
        'C#4': 277.18, 
        'D4': 293.66,
        'D#4': 311.13,
        'E4': 329.63,
        'F4': 349.23,
        'F#4': 369.99,
        'G4': 392.00,
        'G#4': 415.30,
        'A4': 440.00,
        'A#4': 466.16,
        'B4': 493.88,
        'C5': 523.25
    }
    
    templates = {}
    
    for note_name, fundamental_freq in base_frequencies.items():
        harmonics = []
        
        # Generate harmonics: f, 2f, 3f, 4f, etc.
        for harmonic_num in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * harmonic_num
            
            # Only include harmonics below our max frequency
            if harmonic_freq <= max_freq:
                harmonics.append(harmonic_freq)
            else:
                break
                
        templates[note_name] = harmonics
    
    return templates

def print_templates(templates):
    """Print the harmonic templates in a readable format"""
    print("Piano Note Harmonic Templates:")
    print("=" * 50)
    
    for note, harmonics in templates.items():
        print(f"{note:4}: {[f'{h:.1f}' for h in harmonics]}")

# Example usage
if __name__ == "__main__":
    # Create templates
    templates = create_piano_templates()
    print_templates(templates)
    
    print("\nExample - C4 and C5 harmonics:")
    print(f"C4 harmonics: {templates['C4']}")
    print(f"C5 harmonics: {templates['C5']}")
    
    print(f"\nC4's 2nd harmonic: {templates['C4'][1]:.1f} Hz")
    print(f"C5's fundamental:  {templates['C5'][0]:.1f} Hz")
    print("^ These overlap! But C4 has 784Hz, C5 has 1046Hz")
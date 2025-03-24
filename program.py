



"""
Hi.
First of all sorry.
The code is documented by chat GPT because
I realy didnt had time the document it by
my self.
Most of the code is mine, But some I stole...
good luck understanding the code im sorry.

if you have any question ask me
"""
import re
import numpy as np
import sounddevice as sd

import time
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr  # noise reduction library
from numpy.linalg import norm

# Frequency mapping of musical notes (A4 = 440 Hz, C4 = 261.63 Hz, etc.)
NOTE_FREQUENCIES = {
    "C": 261.63, "C#": 277.18, "Db": 277.18, "D": 293.66, "D#": 311.13, "Eb": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "Gb": 369.99, "G": 392.00, "G#": 415.30,
    "Ab": 415.30, "A": 440.00, "A#": 466.16, "Bb": 466.16, "B": 493.88
}

# Chord structures (semitone offsets from the root)
CHORD_TYPES = {
    "": [0, 4, 7],         # Major
    "m": [0, 3, 7],        # Minor
    "7": [0, 4, 7, 10],    # Dominant 7
    "maj7": [0, 4, 7, 11], # Major 7
    "m7": [0, 3, 7, 10],   # Minor 7
    "dim": [0, 3, 6],      # Diminished
    "5": [0, 7],           # Power chord
    "4": [0, 5, 7]         # Suspended 4
}

def playChordSequence(chord_names, durations, sample_rate=44100):
    """Play a sequence of chords with no gap by concatenating their waveforms.
       chord_names: list of chord names (e.g., ["Cmaj7", "Am"])
       durations: list of durations for each chord (in seconds)
    """
    sequence_wave = np.array([], dtype=np.float32)
    
    for chord_name, duration in zip(chord_names, durations):
        # Generate waveform for this chord
        # (Assume playChord() logic is adapted to return the waveform)
        wave = chord_wave_for(chord_name, duration, sample_rate)
        # Concatenate with the sequence
        sequence_wave = np.concatenate((sequence_wave, wave))
    
    # Play the concatenated waveform
    sd.play(sequence_wave, samplerate=sample_rate)
    sd.wait()

def chord_wave_for(chord_name, duration=1.0, sample_rate=44100):
    """Return the waveform for a given chord name, instead of playing it immediately."""
    import re
    # Use regex to extract the root and quality.
    m = re.match(r'^([A-Ga-g][#b]?)(.*)$', chord_name.strip())
    if not m:
        print(f"Invalid chord format: {chord_name}")
        return np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    root, suffix = m.groups()
    root = root.upper()
    suffix = suffix.strip()
    if suffix == "":
        suffix = ""  # Major chord

    if root not in NOTE_FREQUENCIES or suffix not in CHORD_TYPES:
        print(f"Unknown chord: {chord_name}")
        return np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    base_freq = NOTE_FREQUENCIES[root]
    offsets = CHORD_TYPES[suffix]
    freqs = [base_freq * (2 ** (offset / 12)) for offset in offsets]
    
    waves = [generate_sine_wave(freq, duration, sample_rate) for freq in freqs]
    chord_wave = sum(waves) / len(waves)
    return chord_wave

def choose_output_device():
    """
    Selects an output device. Returns the index of the first device
    that has 'Speakers' in its name, or the first available output device.
    """
    devices = sd.query_devices()
    output_devices = [(i, d) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
    for i, d in output_devices:
        if "Speakers" in d["name"]:
            return i
    return output_devices[0][0] if output_devices else None

def generate_sine_wave(freq, duration=1.0, sample_rate=44100):
    """Generate a sine wave for a given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave

def playChord(chord_name, duration=1.0):
    """Convert a chord name (e.g., 'Cmaj7') to frequencies and play the sound."""
    # Set output device automatically
    device = choose_output_device()
    if device is not None:
        sd.default.device = device
        print(f"Using output device {device}: {sd.query_devices(device)['name']}")
    else:
        print("No output device found.")
        return

    # Use regex to extract the root and quality.
    # This regex captures a letter A-G (case-insensitive), an optional '#' or 'b', then the rest.
    m = re.match(r'^([A-Ga-g][#b]?)(.*)$', chord_name.strip())
    if not m:
        print(f"Invalid chord format: {chord_name}")
        return

    root, suffix = m.groups()
    root = root.upper()
    suffix = suffix.strip()  # For example, "maj7", "m", "7", etc.

    # For a chord like "C" with no quality, set suffix to "" (major)
    if suffix == "":
        suffix = ""

    if root not in NOTE_FREQUENCIES or suffix not in CHORD_TYPES:
        print(f"Unknown chord: {chord_name} -> root: {root}, suffix: {suffix}")
        return

    base_freq = NOTE_FREQUENCIES[root]
    offsets = CHORD_TYPES[suffix]

    # Convert semitone offsets to frequencies
    freqs = [base_freq * (2 ** (offset / 12)) for offset in offsets]

    # Generate waves for all frequencies
    sample_rate = 44100
    waves = [generate_sine_wave(freq, duration, sample_rate) for freq in freqs]

    # Mix waves together (average them)
    chord_wave = sum(waves) / len(waves)

    # Play the sound (blocking mode)
    sd.play(chord_wave, samplerate=sample_rate, blocking=False)
    sd.wait()



def chord_to_string(chord):
    """
    chord looks like this:
    [3,1] = Dm. Explanation:
    At index [0] we have 3 which is equivalent to 'D' chord
    At index [1] we have 1 which is equivalent to 'm' (Minor) chord prefix.

    Prefix list:
    1 = 'm' (Minor)
    2 = '7' (Dominant 7)
    3 = '4' (Suspended 4)
    4 = 'maj7' (Major 7) 
    5 = '5' (power chord / fifth)
    6 = 'dim' (diminished)
    """
    chordString = ""
    chord[0] = (chord[0] - 1) % 12 + 1  # Wrapping the root note between 1-12 correctly
    if chord[0] == 1:
        chordString = 'C'
    elif chord[0] == 2:
        chordString = 'Db'
    elif chord[0] == 3:
        chordString = 'D'
    elif chord[0] == 4:
        chordString = 'Eb'
    elif chord[0] == 5:
        chordString = 'E'
    elif chord[0] == 6:
        chordString = 'F'
    elif chord[0] == 7:
        chordString = 'Gb'
    elif chord[0] == 8:
        chordString = 'G'
    elif chord[0] == 9:
        chordString = 'Ab'
    elif chord[0] == 10:
        chordString = 'A'
    elif chord[0] == 11:
        chordString = 'Bb'
    elif chord[0] == 12:
        chordString = 'B'
    else:
        return 'Unknown'

    if len(chord) > 1:  # only if it has a prefix
        if chord[1] == 1:
            chordString += 'm'
        elif chord[1] == 2:
            chordString += '7'
        elif chord[1] == 3:
            chordString += '4'
        elif chord[1] == 4:
            chordString += 'maj7'
        elif chord[1] == 5:
            chordString += '5'
        elif chord[1] == 6:
            chordString += 'dim'
    return chordString

def string_to_chord(chord):
    chordInt = [0, 0]
    chord = chord.strip()  # Strip whitespace
    chord = chord[0].upper() + chord[1:]  # First letter to uppercase
    # map root notes to numbers
    if chord.startswith('C'):
        chordInt[0] = 1
    elif chord.startswith('Db') or chord.startswith('C#'):
        chordInt[0] = 2
    elif chord.startswith('D'):
        chordInt[0] = 3
    elif chord.startswith('Eb') or chord.startswith('D#'):
        chordInt[0] = 4
    elif chord.startswith('E'):
        chordInt[0] = 5
    elif chord.startswith('F'):
        chordInt[0] = 6
    elif chord.startswith('Gb') or chord.startswith('F#'):
        chordInt[0] = 7
    elif chord.startswith('G'):
        chordInt[0] = 8
    elif chord.startswith('Ab') or chord.startswith('G#'):
        chordInt[0] = 9
    elif chord.startswith('A'):
        chordInt[0] = 10
    elif chord.startswith('Bb') or chord.startswith('A#'):
        chordInt[0] = 11  # A# corresponds to 11, Bb also corresponds to 11
    elif chord.startswith('B'):
        chordInt[0] = 12
    else:
        return [0, 0]  # Unknown chord

    # everything after the root note
    root_length = 2 if chord[:2] in ['Db', 'Eb', 'Gb', 'Ab', 'Bb','C#','D#','F#','G#','A#'] else 1
    suffix = chord[root_length:]

    # set chord types correctly
    if suffix == "m":
        chordInt[1] = 1  # Minor
    elif suffix == "7":
        chordInt[1] = 2  # Dominant 7
    elif suffix == "4":
        chordInt[1] = 3  # Suspended 4
    elif suffix == "maj7":
        chordInt[1] = 4  # Major 7
    elif suffix == '5':
        chordInt[1] = 5  # Power chord / fifth
    elif suffix == 'dim':
        chordInt[1] = 6  # Diminished

    return chordInt


def up_pitch(chord, amount= -1):
    chord_to_down = [0, 0]
    if isinstance(chord, str):  # check if string if yes convert
        chord_to_down = string_to_chord(chord)  # convert string to numeric chord

    if chord_to_down == [0, 0]:  # check the converted value
        return "Unknown"
    # move the root note up by 'amount'. wrapping within 1-12
    chord_to_down[0] = (chord_to_down[0] + amount - 1) % 12 + 1  # Wrapping 1-12

    # Fix: Treating A# -> B correctly
    if chord_to_down[0] == 11:
        chord_to_down[0] = 12  # A# should move to B directly.

    return chord_to_string(chord_to_down)


def key_pitch_change(chords, amount = -1):
    for i in range(len(chords)):
        chords[i] = up_pitch(chords[i], amount)  # pass each chord to up_pitch
    return chords


def normalize_chord(chord):
    chord = chord[0].upper() + chord[1:] # first letter to upperCasea
    # Simple normalization: convert common flat names to sharps.
    replacements = {
        "Db": "C#",
        "Eb": "D#",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#"
    }
    for flat, sharp in replacements.items():
        if chord.startswith(flat):
            return chord.replace(flat, sharp, 1)
    return chord

def get_key_from_chords(chords):
    # Define diatonic chords for each major key (using sharps)
    diatonic_chords = {
        "C":  ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
        "C#": ["C#", "D#m", "Fm", "F#", "G#", "A#m", "Ddim"],
        "D":  ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"],
        "D#": ["D#", "Fm", "Gm", "G#", "A#", "Cm", "Ddim"],
        "E":  ["E", "F#m", "G#m", "A", "B", "C#m", "D#dim"],
        "F":  ["F", "Gm", "Am", "A#", "C", "Dm", "Edim"],
        "F#": ["F#", "G#m", "A#m", "B", "C#", "D#m", "E#dim"],
        "G":  ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
        "G#": ["G#", "A#m", "Cm", "C#", "D#", "Fm", "Gdim"],
        "A":  ["A", "Bm", "C#m", "D", "E", "F#m", "G#dim"],
        "A#": ["A#", "Cm", "Dm", "D#", "F", "Gm", "Adim"],
        "B":  ["B", "C#m", "D#m", "E", "F#", "G#m", "A#dim"]
    }
    
    # Normalize and simplify chords: remove "maj7", "dim", etc.
    basic_chords = []
    for chord in chords:
        norm = normalize_chord(chord)
        if norm.endswith("maj7"):
            norm = norm[:-4]
        elif norm.endswith("dim"):
            norm = norm[:-3]
        basic_chords.append(norm)
    
    # Score each key by counting how many chords from basic_chords appear in its diatonic set.
    scores = {}
    for key, diatonic in diatonic_chords.items():
        score = sum(1 for chord in basic_chords if chord in diatonic)
        scores[key] = score
    
    # Choose the key with the highest score, if at least one chord matched.
    best_key = max(scores, key=scores.get)
    if scores[best_key] == 0:
        return "Unknown"
    return best_key

def get_user_chords(num=4):
    chords_list = []
    while len(chords_list) < num:
        chord_input = input().strip()
        # Optionally, check if the chord looks valid (for now, just check length)
        if len(chord_input) < 1:
            print("Please enter a valid chord.")
            continue
        chords_list.append(chord_input)
    return chords_list



# Chord template: an array of length 12 corresponding to pitch classes C, C#, D, ... B.
def major_template(root_index):
    # root_index is 0 for C, 1 for C#/Db, etc.
    template = np.zeros(12)
    # Major chord: root, major third (4 semitones), perfect fifth (7 semitones)
    template[root_index] = 1
    template[(root_index + 4) % 12] = 1
    template[(root_index + 7) % 12] = 1
    return template

def minor_template(root_index):
    # Minor chord: root, minor third (3 semitones), perfect fifth (7 semitones)
    template = np.zeros(12)
    template[root_index] = 1
    template[(root_index + 3) % 12] = 1
    template[(root_index + 7) % 12] = 1
    return template

# Build a dictionary of templates for all 12 major and 12 minor chords.
CHORD_TEMPLATES = {}
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
for i, note in enumerate(notes):
    CHORD_TEMPLATES[note] = major_template(i)
    CHORD_TEMPLATES[note + "m"] = minor_template(i)



def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

def recognize_chord(chroma_avg):
    best_chord = None
    best_similarity = -1
    for chord_name, template in CHORD_TEMPLATES.items():
        similarity = cosine_similarity(chroma_avg, template)
        if similarity > best_similarity:
            best_similarity = similarity
            best_chord = chord_name
    return best_chord, best_similarity

def average_chroma(chroma):
    """
    Average the chroma feature across time.
    Returns a vector of length 12.
    """
    return np.mean(chroma, axis=1)

def get_chord_from_sound(time = 2):
    duration = time
      # seconds
    fs = 44100      # sample rate
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete!")
    
    y = recording.flatten()  # Convert to 1D array
    sr = fs
    
    # Apply noise reduction here if desired (e.g., using noisereduce)
    # For now, we continue with y.
    
    # Extract chroma features using librosa:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    """Hi this is me again i kept this for you
    to see how its working with the plt.show() ect."""
    # Optionally, display the chroma for debugging:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.title('Chroma Feature')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    # Compute the average chroma vector:
    chroma_avg = average_chroma(chroma)
    
    # Recognize the chord based on the template with highest cosine similarity:
    chord, similarity = recognize_chord(chroma_avg)
    print(f"Predicted chord: {chord} (similarity: {similarity:.2f})")
    return chord

    
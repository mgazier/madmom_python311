"""
Simple demo for use of madmom with Python 3.11

Imports an audio file
Applies madmom RNNDownBeatProcessor() to get beat activations
Applies madmom DBNDownBeatTrackingProcessor() to get beat times and downbeats.
    - uses myDBNDownBeatTrackingProcessor() to operate in the current environment

(c) Michael Gazier, 2024
MIT License
"""
import numpy as np
from pydub import AudioSegment
import os
import librosa

# fix Python 3.11 issues in madmom
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence  # Alias MutableSequence to avoid deprecation issues
np.float = float
np.int = int
# import madmom
from madmom.features.downbeats import RNNDownBeatProcessor
from my_madmom_downbeat_processor import myDBNDownBeatTrackingProcessor

def get_waveform(path):
    """
    Load the waveform using pydub and retrieve the sample rate from the file.
    Should be 44.1kHz for Madmom
    Any format with ffmpeg
    """
    print("Loading waveform using pydub...")
    audio = AudioSegment.from_file(path)
    _sr = audio.frame_rate  # Get the sample rate from the file

    # Resample and convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Convert pydub audio to numpy array
    # Normalization: The division by 2**15, 2**31, or 2**7 is to convert integer
    # PCM values to floating-point format, similar to what librosa.load provides by default
    _samples = np.array(audio.get_array_of_samples())
    if audio.sample_width == 2:
        _samples = _samples.astype(np.float32) / 2 ** 15
    elif audio.sample_width == 4:
        _samples = _samples.astype(np.float32) / 2 ** 31
    else:
        _samples = _samples.astype(np.float32) / 2 ** 7  # Assuming 8-bit audio

    # Normalize by the number of channels
    _samples = _samples / audio.channels

    print("Waveform loaded successfully.")
    return _samples, _sr

def analyze(y, _sr, min_bpm=55, max_bpm=170):
    """
    Analyze audio data for beats, downbeats, tempo, and musical notes.
    """
    print("Analyzing audio data...")
    _data = {
        'sample_rate': _sr,
        'duration': librosa.get_duration(y=y, sr=_sr),
    }

    # Processing beats and downbeats using the custom downbeat processor
    print("Executing RNNDownBeatProcessor...")
    downbeat_processor = RNNDownBeatProcessor()
    activations = downbeat_processor(y)
    # print("Processing activations with myDBNDownBeatTrackingProcessor...") # message in myDBNDownBeatTrackingProcessor() if quiet=False
    dbn_processor = myDBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100, min_bpm=min_bpm, max_bpm=max_bpm, quiet=False)
    beats_and_downbeats = dbn_processor(activations)

    beat_times = beats_and_downbeats[:, 0]
    beat_positions_in_bar = beats_and_downbeats[:, 1].astype(int)  # Ensuring positions are integer

    # Data aggregation for output
    _data['number_of_beats'] = len(beat_times)
    _data['beat_times'] = beat_times

    # Extracting downbeats (first beat of each bar)
    downbeats = beats_and_downbeats[beat_positions_in_bar == 1]
    _data['downbeats'] = list(zip(downbeats[:, 0], ['Beat 1'] * len(downbeats)))  # assuming beat 1 represents downbeat

    # Compute the tempo if enough beats are detected
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)  # Calculate the differences between consecutive beat times
        tempo = 60 / np.median(intervals) if intervals.size > 0 else 0  # Median interval converted to BPM
    else:
        tempo = 0  # Default to zero if not enough beats to calculate intervals
    _data['tempo_float'] = tempo

    print("Extracting spectral features... ")
    print("Lists the notes / intensity at each beat ordered by their chroma intensity from highest to lowest.")
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=_sr, hop_length=512)
        frame_times = librosa.frames_to_time(range(chroma.shape[1]), sr=_sr, hop_length=512)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Map beat times to the nearest chroma frame and retrieve corresponding notes
        beat_frames = np.searchsorted(frame_times, beat_times, side='left')
        beat_notes = [
            sorted(
                [(note_names[i], chroma[i, frame]) for i in np.where(chroma[:, frame] >= MIN_CHROMA)[0]],
                key=lambda x: x[1],
                reverse=True
            )
            if frame < chroma.shape[1] else [('N/A', 0)]
            for frame in beat_frames
        ]

        # Combining beat positions, times, and notes
        _data['all_beats'] = list(zip(range(len(beat_times)), beat_times, beat_positions_in_bar, beat_notes))
    except Exception as e:
        print(f"Error processing chroma features: {e}")
        _data.update({'notes': [0], 'dominant_note': 0})

    print("Completed audio analysis...")
    return _data

def print_data(_data, _filename):
    """
    Print the results of the audio analysis.
    """
    print(f"\nRESULTS OF ANALYSIS for '{_filename}'\n=========================================")

    # Printing detailed information about beats and downbeats
    print(f"All Detected Beats (index, beat #, time, and note with chroma [0-1, cutoff notes at {MIN_CHROMA}]):")

    for idx, time, beat_num, notes in _data['all_beats']:
        minutes, seconds = divmod(time, 60)
        time_str = f"{int(minutes)}m:{seconds:.2f}s"
        notes_str = ', '.join(notes)
        print(f"Index: {idx}, Beat #: {beat_num}, Time: {time_str}, Notes: {notes_str}")

    # Printing basic audio and analysis metrics
    print("\nGeneral Analysis Results:")
    print(f"Sample Rate: {_data['sample_rate']} Hz")
    minutes, seconds = divmod(_data['duration'], 60)
    time_str = f"{int(minutes)}m:{seconds:.2f}s"
    print(f"Total Duration: {time_str}")
    print(f"Total Number of Beats: {_data['number_of_beats']}")
    # Printing tempo calculated from beat intervals
    print(f"Calculated Tempo: {_data['tempo_float']:.1f} BPM")

def print_data(_data, _filename):
    """
    Print the results of the audio analysis.
    """
    print(f"\nRESULTS OF ANALYSIS for {_filename}\n=========================================")

    # Printing detailed information about beats and downbeats
    print(f"All Detected Beats (index, beat #, time, and note with chroma >= {MIN_CHROMA} [chroma = 0-1 = harmonic content of the audio at each beat]):")

    for idx, time, beat_num, notes in _data['all_beats']:
        minutes, seconds = divmod(time, 60)
        time_str = f"{int(minutes)}m:{seconds:.2f}s"
        note_names = ', '.join([note for note, _ in notes])
        chroma_values = ', '.join([f"{chroma:.2f}" for _, chroma in notes])
        notes_str = f"{note_names} [{chroma_values}]"
        print(f"Index: {idx}, Beat #: {beat_num}, Time: {time_str}, Notes [Chroma]: {notes_str}")

    # Printing basic audio and analysis metrics
    print("\nGeneral Analysis Results:")
    print(f"Sample Rate: {_data['sample_rate']} Hz")
    minutes, seconds = divmod(_data['duration'], 60)
    time_str = f"{int(minutes)}m:{seconds:.2f}s"
    print(f"Total Duration: {time_str}")
    print(f"Total Number of Beats: {_data['number_of_beats']}")
    # Printing tempo calculated from beat intervals
    print(f"Calculated Tempo: {_data['tempo_float']:.1f} BPM")

if __name__ == '__main__':
    ROOT_DIR = "/my/root/path/"
    filename = "downbeat tracker test.m4a"
    filePath = os.path.join(ROOT_DIR, filename)
    samples, sr = get_waveform(filePath) # should be 44.1kHz
    MIN_CHROMA = 0.4  # default is 0.5
    data = analyze(samples, sr)
    print_data(data, filename)
    print("Analysis complete.")
"""
Simple demo for use of madmom with Python 3.11

Imports an audio file
Applies madmom RNNDownBeatProcessor() to get beat activations
Applies madmom DBNDownBeatTrackingProcessor() to get beat times and downbeats.
    - uses myDBNDownBeatTrackingProcessor() to operate in the current environment

(c) Michael Gazier, 2024
MIT License
"""
from pydub import AudioSegment
import os
import librosa
from datetime import datetime

# fix Python 3.11 issues in madmom
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence  # Alias MutableSequence to avoid deprecation issues
import numpy as np
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

    # Calculate the duration
    duration_ms = len(_samples) / _sr * 1000
    minutes, seconds = divmod(duration_ms / 1000, 60)

    print(f"Waveform loaded successfully: {int(minutes)}m:{seconds:.2f}s source audio")
    return _samples, _sr

def analyze_audio_beats_and_bpm(audio, _sr, min_bpm=None, max_bpm=None):
    """
    Analyze audio data for beats, downbeats, tempo, and musical notes.
    min_bpm and max_bpm None mean use defaults of 55 and 170
    """
    print("Analyzing audio data...")
    beat_data = {
        'sample_rate': _sr,
        'duration': librosa.get_duration(y=audio, sr=_sr),
    }
    if min_bpm is None:
        min_bpm = 55
    if max_bpm is None:
        max_bpm = 170

    # Processing beats and downbeats using the custom downbeat processor
    print(f"Executing RNNDownBeatProcessor [slow, about 6 sec / min source audio on MacM1max]...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    downbeat_processor = RNNDownBeatProcessor()
    activations = downbeat_processor(audio)
    print(f"Done RNNDownBeatProcessor ...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    dbn_processor = myDBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100, min_bpm=min_bpm, max_bpm=max_bpm)
    beats_and_downbeats = dbn_processor(activations, quiet=False)

    beat_times = beats_and_downbeats[:, 0]
    beat_number_in_bar = beats_and_downbeats[:, 1].astype(int)  # the beat # 1-3 or 1-4

    # Data aggregation for output
    beat_data['number_of_beats'] = len(beat_times)
    beat_data['beat_times'] = beat_times.tolist()
    beat_data['beat_numbers'] = beat_number_in_bar.tolist()

    # Extracting downbeats (first beat of each bar)
    downbeats = beats_and_downbeats[beat_number_in_bar == 1]  # the beat # 1-3 or 1-4
    beat_data['downbeats'] = list(zip(downbeats[:, 0], ['Beat 1'] * len(downbeats)))  # beat 1 represents downbeat

    # Compute the tempo if enough beats are detected
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)  # Calculate the differences between consecutive beat times
        tempo = 60 / np.median(intervals) if intervals.size > 0 else 0  # Median interval converted to BPM
    else:
        tempo = 0  # Default to zero if not enough beats to calculate intervals
    beat_data['tempo_float'] = tempo

    # time_signature 3/4 or 4/4 for this audio segment
    largest_beat_number = max(beat_number_in_bar) # max of beat+number 1-3 or 1-4
    beat_data['time_signature'] = f"{largest_beat_number}/4"

    # extract notes/chroma
    beat_data['chroma'] = extract_chroma_features(audio, _sr, beat_times, beat_number_in_bar)

    print("Completed audio analysis...")
    return beat_data

def extract_chroma_features(audio, _sr, beat_times, beat_positions_in_bar):
    """
    Extract chroma features from the audio and map beat times to the nearest chroma frame
    """
    print("Extracting spectral features... ")
    print("List the notes / intensity at each beat ordered by their chroma intensity from highest to lowest")
    chroma = librosa.feature.chroma_cqt(y=audio, sr=_sr, hop_length=512)

    frame_times = librosa.frames_to_time(range(chroma.shape[1]), sr=_sr, hop_length=512)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Map beat times to the nearest chroma frame and retrieve corresponding notes
    beat_frames = np.searchsorted(frame_times, beat_times, side='left')
    beat_chroma = [
        sorted(
            [(note_names[i], chroma[i, frame]) for i in np.where(chroma[:, frame] >= MIN_CHROMA)[0]],
            key=lambda x: x[1],
            reverse=True
        )
        if frame < chroma.shape[1] else [('N/A', 0)]
        for frame in beat_frames
    ]

    return beat_chroma

def print_data(_beat_data, _filename, quiet=True):
    """
    Print the results of the audio analysis.
    quiet True means don't print all detected beats and notes [chroma]
    """
    print(f"\nRESULTS OF ANALYSIS for '{_filename}'\n===================================================================================================")

    # Printing detailed information about beats and downbeats
    if not quiet:
        print(f"All Detected Beats (index, beat #, time, and note with chroma [for chroma {MIN_CHROMA} .. 1.0]):\n")

        for idx, (time, beat_num, notes) in enumerate(zip(_beat_data['beat_times'], _beat_data['beat_numbers'], _beat_data['chroma'])):
            minutes, seconds = divmod(time, 60)
            time_str = f"{int(minutes)}m:{seconds:.2f}s"
            note_list = [f"{note}" for note, chroma in notes]
            chroma_list = [f"{chroma:.2f}" for note, chroma in notes]
            notes_str = ', '.join(note_list) + " [" + ', '.join(chroma_list) + "]"
            print(f"Index: {idx}, Beat #: {beat_num}, Time: {time_str}, Notes: {notes_str}")

    # Printing basic audio and analysis metrics
    print("\nGeneral Analysis Results:")
    print(f"Sample Rate: {_beat_data['sample_rate']} Hz")
    minutes, seconds = divmod(_beat_data['duration'], 60)
    time_str = f"{int(minutes)}m:{seconds:.2f}s"
    print(f"Total Duration: {time_str}")
    print(f"Total Number of Beats: {_beat_data['number_of_beats']}")
    # Printing tempo calculated from beat intervals
    print(f"Calculated Tempo: {_beat_data['tempo_float']:.1f} BPM")
    print(f"Time signature: {_beat_data['time_signature']}")  # time_signature 3/4 or 4/4 (string)

if __name__ == '__main__':
    ROOT_DIR = "/my/path/"
    filename = "downbeat tracker test.m4a"
    filePath = os.path.join(ROOT_DIR, filename)
    samples, sr = get_waveform(filePath)
    # should be 44.1kHz
    MIN_CHROMA = 0.4  # default is 0.5
    data = analyze_audio_beats_and_bpm(samples, sr, min_bpm=None, max_bpm=None)
    print_data(data, filename, quiet = False)
    print("Analysis complete.")
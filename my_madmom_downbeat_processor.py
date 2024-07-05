"""
Demo for use of madmom with Python 3.11

There are several minor but important issues with Python 3.11 that are fixed in this code
1- alias
2- subclass to avoid strange error possibly due to np handling between python and support library versions
   ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape
   after 2 dimensions. The detected shape was (2, 2) + inhomogeneous part.

(c) MGazier, 2024
MIT License

Example usage with this code
  Main code does before using this code to get beat activations
    downbeat_processor = RNNDownBeatProcessor()
    activations = downbeat_processor(y)
  This code is then called as (this code's filename is 'my_madmom_downbeat_processor') to get beat#/downbeats
    from my_madmom_downbeat_processor import myDBNDownBeatTrackingProcessor
    proc = CustomDBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    or
    proc = myDBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100, min_bpm=min_bpm, max_bpm=max_bpm)

beats_and_downbeats = proc(activations)
"""

import numpy as np
from datetime import datetime

# fix (patch) Python 3.11 issues in madmom
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence  # Alias MutableSequence to avoid deprecation issues
np.float = float
np.int = int
# import madmom class to subclass from
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
# from madmom.ml.hmm import HiddenMarkovModel  # only used in __init__ that we are inheriting

class myDBNDownBeatTrackingProcessor(DBNDownBeatTrackingProcessor):
    """
    Custom processor subclassing the original DBNDownBeatTrackingProcessor
    to allow Python 3.11 operation.
    """

    @staticmethod
    def _process_dbn(process_tuple):
        """
        Extract the best path through the state space in an observation sequence.
        This proxy function is necessary to process different sequences in parallel
        using the multiprocessing module.
        Copied original _process_dbn() here as we are subclassing and want to keep simple.
        """
        return process_tuple[0].viterbi(process_tuple[1])

    def process(self, activations, quiet=True, **kwargs):
        """
        Detect the (down-)beats in the given activation function.

            Process the activations to extract the best path through the state space.
            This involves thresholding, decoding with HMMs, selecting the best HMM,
            and correcting beats if specified.

            Call this DBNDownBeatTrackingProcessor with the beat activation function
            returned by RNNDownBeatProcessor to obtain the beat positions.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        print(f"Processing activations with my DBNDownBeatTrackingProcessor... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        import itertools as it # done here to be similar to original DBNDownBeatTrackingProcessor
        if not quiet:
            print("myDBNDownBeatTrackingProcessor(DBNDownBeatTrackingProcessor) processing of activations")
        first = 0
        if self.threshold:
            # this was def threshold_activations(activations, threshold):
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = np.min(idx)
                last = np.max(idx) + 1
                activations = activations[first:last]
            else:
                print("  No activations above threshold.")
                return np.empty((0, 2))

        if not activations.any():
            print("  No activations to process after thresholding.")
            return np.empty((0, 2))

        # Decoding the activations with the HMMs
        if not quiet:
            print("myDBNDownBeatTrackingProcessor(DBNDownBeatTrackingProcessor) Decoding the activations with HMMs")
        results = list(self.map(self._process_dbn, zip(self.hmms, it.repeat(activations))))

        # Choose the best HMM (highest log probability)
        log_probs = [result[1] for result in results]
        best = np.argmax(log_probs)
        path, _ = results[best]

        # Retrieve the best model's state space and observation model
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model

        # Get positions and calculate beat numbers
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1  # Natural counting

        # Correct beats to the nearest activation peak
        if not quiet:
            print("myDBNDownBeatTrackingProcessor(DBNDownBeatTrackingProcessor) Correct beats")
        if self.correct:
            beats = self._correct_beats(activations, path, om)
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1  # Where beat numbers change

        # Convert beat positions to seconds
        beat_times = (beats + first) / float(self.fps)
        beat_info = np.vstack((beat_times, beat_numbers[beats])).T

        if not quiet:
            print(f"myDBNDownBeatTrackingProcessor(DBNDownBeatTrackingProcessor) beat_times and beat_numbers complete... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return beat_info

    def _correct_beats(self, activations, path, observation_model):
        """
        Correct beat positions to nearest activation peak.
        """
        beats = np.empty(0, dtype=int)
        # for each detection determine the "beat range", i.e. states where
        # the pointers of the observation model are >= 1
        beat_range = observation_model.pointers[path] >= 1
        # if there aren't any in the beat range, there are no beats
        if not beat_range.any():
            return np.empty((0, 2))
        # get all change points between True and False (cast to int before)
        changes = np.diff(beat_range.astype(int))
        idx = np.nonzero(changes)[0] + 1
        # if the first frame is in the beat range, add a change at frame 0
        if beat_range[0]:
            idx = np.r_[0, idx]
        # if the last frame is in the beat range, append the length of the
        # array
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        # iterate over all regions
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                # pick the frame with the highest activations value
                # Note: we look for both beats and down-beat activations;
                # since np.argmax works on the flattened array, we
                # need to divide by 2
                peak = np.argmax(activations[left:right]) // 2 + left
                beats = np.hstack((beats, peak))
        return beats

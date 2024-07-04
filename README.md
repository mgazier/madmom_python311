# madmom_python311

# Rough demo / example for Madmom use with Python 3.11
- loads audio from file (should be 44.1kHz for Madmom - i didn't include rate resampling via librosa.resample)
- use Madmom to determine beats per measure, and BPM
   1- RNNDownBeatProcessor()
   2- DBNDownBeatTrackingProcessor()
- i also played a bit with librosa's chroma_cqt() for note detection - not a primary goal just to explore including time synch
- it appears signficantly more precise than librosa tempo estimation although it's much slower - addressing gpu support would be ideal

# Why
- There are current compatibility issues between Python 3.11 and Madmom and numpy
- I had trouble finding clean code examples for Madmom use (and have seen other people say the same). Other code didn't work given the compatibility issues or was too handcrafted / incomplete.

# So ..
- i put together a rough short program to test out and & debug my solution with Madmom
	- i added some minimal non intrusive patches (subclass, aliases,..)
- it appears to work well, although i haven't fully tested it yet
- i'll be integrating & testing this in my actual project now that it works

This code is raw, lightly tested and meant to share ideas, it is far from a product.

Sample output is simple: 
![minidemo1](https://github.com/mgazier/madmom_python311/madmom_python311_1.jpeg)
![minidemo2](https://github.com/mgazier/madmom_python311/madmom_python311_2.jpeg)


(c) Michael Gazier, 2023-2024 
MIT License

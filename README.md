# madmom_python311

# Rough demo / example for Madmom use with Python 3.11
- loads audio from file (should be 44.1kHz for Madmom - i didn't include rate resampling via librosa.resample)
- use Madmom to determine time signature (3/4 or 4/4), beats per minute (BPM), and the beat # in the measure including downbeat
   1- RNNDownBeatProcessor()
   2- DBNDownBeatTrackingProcessor()
- i also played a bit with librosa's chroma_cqt() for note detection - not a need, just to explore including time synch
- Madmom appears signficantly more precise than librosa tempo estimation although it's much slower - addressing gpu support would be ideal

# Why
- There are compatibility issues between Python 3.11 and Madmom and numpy (Jun 2024)
- I had trouble finding working & clean code examples for Madmom use (and have seen other people say the same).

# So ..
- i put together a rough short program to test out and & debug my solution with Madmom
	- i added some minimal non intrusive patches (subclass, aliases,..) as i didn't want to change the Madmom source etc.
- it appears to work well, although i haven't fully tested it yet

This code is raw, lightly tested and meant to share ideas, it is far from a product.

Output is simple and its general analysis is at the end

![minidemo1](https://github.com/mgazier/madmom_python311/blob/main/madmom_python311_1.jpeg)
[...]
![minidemo2](https://github.com/mgazier/madmom_python311/blob/main/madmom_python311_2.jpeg)


(c) Michael Gazier, 2023-2024 
MIT License for my code
Given parts of this code are essentially Madmom code, you should comply/check with the author Dr. Gerhard Widmer
https://github.com/CPJKU/madmom
Current web site https://www.jku.at/en/institute-of-computational-perception/about-us/people/gerhard-widmer/

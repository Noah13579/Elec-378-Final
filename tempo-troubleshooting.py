"""
This file exists to troubleshoot issues with the tempo feature of librosa.

If you get the error
AttributeError: module 'scipy.signal' has no attribute 'hann'
when attempting to run this file, you should modify librosa's beat.py file 
(located at .venv/lib/python3.12/site-packages/librosa/beat.py) as follows:

line 507: change from 
        smooth_boe = scipy.signal.convolve(localscore[beats], scipy.signal.hann(5), "same")
to
        smooth_boe = scipy.signal.convolve(localscore[beats], scipy.signal.windows.hann(5), "same")
(add "windows" to scipy path bc it has trouble finding "hann" without this)
"""

import librosa

y, sr = librosa.load(librosa.ex('choice'), duration=10)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print(tempo)

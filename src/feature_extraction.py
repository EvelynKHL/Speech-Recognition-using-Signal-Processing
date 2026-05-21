import wave
import numpy as np
from python_speech_features import mfcc

SAMPLE_RATE = 16000


def extract_features(file_path):
    with wave.open(file_path, 'r') as wf:
        n_frames = wf.getnframes()
        audio_frames = wf.readframes(n_frames)

    waveform = np.frombuffer(audio_frames, dtype=np.int16)
    features = mfcc(waveform, SAMPLE_RATE)

    return features

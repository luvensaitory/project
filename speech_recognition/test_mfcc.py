import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
 
y, sr = librosa.load("ctc/audio.wav")
mfccs1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
fs, au = wav.read("ctc/audio.wav")
mfccs2 = mfcc(au, samplerate=fs)
print("sr=", sr, "\tfs=", fs)
print("y=", y, "\tau=", au)
print("mfcc1:\n", mfccs1.shape)
print("mfcc2:\n", mfccs2.shape)
'''
feature_inputs = np.asarray(mfccs[np.newaxis, :])
print("new axis:\n", feature_inputs.shape)

feature_inputs = (feature_inputs - np.mean(feature_inputs))/np.std(feature_inputs)
print("norm:\n", feature_inputs.shape)

fl = [feature_inputs.shape[1]]
print("le n= ", fl)
''''''
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
'''


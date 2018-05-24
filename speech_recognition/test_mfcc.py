import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("audio/mp3001/001_001.mp3")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
print("mfcc:\n", mfccs.shape)

feature_inputs = np.asarray(mfccs[np.newaxis, :])
print("new axis:\n", feature_inputs.shape)

feature_inputs = (feature_inputs - np.mean(feature_inputs))/np.std(feature_inputs)
print("norm:\n", feature_inputs.shape)

fl = [feature_inputs.shape[1]]
print("le n= ", fl)
'''
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
'''

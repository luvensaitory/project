import librosa
import librosa.display
import numpy as np

fp = open('people.txt', 'r')
num = (int)(fp.readline())
error = 0
for j in range(num):
    if j<9:
        people = "00" + str(j+1)
    elif j>=9 and j<99:
        people = "0" + str(j+1)
    else:
        people = str(j)
    path = "audio/wav" + people + "/"

    for i in range(1, 101):
        if i<10:
            data = path + people + "_00" + str(i)
        elif i>=10 and i<100:
            data = path + people + "_0" + str(i)
        else:
            data = path + people + "_" + str(i)
        fname = data + ".wav"
        y, sr = librosa.load(fname)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        if np.array(mfccs).shape[1]<120:
            print(data, " wrong ", np.array(mfccs).shape)

if error>0:
    print("Finish... ", error, "error.")
else:
    print("Finish... All audio are fine.")
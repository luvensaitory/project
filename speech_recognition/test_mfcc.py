import librosa
import librosa.display
import numpy as np
import pandas as pd

bad_data = pd.read_csv("bad.csv", header=None)
bad_data = bad_data.values.reshape((bad_data.shape[1],))

fp = open('people.txt', 'r')
num = (int)(fp.readline())

for j in range(num*5):
    error = 0
    if j<9:
        people = "00" + str(j+1)
    elif j>=9 and j<99:
        people = "0" + str(j+1)
    else:
        people = str(j)
    path = "audio/fixwav" + people + "/"

    for i in range(1, 101):
        if i in bad_data:
            continue
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
            sec = librosa.get_duration(y=y, sr=sr)
            print(data, " wrong ", np.array(mfccs).shape, "duration: ", sec)

        if error>0:
            print(people, " finish... ", error, "error.")
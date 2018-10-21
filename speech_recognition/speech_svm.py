
from __future__ import print_function
import numpy as np

# for file operations
import os
import librosa
import pandas as pd
import sys
sys.path.insert(0, "/home/speech/deeplearning/lib/python3.5/site-packages")
from sklearn import svm
np.random.seed(1432)
from random import randint
import gc
from tqdm import tqdm

bad_data = pd.read_csv("bad.csv", header=None)
bad_data = bad_data.values.reshape((bad_data.shape[1],))
classes = 90

def load_audio(start_person, end_person, ignore_num=None):
    
    label = [ num for num in range(0, classes)]
 
    audio_path = "audio/"

    audio_array = []
    class_array = []

    for person in tqdm(range(start_person, end_person+1), ncols=77):
        #person_path, ex: /wav001/
        #people     , ex: 001
        if ignore_num != None:
            if person in ignore_num:
                continue
        if person<10:
            person_path = audio_path + "/fixwav00" + str(person) + "/"
            people = "00" + str(person)
        elif person>=10 and person<100:
            person_path = audio_path + "/fixwav0" + str(person) + "/"
            people = "0" + str(person)
        else:
            person_path = audio_path + "/fixwav" + str(person) + "/"
            people = str(person)
        
        count = 0
        for person_audio in tqdm(range(1, 101), ncols=77):
            #audio_name, ex: 001_001.wav
            if person_audio in bad_data:
                pass
            else:
                if person_audio<10:
                    audio_name = person_path + people + "_00" + str(person_audio) + ".wav"
                elif person_audio>=10 and person_audio<100:
                    audio_name = person_path + people + "_0" + str(person_audio) + ".wav"
                else:
                    audio_name = person_path + people + "_" + str(person_audio) + ".wav"
                
                y, sr = librosa.load(audio_name)
                mfccs_feature = librosa.feature.mfcc(y=y, sr=sr)
                mfccs = np.array(mfccs_feature[:, 0:120])
                mfccs = (mfccs - np.mean(mfccs))/np.std(mfccs)
                audio_array.append(mfccs)
                class_array.append(count)
                count = count + 1

    # label_sparse = np.zeros((len(class_array), len(label)))
    # label_sparse[np.arange(len(class_array)), class_array] = 1
    #print("data_array: ", np.array(audio_array).shape)

    return np.array(audio_array).reshape((-1, 2400)), class_array

#label
label = [ num for num in range(0, classes)]

#load data
train_person = 14
ignore_list = [ train_person + x * 14 for x in range(5) ]
X_train, y_train = load_audio(1, 13, ignore_list)


clf = svm.SVC()
#training
clf.fit(X_train, y_train)

X_train = None
y_train = None

#testing
print("\nTesting......\n")
X_test, y_test = load_audio(14, 14)

preds = clf.predict(X_test)

accuracy = 0.0
for i in range(0, len(preds)):
    if preds[i] == y_test[i]:
        accuracy += 1

print("Validation accuracy: ", 100*accuracy/len(preds), " %")
#print("Confusion matrix:")
#print(label)
#print(confusion_matrix)

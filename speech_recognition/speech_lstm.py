
from __future__ import print_function
import numpy as np

# for file operations
import os
import librosa
import pandas as pd
import sys
sys.path.insert(0, "/home/speech/deeplearning/lib/python3.5/site-packages")

import keras.backend as K
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU # you can also try using GRU layers
from keras.optimizers import RMSprop, Adadelta, adam, sgd # you can try all these optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from random import randint
import gc

np.random.seed(1432)

bad_data = pd.read_csv("bad.csv", header=None)
bad_data = bad_data.values.reshape((bad_data.shape[1],))
classes = 90

def load_audio(start_person, end_person, ignore_num=None):
    
    label = [ num for num in range(0, classes)]
 
    audio_path = "audio/"

    audio_array = []
    class_array = []

    for person in range(start_person, end_person+1):
        #person_path, ex: /wav001/
        #people     , ex: 001
        if ignore_num != None:
            if person == ignore_num:
                continue
        if person<10:
            person_path = audio_path + "/wav00" + str(person) + "/"
            people = "00" + str(person)
        elif person>=10 and person<100:
            person_path = audio_path + "/wav0" + str(person) + "/"
            people = "0" + str(person)
        else:
            person_path = audio_path + "/wav" + str(person) + "/"
            people = str(person)
        
        count = 0
        for person_audio in range(1, 101):
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

    label_sparse = np.zeros((len(class_array), len(label)))
    label_sparse[np.arange(len(class_array)), class_array] = 1
    #print("data_array: ", np.array(audio_array).shape)

    return np.array(audio_array), label_sparse

#label
label = [ num for num in range(0, classes)]

#build model......

model = Sequential()

model.add(LSTM(units=80, activation='tanh', input_shape=(20, 120)))
model.add(Dropout(.1))
model.add(Dense(45, activation='tanh'))
model.add(Dense(90, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

batch_size = 50
epochs = 300

#load data
X_train, y_train = load_audio(1, 13, 14)
#training
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, shuffle=True, verbose=1)

X_train = None
y_train = None

#testing
print("\nTesting......\n")
X_test, y_test = load_audio(14, 14)

preds = model.predict(X_test)

confusion_matrix = np.zeros(shape=(y_test.shape[0], y_test.shape[1]))
accuracy = 0.0
for i in range(0, len(preds)):
    confusion_matrix[np.argmax(preds[i])][np.argmax(np.array(y_test[i]))] += 1
    
    #print("Pred:", np.argmax(preds[i], ", actual:", np.argmax(np.array(y_test[i]))))

    if np.argmax(preds[i]) == np.argmax(np.array(y_test[i])):
        accuracy += 1

print("Validation accuracy: ", 100*accuracy/len(preds), " %")
#print("Confusion matrix:")
#print(label)
#print(confusion_matrix)

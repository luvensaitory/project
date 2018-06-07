import pandas as pd
import numpy as np
import os

bad_data = pd.read_csv("bad.csv", header=None)
bad_data = bad_data.values.reshape((bad_data.shape[1],))


path = "audio/wav"
fp = open('people.txt', 'r')
person_number = (int)(fp.readline())
class_size = 100

for i in range(1, person_number+1):
    if i<10:
        personname = "00" + str(i)
    elif i>=10 and i<100:
        personname = "0" + str(i)
    else:
        personname = str(i)
    
    #set to mp3 file directory 
    path2person = path + personname + "/"

    for j in range(1, class_size+1):
        if j<10:
            fname = personname + "_00" + str(j) + ".wav"
        elif j>=10 and j<100:
            fname = personname + "_0" + str(j) + ".wav"
        else:
            fname = personname + "_" + str(j) + ".wav"
        if j in bad_data:
            rm = "rm " + path2person + fname
            #os.system(rm)
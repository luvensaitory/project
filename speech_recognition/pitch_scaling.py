import os
import librosa
import sys
sys.path.insert(0, "/home/speech/deeplearning/lib/python3.5/site-packages")

path = "audio/"
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
    
    #set to m4a file directory 
    path2person = path + personname + "/"
    for k in range(1, 5):
        pitch_name = i + person_number * k
        pitch_name = "0" + str(pitch_name)
        #make new directory for storing files that be modified using ffmpeg
        newdir = path + "fixwav" + pitch_name
        mkdir = "mkdir " + newdir
        os.system(mkdir)

        pitch = 0.7 + k * 0.1
        if pitch == 0.1:
            pitch += 0.1
        for j in range(1, class_size+1):
            if j<10:
                fname = pitch_name + "_00" + str(j) 
                output = fname + ".wav"
                filename = personname + "_00" + str(j) + ".m4a"
            elif j>=10 and j<100:
                fname = pitch_name + "_0" + str(j)
                output = fname + ".wav"
                filename = personname + "_0" + str(j) + ".m4a"
            else:
                fname = pitch_name + "_" + str(j)
                output = fname + ".wav"
                filename = personname+ "_" + str(j)  + ".m4a"
            
            #pitch scaling
            ffmpeg = "ffmpeg -i " + path2person + filename + " -filter:a \"asetrate=r=" + str( int( 44 * pitch ) ) + "K\" -vn " + newdir + "/" + output
            os.system(ffmpeg)
            
            #fixing length
            new_file = newdir + "/" + output
            y, sr = librosa.load(new_file)
            sec = librosa.get_duration(y=y, sr=sr)
            atempo = sec/3
            if atempo>=0.5 and atempo<=2:
                ffmpeg = "ffmpeg -i " + new_file + " -filter:a \"atempo=" + str(atempo) + "\" -y " + new_file
            elif atempo>2:
                ffmpeg = "ffmpeg -i " + new_file + " -filter:a \"atempo=2.0, atempo=" + str(atempo/2.0) + "\" -y " + new_file
            else: 
                ffmpeg = "ffmpeg -i " + new_file + " -filter:a \"atempo=0.5, atempo=" + str(atempo/0.5) + "\" -y " + new_file
            os.system(ffmpeg)
            
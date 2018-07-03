import os
import librosa

try:
    while True:
        person = input("Person: ")
        data = input("Data: ")
        person = int(person)
        data = int(data)
        count = 0
        path = "audio/fixwav"
        if person < 10:
            file_dir = path + "00" + str(person) + "/00" + str(person) 
        elif person < 100 and person >= 10:
            file_dir = path + "0" + str(person) + "/0" + str(person) 
        if data < 10:
            file_name = file_dir + "_00" + str(data) + ".wav"
        elif data < 100 and data >= 10:
            file_name = file_dir + "_0" + str(data) + ".wav"
        else:
            file_name = file_dir + "_" + str(data) + ".wav"

        while person > 14:
            person -= 14
            count += 1
        pitch = 0.7 + 0.1 * count
        if pitch == 1.0:
            pitch += 0.1
        
        y, sr = librosa.load(file_name)
        sec = librosa.get_duration(y=y, sr=sr)
        atempo = sec/3
        
        if atempo>=0.5 and atempo<=2:
            ffmpeg = "ffmpeg -i " + file_name + " -filter:a \"atempo=" + str(atempo) + "\" -y " + file_name
        elif atempo>2:
            ffmpeg = "ffmpeg -i " + file_name + " -filter:a \"atempo=2.0, atempo=" + str(atempo/2.0) + "\" -y " + file_name
        else: 
            ffmpeg = "ffmpeg -i " + file_name + " -filter:a \"atempo=0.5, atempo=" + str(atempo/0.5) + "\" -y " + file_name
        
        os.system(ffmpeg)

        y, sr = librosa.load(file_name)
        sec2 = librosa.get_duration(y=y, sr=sr)
        print("Before:\t", sec, "\nAfter:\t", sec2)
except EOFError:
    print("\n")
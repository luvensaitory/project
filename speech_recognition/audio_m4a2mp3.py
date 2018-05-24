import os

path = "audio/"
person_number = 8
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
    
    #make new directory for storing files that be modified using ffmpeg
    newdir = path + "mp3" + personname 
    mkdir = "mkdir " + newdir
    os.system(mkdir)

    for j in range(1, class_size+1):
        if j<10:
            filename = personname + "_00" + str(j)
        elif j>=10 and j<100:
            filename = personname + "_0" + str(j)
        else:
            filename = personname + "_" + str(j)
        ffmpeg = "ffmpeg -i " + path2person + filename + ".m4a " + "-acodec libmp3lame -ab 256k " + newdir + "/" + filename + ".mp3"
        os.system(ffmpeg)

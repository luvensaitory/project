# LSTM-KTH
A recurrent deep neural network for human activity recognition, using the KTH dataset as an example.
Reference from https://github.com/inubushi/LSTM-KTH .

## prerequisites
This code contains OS commands for linux. It also uses keras with tensorflow, and ffmpeg. I have not tested the code on Python 3.

## Instructions
After downloading the KTH dataset:

1. Edit the path to the dataset in both KTH_prepare.py and KTH_LSTM.py
2. Run KTH_prepare.py to extract frames for the relevant segments of the dataset.
3. Try running KTH_LSTM.py to train the network. You might want to change the parameters like the amount of data used for training and testing.
4. Modify the network architecture and hyper parameters to improve accuracy.

##bugs 
1. In KTM_prepare.py, line 94, rewrite the condition, delete "and j>20".
2. Is a very deadly bug. It costs me 4 hours to find out. The array for LSTM is 2400 x 25 x 120 x 160 x 1, while 1 stands for 1 channel, 120 x 160 for image size, 25 for each sequence, 2400 for ( 6 classes x 25 people x 4 records x 4 segments ). But in the number 547 of the array(547 x 25 x 120 x 160 x 1)has a problem. The default sequence of images is 25, 547 which means person10 handclapping d1 seg4 has only 24 images. This causes the numpy can't fit data_array in KTH_LSTM.py to (2400,25,120,160,1) instead (2400,). So, trivial solution is to modified the LSTM_sequence.txt. In line 452, change "201-260, 261-284" to "201-259, 259-284".

The Step of Total Works:
1.Data preprocessing
	***origin data contains 14 people's audio about 100 Taiwanese vocabularies.***
	1-1.Data Augmentation
		Use "pitch_scaling.py" to generate audio data through pitch{ 0.8, 0.9, 1.1, 1.2 }. Then we can have 5 times as origin data.
	1-2.Data Fixed Size
		Use "audio_m4a2fixwav.py" to fixs each audio file to 3 sec.

2.Model Training
	***use LSTM model.***
	2-1.Training
		Use "speech_lstm.py" to train lstm model.

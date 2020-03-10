from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
from scipy.io import wavfile
import os
from nltk import tokenize
import librosa
#import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import load
from torchaudio.transforms import MFCC

data_dir = '/home/val/VoiceDataset'

sound_len = 0

class VoiceDataset(Dataset):

	def __init__(self, frame, sound_len):

		self.frame = frame
		self.sound_len = sound_len

	def __len__(self):
		return self.frame.shape[0]

	def __getitem__(self, item):
		if torch.is_tensor(item):
			return item


class VoiceModel(nn.Module):

	def __init__(self):

		super().__init__()
		self.input = nn.GRU(input_size=sound_len, bidirectional=True)
		self.gru = nn.GRU(bidirectional=False)
		self.dense = nn.Linear(512, 256)
		self.dropout = nn.Dropout(0.3)
		self.output = nn.Linear(256, 128)

	def forward(x):
		pass


def get_mfcc(arr):

	mfcc_tensor = MFCC(sample_rate=22000)
	return mfcc_tensor.forward(arr)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

audio_dir = 'clips'

audio_data = pd.read_csv('train.tsv', sep='\t')	# .loc[25:75, :]
print(audio_data.columns)
sentences = audio_data.loc[:, 'sentence']
print(sentences.tolist()[1])

batch_size = 16

lens = []
for i in audio_data.index:
	file = os.path.join(data_dir, audio_dir, audio_data.loc[i, 'path'])
	waveform, sample_rate = load(file)
	#print(waveform, sample_rate)
	
	speech = AudioSegment.from_mp3(file)
	#print(speech.channels, 'channels')
	#lens.append(len(speech))
	#speech_array = np.array(speech.get_array_of_samples())
	audio_data.loc[i, 'len'] = len(waveform[0])
	#audio_data.loc[i, 'max_amplitude'] = max(waveform[0].tolist())
	#audio_data.loc[i, 'min_amplitude'] = abs(min(waveform[0].tolist()))
	#audio_data.loc[i, 'mfcc'] = get_mfcc(waveform[0])
	#print(get_mfcc(waveform[0]).shape)
	#print(max(waveform[0].tolist()))
	#print(len(speech_array))
	#plt.plot(range(len(speech_array)), speech_array)
	#plt.show()
	#break

print(audio_data.describe())


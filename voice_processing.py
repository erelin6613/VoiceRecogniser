from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
from scipy.io import wavfile
import os
from nltk import tokenize
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchaudio import load
from torchaudio.transforms import MFCC
#from transformers import BertTokenizer, BertModel

data_dir = '../VoiceDataset'
audio_dir = 'clips'

sound_len = 0


class VoiceInstance:

	def __init__(self, file, text, sound_len=10**10):

		self.file = file
		# self.rate = rate
		self.text = text
		self.mfcc = _transform_audio(self.file)
		self.emb = _get_embeddings(self.text)

	def _transform_audio(self):

		waveform, rate = load(os.path.join(self.file))
		self.audio = self._get_mfcc(waveform, rate)

	def _get_mfcc(self, arr, sample_rate=22000):

		mfcc_tensor = MFCC(sample_rate)
		return mfcc_tensor.forward(arr)

	def _get_embeddings(self):

		def remove_punkts(self):

			punkts = ['.', '?', '!', '\'', '"', '’', '‘', ',', ';']
			for punkt in punkts:
				self.text = str(self.text).replace(punkt, '')


		def tokenize_text(self, text):

			tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			return tokenizer.tokenize(text)

		def get_inputs(self, tokens):
			return tokenizer.convert_tokens_to_ids(tokens)

		def get_bert(self):
			return BertModel.from_pretrained('bert-base-uncased')

		remove_punkts()
		self.emb = self.tokenize_text(self.text)


class VoiceDataset(Dataset):

	def __init__(self, info_frame):

		self.info_frame = info_frame
		self.load_data()

	def __len__(self):
		return len(self.audio)

	def __getitem__(self, item):
		return item

	def load_data(self):

		self.instances = []

		for i in self.info_frame.index:
			audio = VoiceInstance(file=os.path.join(self.audio_dir, self.info_frame.loc[i, 'path']),
									text=self.info_frame.loc[i, 'sentence'])
			self.instances.append(audio)



class VoiceModel(nn.Module):

	def __init__(self):

		super().__init__()
		self.input = nn.GRU(input_size=sound_len, bidirectional=True, hidden_size=512)
		self.gru = nn.GRU(input_size=512, bidirectional=False, hidden_size=512)
		self.dense = nn.Linear(512, 256)
		self.dropout = nn.Dropout(0.3)
		self.output = nn.Linear(256, 128)


	def forward(audio):
		audio = F.relu(self.input(audio))
		audio = F.relu(self.gru(audio))
		audio = F.relu(self.dense(audio))
		audio = F.relu(self.dropout(audio))
		audio = self.output(audio)

		return x

def main():

	batch_size = 16
	epochs = 1
	model = VoiceModel()
	criterion = nn.NLLLoss()
	optimizer = Adam(model.parameters(), lr=0.001)

	audio_data = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t')	# .loc[25:75, :]
	print(audio_data.columns)


	dataset = VoiceDataset(info_frame=audio_data)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	for epoch in range(epochs):
		running_loss = 0
		for batch in dataset:

			optimizer.zero_grad()
			output = model(batch)
			loss = criterion(batch[0], batch[1])
			loss.backwards()
			optimizer.step()
			running_loss += loss.item()
			print(running_loss)

if __name__ == '__main__':
	main()

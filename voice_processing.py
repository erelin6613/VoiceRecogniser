#!/usr/bin/python3

"""
Author: Valentyna Fihurska
https://github.com/erelin6613

****** VoiceRecogniser ******

The main purpose of this project is to create
a new voice recognition model which:

a) is free and open sourced
b) 'makes sense' out of the output it 'hears'

In my experience even paid voice recognition 
systems mispredicted words making non-sensical
predictions (f.e. '... I cant reach' being
predicted as '... I camp rich').

My attempt is to combine signal processing techniques
and recent advances in natural language processing
to create a better system.

As I mentioned I want it to be open sourced and
the data I use to train it is data collected by 
Common Voice project, find more details here:
https://voice.mozilla.org

****** Current state of development ******

For now the program runs smoothly but works not
as intendent. The main issues have to do with dimensions
which I am working on to solve.

****** Contributing ******

I will be happy to know my code was useful to
anybody and if you want to help with development
I would be grateful for any suggestion or help.

"""

import pandas as pd
from scipy.io import wavfile
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, tanh, flatten, squeeze, mean, stft
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torchaudio import load
from torchaudio.transforms import MFCC, Resample
from torchaudio.functional import istft
import numpy as np

data_dir = '../common-voice'
audio_dir = 'cv-valid-train'
sound_len = 10
bert_path = '../bert-base-uncased'
bert_vocab_path = 'vocab.txt'
bert_model_path = 'bert-base-uncased/pytorch_model.bin'
max_len_text = 30
window = 40

class VoiceInstance:
    
    def __init__(self, file, tokenizer, sound_len=10**10, text=None):
        
        self.file = file
        self.text = text
        self.tokenizer = tokenizer
        self._transform_audio()
        self.emb = self._get_embeddings()
        
    @staticmethod
    def pad_seq(sequence):
        return Tensor(np.pad(sequence, 0))
    
    def _transform_audio(self):
        
        waveform, rate = load(os.path.join(data_dir, self.file))
        new_rate = rate/100
        resampled = Resample(rate, new_rate)(waveform)
        self.stft = self._get_stft(resampled, new_rate)
        self.mfcc = self._get_mfcc(resampled, new_rate)
        
    def _get_mfcc(self, arr, sample_rate=22000):
        
        mfcc_tensor = MFCC(sample_rate, n_mfcc=window)
        return mfcc_tensor.forward(arr)
    
    def _get_stft(self, waveform, rate):
        return stft(waveform, int(rate))
    
    def _get_embeddings(self):
        
        def remove_punkts(text):
            
            punkts = ['.', '?', '!', '\'', '"', '’', '‘', ',', ';']
            for punkt in punkts:
                text = str(self.text).replace(punkt, '')
            return text
        
        def tokenize_text(text):
            return self.tokenizer.tokenize(text)
        
        def get_inputs(tokens):
            return self.tokenizer.convert_tokens_to_ids(tokens)
        
        if self.text:
            self.text = remove_punkts(self.text)
            self.tokens = tokenize_text(self.text)
            emb = get_inputs(self.tokens)
            return emb


class VoiceDataset(Dataset):
    
    def __init__(self, info_frame, audio_dir, tokenizer):
        
        self.info_frame = info_frame
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.load_data()
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, item):
        return item
    
    def load_data(self):
        
        self.instances = []
        for i in self.info_frame.index:
            if self.info_frame.loc[i, 'text'] is not None:
                audio = VoiceInstance(file=os.path.join(self.audio_dir, self.info_frame.loc[i, 'filename']),
                                      text=self.info_frame.loc[i, 'text'], tokenizer=self.tokenizer)
                self.instances.append(audio)
        embs = []
        mfccs = []
        for each in self.instances:
            embs.append(each.emb)
            mfccs.append(each.mfcc)
        embs = pad_sequences(embs, maxlen=max_len_text, padding='pre', value=0)
        for i in range(len(self.instances)):
            self.instances[i].emb = embs[i]

        
class VoiceModel(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.lstm_enc = nn.LSTM(input_size=window, hidden_size=512, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=512, hidden_size=256)
        self.input = nn.GRU(input_size=window, bidirectional=True, hidden_size=512)
        self.gru = nn.GRU(input_size=512*2, bidirectional=False, hidden_size=512)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(128, max_len_text)
        self.output = nn.Linear(max_len_text*100, 1)
    
    def forward(self, audio):
        audio = Tensor(audio)
        audio, hidden_enc = self.lstm_enc(audio.reshape(1, -1, window))
        audio = tanh(audio)
        audio, hidden_dec = self.lstm_dec(audio)
        audio = tanh(audio)
        audio = F.relu(self.dense(audio))
        audio = self.flatten(audio)
        # that's to see if it works, later I will figure an appropriate value
        max_sound_len = 3000
        audio = nn.functional.pad(audio, (max_sound_len-audio.shape[-1], 0), mode='constant', value=0)
        audio = self.output(audio)
        print(audio.shape)
        
        return audio

def main():
    
    batch_size = 16
    epochs = 10
    model = VoiceModel()
    criterion = nn.CosineSimilarity(dim=0)
    optimizer = Adam(model.parameters(), lr=0.001)
    info_frame = pd.read_csv(os.path.join(data_dir, 'cv-valid-train.csv'))[:50]	# .loc[25:75, :]
    
    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, bert_vocab_path), return_tensors='pt')
    
    real_texts = []
    lengths = []
    for i in info_frame.index:
        tokens = tokenizer.tokenize(info_frame.loc[i, 'text'])
        real_texts.append(tokenizer.convert_tokens_to_ids(tokens))
        lengths.append(len(real_texts[i]))
        
    print(info_frame.columns)
    dataset = VoiceDataset(info_frame=info_frame, audio_dir=audio_dir, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        running_loss = 0
        i=0
        for batch in dataset.instances:
            
            optimizer.zero_grad()
            output = model(batch.mfcc)
            i += 1
        
    model.eval()
    
    test_frame = pd.read_csv(os.path.join(data_dir, 'cv-valid-train.csv'))[50:75]
    test_frame.reset_index()
    print(test_frame.index)
    
    for text, file in zip(test_frame['text'], test_frame['filename']):
        tokens = tokenizer.tokenize(text)
        real_emb = tokenizer.convert_tokens_to_ids(tokens)
        print('real embedding:', real_emb)
        test_instance = VoiceInstance(file='cv-valid-train/'+file, 
                                      tokenizer=tokenizer)
        pred = model(test_instance.mfcc)
        print('predicted:', pred)
            
if __name__ == '__main__':
    main()

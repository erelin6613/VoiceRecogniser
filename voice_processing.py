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
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from keras_preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, DistilBertModel
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

sys.path.insert(0, "../input/transformers/transformers-master/")

data_dir = '../input/common-voice'
audio_dir = 'cv-valid-train'
sound_len = 10
bert_path = '../input/distilbertbaseuncased/'
bert_model_path = 'bert-base-uncased'
max_len_text = 30
window = 40

class VoiceInstance:
    
    def __init__(self, file, tokenizer, lang_model, sound_len=10**10, text=None):
        
        self.file = file
        self.text = text
        self.tokenizer = tokenizer
        self.lang_model = lang_model
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
        
        def get_inputs():
            input_ids = Tensor(self.tokenizer.encode(self.text, add_special_tokens=True)).unsqueeze(0)
            return self.lang_model(input_ids.long())
        
        
        if self.text:
            self.text = remove_punkts(self.text)
            emb = get_inputs()
            return emb

class VoiceDataset(Dataset):
    
    def __init__(self, info_frame, audio_dir, tokenizer, lang_model):
        
        self.info_frame = info_frame
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.lang_model = lang_model
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
                                      text=self.info_frame.loc[i, 'text'], 
                                      tokenizer=self.tokenizer,
                                      lang_model=self.lang_model)
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
        self.lstm_enc = nn.LSTM(input_size=max_len_text, hidden_size=512, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=512, hidden_size=256)
        self.input = nn.GRU(input_size=window, bidirectional=True, hidden_size=512)
        self.gru = nn.GRU(input_size=512*2, bidirectional=False, hidden_size=512)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256, max_len_text*3)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(128, max_len_text)
        self.output = nn.Linear(max_len_text*100, max_len_text)
    
    def forward(self, audio):
        audio = Tensor(audio)
        audio = nn.functional.pad(audio, (max_len_text - audio.shape[-1], 0), mode='constant', value=0)
        audio, hidden_enc = self.lstm_enc(audio)
        audio = tanh(audio)
        audio, hidden_dec = self.lstm_dec(audio)
        audio = tanh(audio)
        audio = F.relu(self.dense(audio))
        audio = self.dropout(audio)
        audio = self.flatten(audio).T
        audio = mean(audio.view(max_len_text, -1), 1)
        
        return audio

def main():
    
    #info_frame = pd.read_csv('../input/common-voice/cv-valid-train.csv')
    
    batch_size = 16
    epochs = 1
    model = VoiceModel()
    criterion = nn.MSELoss()
    #criterion.requres_grad = True
    optimizer = Adam(model.parameters(), lr=0.001)
    info_frame = pd.read_csv(os.path.join(data_dir, 'cv-valid-train.csv')) [:100]	# .loc[25:75, :]
    print(len(info_frame))
    
    tokenizer = DistilBertTokenizer.from_pretrained(bert_path)
    vocab_size = len(tokenizer)
    print(tokenizer)
    #model = BertModel.from_pretrained('bert-base-uncased')
    print(os.path.join(bert_path, bert_model_path))
    lang_model = DistilBertModel.from_pretrained(bert_path)
    
    real_texts = []
    lengths = []
    for i in info_frame.index:
        tokens = tokenizer.tokenize(info_frame.loc[i, 'text'])
        real_texts.append(tokenizer.convert_tokens_to_ids(tokens))
        lengths.append(len(real_texts[i]))
        
    #print(info_frame.columns)
    #print(lengths)
    
    
    dataset = VoiceDataset(info_frame=info_frame, audio_dir=audio_dir, 
                           tokenizer=tokenizer, lang_model=lang_model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #print(dataset.instances[0].fft)
    
    for epoch in range(epochs):
        running_loss = 0
        i=0
        for batch in dataset.instances:
            scaled_emb = scale_embeddings(batch.emb, vocab_size)
            optimizer.zero_grad()
            #print(batch.fft) # torch.Size([1, 40, 10])
            #print(batch.stft.shape)
            output = model(batch.mfcc)
            #print(output, batch.emb)
            #loss = criterion(batch.emb, real_texts[i])
            #print(batch.emb) #, real_texts[i])
            #print('models output shape:', output.shape, '\nlabels shape:', Tensor(batch.emb).shape)
            loss = criterion(output, Tensor(batch.emb))
            
            loss.backward()
            print(output, Tensor(batch.emb))
            print('loss:', loss)
            #optimizer.step()
            #running_loss += loss.item()
            #print(running_loss)
            i += 1
        # print(loss.mean())
        
    model.eval()
    
    test_frame = pd.read_csv(os.path.join(data_dir, 'cv-valid-train.csv')) [100:125]
    print(len(test_frame))
    test_frame.reset_index()
    print(test_frame.index)
    
    for text, file in zip(test_frame['text'], test_frame['filename']):
        print('real embedding:', text)
        test_instance = VoiceInstance(file='cv-valid-train/'+file, 
                                      tokenizer=tokenizer)
        pred = model(test_instance.mfcc)
        #pred = [int(x*vocab_size) for x in pred]
        #pred = tokenizer.decode(pred)
        print('predicted:', pred)
            
if __name__ == '__main__':
    main()

#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 12 Making Predictions with Sequences Using Recurrent Neural Networks
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Writing your own War and Peace with RNNs 

# ## Acquiring and analyzing the training data 

with open('warpeace_input.txt', 'r', encoding="utf8") as fp:
    raw_text = fp.read()
raw_text = raw_text.lower()


print(raw_text[:200])


all_words = raw_text.split()
unique_words = list(set(all_words))
print(f'Number of unique words: {len(unique_words)}')


n_chars = len(raw_text)
print(f'Total characters: {n_chars}')


chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
print(f'Total vocabulary (unique characters): {vocab_size}')
print(chars)


# ## Constructing the training set for the RNN text generator

index_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))
print(char_to_index)


import numpy as np
text_encoded = np.array(
    [char_to_index[ch] for ch in raw_text],
    dtype=np.int32)


seq_length = 40
chunk_size = seq_length + 1

text_chunks = np.array([text_encoded[i:i+chunk_size] 
               for i in range(len(text_encoded)-chunk_size+1)]) 


import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()
    
seq_dataset = SeqDataset(torch.from_numpy(text_chunks))


from torch.utils.data import DataLoader
 
batch_size = 64

torch.manual_seed(0)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ## Building and Training an RNN text generator 

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_dim, 
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_dim)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_dim)
        return hidden, cell


embed_dim = 256
rnn_hidden_dim = 512

torch.manual_seed(0)
model = RNN(vocab_size, embed_dim, rnn_hidden_dim) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model 


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


num_epochs = 10000

torch.manual_seed(0)

for epoch in range(num_epochs):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    seq_batch = seq_batch.to(device)
    target_batch = target_batch.to(device)
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden.to(device), cell.to(device)) 
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item()/seq_length
    if epoch % 500 == 0:
        print(f'Epoch {epoch} - loss: {loss:.4f}')


from torch.distributions.categorical import Categorical

def generate_text(model, starting_str, len_generated_text=500):
    encoded_input = torch.tensor([char_to_index[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))

    generated_str = starting_str

    model.eval()

    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str)-1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell) 
    
    last_char = encoded_input[:, -1]
    for _ in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell) 
        logits = torch.squeeze(logits, 0)
        last_char = Categorical(logits=logits).sample()
        generated_str += str(index_to_char[last_char.item()])
        
    return generated_str


model.to('cpu')
torch.manual_seed(0)
print(generate_text(model, 'the emperor', 500))


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch12_part3.ipynb --TemplateExporter.exclude_input_prompt=True')


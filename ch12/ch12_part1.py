#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 12 Making Predictions with Sequences Using Recurrent Neural Networks
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Analyzing movie review sentiment with RNNs

# ## Analyzing and preprocessing the data 

from torchtext.datasets import IMDB

train_dataset = list(IMDB(split='train'))
test_dataset = list(IMDB(split='test'))

print(len(train_dataset), len(test_dataset))


# !conda install -c pytorch torchtext -y


# !conda install -c conda-forge portalocker -y


import re
from collections import Counter, OrderedDict

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized

token_counts = Counter()
train_labels = []
for label, line in train_dataset:
    train_labels.append(label)
    tokens = tokenizer(line)
    token_counts.update(tokens)
 
    
print('Vocab-size:', len(token_counts))
print(Counter(train_labels))


from torchtext.vocab import vocab

sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)

vocab_mapping = vocab(ordered_dict)

vocab_mapping.insert_token("<pad>", 0)
vocab_mapping.insert_token("<unk>", 1)
vocab_mapping.set_default_index(1)


print([vocab_mapping[token] for token in ['this', 'is', 'an', 'example']])
print([vocab_mapping[token] for token in ['this', 'is', 'example2']])


import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_transform = lambda x: [vocab[token] for token in tokenizer(x)]    

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(1. if _label == 2 else 0.)
        processed_text = [vocab_mapping[token] for token in tokenizer(_text)]    
        text_list.append(torch.tensor(processed_text, dtype=torch.int64))
        lengths.append(len(processed_text))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(
        text_list, batch_first=True)
    return padded_text_list.to(device), label_list.to(device), lengths.to(device)


# from torch.nn.utils.rnn import pad_sequence
# a = [torch.tensor([11, 7, 35, 462], dtype=torch.int64), torch.tensor([11, 7, 35, 462, 11], dtype=torch.int64)]
# b = [torch.tensor([11, 7, 35], dtype=torch.int64), torch.tensor([11, 7, 35, 462, 11, 12], dtype=torch.int64)]
# # c = torch.ones(1, 15, 300)
# pad_sequence(a, True).size()


from torch.utils.data import DataLoader
torch.manual_seed(0)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)
text_batch, label_batch, length_batch = next(iter(dataloader))
print(text_batch)
print(label_batch)
print(length_batch)
print(text_batch.shape)


batch_size = 32  

train_dl = DataLoader(train_dataset, batch_size=batch_size,
                      shuffle=True, collate_fn=collate_batch)

test_dl = DataLoader(test_dataset, batch_size=batch_size,
                     shuffle=False, collate_fn=collate_batch)


# ## Building a simple LSTM network 

vocab_size = len(vocab_mapping)
embed_dim = 32
rnn_hidden_dim = 50
fc_hidden_dim = 32


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_dim, fc_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 
                                      embed_dim, 
                                      padding_idx=0) 
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_dim, 
                           batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_dim, fc_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
 


model = RNN(vocab_size, embed_dim, rnn_hidden_dim, fc_hidden_dim) 
model = model.to(device)


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


def train(model, dataloader, optimizer):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, length_batch in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, length_batch)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)

    total_loss /= len(dataloader.dataset)
    total_acc /= len(train_dl.dataset)
    print(f'Epoch {epoch+1} - loss: {total_loss:.4f} - accuracy: {total_acc:.4f}')
 


torch.manual_seed(0)
num_epochs = 10 
for epoch in range(num_epochs):
    train(model, train_dl, optimizer)


def evaluate(model, dataloader):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
    print(f'Accuracy on test set: {100 * total_acc/len(dataloader.dataset)} %')
 
evaluate(model, test_dl)


# ## Stacking multiple LSTM layers 

nn.LSTM(embed_dim, rnn_hidden_dim, num_layers=2, batch_first=True)


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch12_part1.ipynb --TemplateExporter.exclude_input_prompt=True')


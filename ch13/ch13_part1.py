#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 13 Advancing language understanding and Generation with the Transformer models
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Understanding self-attention 

import torch

sentence = torch.tensor(
    [0, # python 
     8, # machine      
     1, # learning 
     6, # by 
     2] # example 
)

sentence


torch.manual_seed(0)
embed = torch.nn.Embedding(10, 16)
sentence_embed = embed(sentence).detach()


sentence_embed


d = sentence_embed.shape[1]
w_key = torch.rand(d, d)
w_query = torch.rand(d, d)
w_value = torch.rand(d, d)


token1_embed = sentence_embed[0]
key_1 = w_key.matmul(token1_embed)
query_1 = w_query.matmul(token1_embed)
value_1 = w_value.matmul(token1_embed)


key_1


keys = sentence_embed.matmul(w_key.T)


keys[0]


values = sentence_embed.matmul(w_value.T)


import torch.nn.functional as F
a1 = F.softmax(query_1.matmul(keys.T) / d ** 0.5, dim=0)


a1


z1 = a1.matmul(values)
z1


# # Improving sentiment analysis with BERT and Transformers

# ## Fine-tuning a pre-trained BERT model for sentiment Analysis

from torchtext.datasets import IMDB

train_dataset = list(IMDB(split='train'))
test_dataset = list(IMDB(split='test'))

print(len(train_dataset), len(test_dataset))


train_texts = [train_sample[1] for train_sample in train_dataset]
train_labels = [train_sample[0] for train_sample in train_dataset]

test_texts = [test_sample[1] for test_sample in test_dataset]
test_labels = [test_sample[0] for test_sample in test_dataset]


import transformers
from transformers import DistilBertTokenizerFast

# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', local_files_only=True)


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


train_encodings[0] 


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([0., 1.] if self.labels[idx] == 2 else [1., 0.])
        return item

    def __len__(self):
        return len(self.labels)


train_encoded_dataset = IMDbDataset(train_encodings, train_labels)
test_encoded_dataset = IMDbDataset(test_encodings, test_labels)


batch_size = 32
train_dl = torch.utils.data.DataLoader(train_encoded_dataset, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_encoded_dataset, batch_size=batch_size, shuffle=False)


from transformers import DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', local_files_only=True)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)


def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss'] 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*len(batch)

    return total_loss/len(dataloader.dataset)
    


def evaluate(model, dataloader):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            pred = torch.argmax(logits, 1)
            total_acc += (pred == torch.argmax(labels, 1)).float().sum().item()

    return  total_acc/len(dataloader.dataset)
 


torch.manual_seed(0)
num_epochs = 1 
for epoch in range(num_epochs):
    train_loss = train(model, train_dl, optimizer)
    train_acc = evaluate(model, train_dl)
    print(f'Epoch {epoch+1} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f}')


test_acc = evaluate(model, test_dl)
print(f'Accuracy on test set: {100 * test_acc:.2f} %')


# torch.cuda.mem_get_info()


# torch.cuda.empty_cache()


# free up memory
del model 


# ## Using the Trainer API to train Transformer models 

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', local_files_only=True)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


# !conda install -c conda-forge accelerate -y


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=1,     
    per_device_train_batch_size=32, 
    logging_dir='./logs',
    logging_steps=50,
)


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_encoded_dataset,
#     optimizers=(optim, None)
# )


from datasets import load_metric
import numpy as np

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred  
    pred = np.argmax(logits, axis=-1)
    return metric.compute(predictions=pred, references=np.argmax(labels, 1))


trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=training_args,
    train_dataset=train_encoded_dataset,
    eval_dataset=test_encoded_dataset,
    optimizers=(optim, None)
)


trainer.train()


print(trainer.evaluate())


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch13_part1.ipynb --TemplateExporter.exclude_input_prompt=True')


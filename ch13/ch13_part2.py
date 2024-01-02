#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 13 Advancing language understanding and Generation with the Transformer models
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Generating text using GPT 

# ## Writing your own War and Peace with GPT

from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')
set_seed(0)
generator("I love machine learning",
          max_length=20,
          num_return_sequences=3)


from transformers import TextDataset, GPT2Tokenizer

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)


text_dataset = TextDataset(tokenizer=tokenizer, file_path='warpeace_input.txt', block_size=128)


len(text_dataset)


from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


import torch
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optim = torch.optim.Adam(model.parameters(), lr=5e-5)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./gpt_results', 
    num_train_epochs=20,     
    per_device_train_batch_size=16, 
    logging_dir='./gpt_logs',
    save_total_limit=1,
    logging_steps=500,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=text_dataset,
    optimizers=(optim, None)
)


trainer.train()


def generate_text(prompt_text, model, tokenizer, max_length):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    # Generate response
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.9,
    )

    # Decode the generated responses
    responses = []
    for response_id in output_sequences:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses


prompt_text = "the emperor"
responses = generate_text(prompt_text, model, tokenizer, 100)

for response in responses:
    print(response)


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch13_part2.ipynb --TemplateExporter.exclude_input_prompt=True')


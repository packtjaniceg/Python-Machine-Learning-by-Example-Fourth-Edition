#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 14 Building an Image Search Engine Using Multimodal Models
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Finding images with words 

# ## Image search using the pre-trained CLIP model

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('clip-ViT-B-32')


import os
import glob
from PIL import Image
import torch


image_paths = list(glob.glob('flickr8k/Flicker8k_Dataset/*.jpg'))

all_image_embeddings = []
for img_path in image_paths:
    img = Image.open(img_path)
    all_image_embeddings.append(model.encode(img, convert_to_tensor=True))
 


import matplotlib.pyplot as plt
 

def search_top_images(model, image_embeddings, query, top_k=1):
    query_embeddings = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_embeddings,  image_embeddings, top_k=top_k)[0]
    return hits


query = "a swimming dog"
hits = search_top_images(model, all_image_embeddings, query)

for hit in hits:
    img_path = image_paths[hit['corpus_id']]
    image = Image.open(img_path)
    plt.imshow(image)
    plt.title(f"Query: {query}")
    plt.show()


image_query = Image.open("flickr8k/Flicker8k_Dataset/240696675_7d05193aa0.jpg")
hits = search_top_images(model, all_image_embeddings, image_query, 3)[1:]

plt.imshow(image_query)
plt.title(f"Query image")
plt.show()

for hit in hits:
    img_path = image_paths[hit['corpus_id']]
    image = Image.open(img_path)
    plt.imshow(image)
    plt.title(f"Similar image")
    plt.show()


# ## Zero-shot classification

from torchvision.datasets import CIFAR100
cifar100 = CIFAR100(root="CIFAR100", download=True, train=False)


print(cifar100.classes)
print("Number of classes in CIFAR100 dataset:", len(cifar100.classes))


sample_index = 0
img, class_id = cifar100[sample_index]
print(f"Class of the sample image: {class_id} - {cifar100.classes[class_id]}")


sample_image_embeddings = model.encode(img, convert_to_tensor=True)


class_text = model.encode(cifar100.classes, convert_to_tensor=True)


hits = util.semantic_search(sample_image_embeddings,  class_text, top_k=1)[0]
pred = hits[0]['corpus_id']
print(f"Predicted class of the sample image: {pred}")


all_image_embeddings = []
class_true = []
for img, class_id in cifar100:
    class_true.append(class_id)
    all_image_embeddings.append(model.encode(img, convert_to_tensor=True))


class_pred = []
for hit in util.semantic_search(all_image_embeddings,  class_text, top_k=1):
    class_pred.append(hit[0]['corpus_id'])


from sklearn.metrics import accuracy_score
acc = accuracy_score(class_true, class_pred)
print(f"Accuracy of zero-shot classification: {acc * 100}%")


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch14_part2.ipynb --TemplateExporter.exclude_input_prompt=True')





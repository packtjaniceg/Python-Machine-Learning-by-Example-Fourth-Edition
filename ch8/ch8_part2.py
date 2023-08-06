#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 8 Discovering Underlying Topics in the Newsgroups Dataset with Clustering and Topic Modeling
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Clustering newsgroups dataset

# ## Clustering newsgroups data using k-means 

from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

groups = fetch_20newsgroups(subset='all', categories=categories)

labels = groups.target
label_names = groups.target_names


from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def get_cleaned_data(groups, lemmatizer, remove_words):
    data_cleaned = []

    for doc in groups.data:
        doc = doc.lower()
        doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in remove_words)
        data_cleaned.append(doc_cleaned)
        
    return data_cleaned

data_cleaned = get_cleaned_data(groups, lemmatizer, all_names)


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)
data_cv = count_vector.fit_transform(data_cleaned)


from sklearn.cluster import KMeans
k = 4
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)

kmeans.fit(data_cv)


clusters = kmeans.labels_

from collections import Counter
print(Counter(clusters))


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vector = TfidfVectorizer(stop_words='english', max_features=None, max_df=0.5, min_df=2)


data_tv = tfidf_vector.fit_transform(data_cleaned)
kmeans.fit(data_tv)
clusters = kmeans.labels_
print(Counter(clusters))


import numpy as np
cluster_label = {i: labels[np.where(clusters == i)] for i in range(k)}

terms = tfidf_vector.get_feature_names_out()
centroids = kmeans.cluster_centers_
for cluster, index_list in cluster_label.items():
    counter = Counter(cluster_label[cluster])
    print(f'cluster_{cluster}: {len(index_list)} samples')
    for label_index, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print(f'- {label_names[label_index]}: {count} samples')
    print('Top 10 terms:')
    for ind in centroids[cluster].argsort()[-10:]:
        print('%s ' % terms[ind], end="")
    print('\n')


# ## Describing the clusters using GPT 

keywords = ' '.join(terms[ind] for ind in centroids[0].argsort()[-100:])  


print(keywords)


import openai


# openai.api_key = '<YOUR API KEY>'


def get_completion(prompt, model="text-davinci-003"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message["content"]


# response = get_completion(f"Describe a common topic based on the following keywords: {keywords}")
# print(response)


# # Discovering underlying topics in newsgroups 

# ## Topic modeling using NMF 

from sklearn.decomposition import NMF

t = 20
nmf = NMF(n_components=t, random_state=42)


nmf.fit(data_cv)

print(nmf.components_)


terms_cv = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms_cv[i] for i in topic.argsort()[-10:]]))


# ## Topic modeling using LDA 

from sklearn.decomposition import LatentDirichletAllocation

t = 20
lda = LatentDirichletAllocation(n_components=t, learning_method='batch',random_state=42)


lda.fit(data_cv)

print(lda.components_)


for topic_idx, topic in enumerate(lda.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms_cv[i] for i in topic.argsort()[-10:]]))


data_cleaned = get_cleaned_data(groups_3, lemmatizer, all_names)


data_embedding = []

for doc in data_cleaned:
#     print(doc)
    doc_vector = np.mean([model[word] for word in doc.split() if word in model], axis=0)
    data_embedding.append(doc_vector)
 
        
data_tsne = tsne_model.fit_transform(np.array(data_embedding))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_3.target)

plt.show()


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch8_part2.ipynb --TemplateExporter.exclude_input_prompt=True')


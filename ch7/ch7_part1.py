#!/usr/bin/env python
# coding: utf-8

# 
# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 7 Mining the 20 Newsgroups Dataset with Text Analysis Techniques
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
# 

# # Touring popular NLP libraries and picking up NLP basics 

# ## Corpora 

import nltk
# nltk.download()


from nltk.corpus import names
print(names.words()[:10])

print(len(names.words()))


# ## Tokenization

from nltk.tokenize import word_tokenize
sent = '''I am reading a book.
          It is Python Machine Learning By Example,
          4th edition.'''

print(word_tokenize(sent))


sent2 = 'I have been to U.K. and U.S.A.'
print(word_tokenize(sent2))


import spacy

nlp = spacy.load('en_core_web_sm')
tokens2 = nlp(sent2)

print([token.text for token in tokens2])


from nltk.tokenize import sent_tokenize
print(sent_tokenize(sent))


# ## PoS tagging 

import nltk
tokens = word_tokenize(sent)
print(nltk.pos_tag(tokens))


nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('VBP')


print([(token.text, token.pos_) for token in tokens2])


# ## NER

tokens3 = nlp('The book written by Hayden Liu in 2024 was sold at $30 in America')
print([(token_ent.text, token_ent.label_) for token_ent in tokens3.ents])


# ## Stemming and lemmatization 

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()


porter_stemmer.stem('machines')


porter_stemmer.stem('learning')


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


lemmatizer.lemmatize('machines')


lemmatizer.lemmatize('learning')


# # Getting the newsgroups data

from sklearn.datasets import fetch_20newsgroups


groups = fetch_20newsgroups()


groups.keys()


groups['target_names']


groups['target']


import numpy as np
np.unique(groups.target)


import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(groups.target, bins=20)
plt.xticks(range(0, 20, 1))
plt.show()


groups.data[0]


groups.target[0]


groups.target_names[groups.target[0]]


# # Thinking about features for text data

# ## Counting the occurrence of each word token 

from sklearn.feature_extraction.text import CountVectorizer


count_vector = CountVectorizer(max_features=500)
data_count = count_vector.fit_transform(groups.data)


data_count


data_count[0]


data_count.toarray()[0]


print(count_vector.get_feature_names_out())


# ## Text preprocessing

data_cleaned = []
for doc in groups.data:
    doc_cleaned = ' '.join(word for word in doc.split() if word.isalpha())
    data_cleaned.append(doc_cleaned)


# ## Dropping stop words 

from sklearn.feature_extraction import _stop_words
print(_stop_words.ENGLISH_STOP_WORDS)


count_vector = CountVectorizer(stop_words="english",max_features=500)


# ## Reducing inflectional and derivational forms of words 

all_names = set(names.words())


def get_cleaned_data(groups, lemmatizer, remove_words):
    data_cleaned = []

    for doc in groups.data:
        doc = doc.lower()
        doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in remove_words)
        data_cleaned.append(doc_cleaned)
        
    return data_cleaned


count_vector_sw = CountVectorizer(stop_words="english", max_features=500)

data_cleaned = get_cleaned_data(groups, lemmatizer, all_names)

data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)


sum(len(set(doc.split())) for doc in data_cleaned)


print(count_vector_sw.get_feature_names_out())


# # Visualizing the newsgroups data with t-SNE 

# ## t-SNE for dimensionality reduction 

from sklearn.manifold import TSNE


categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']

groups_3 = fetch_20newsgroups(categories=categories_3)


data_cleaned = get_cleaned_data(groups_3, lemmatizer, all_names)
 
data_cleaned_count_3 = count_vector_sw.fit_transform(data_cleaned)


tsne_model = TSNE(n_components=2,  perplexity=40, random_state=42, learning_rate=500)

data_tsne = tsne_model.fit_transform(data_cleaned_count_3.toarray())


plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_3.target)
plt.show()


categories_5 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'comp.windows.x']
groups_5 = fetch_20newsgroups(categories=categories_5)

data_cleaned = get_cleaned_data(groups_5, lemmatizer, all_names)
 
data_cleaned_count_5 = count_vector_sw.fit_transform(data_cleaned)

data_tsne = tsne_model.fit_transform(data_cleaned_count_5.toarray())

plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_5.target)

plt.show()


# # Building embedding models using shadow neural networks

# ## Utilizing pre-trained embedding models 

import gensim.downloader as api
model = api.load("glove-twitter-25")


vector = model['computer']
print('Word computer is embedded into:\n', vector)


similar_words = model.most_similar("computer")
print('Top ten words most contextually relevant to computer:\n', 
           similar_words)


doc_sample = ['i', 'love', 'reading', 'python', 'machine', 
                 'learning', 'by', 'example']
doc_vector = np.mean([model[word] for word in doc_sample], axis=0)
print('The document sample is embedded into:\n', doc_vector)


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch7_part1.ipynb --TemplateExporter.exclude_input_prompt=True')


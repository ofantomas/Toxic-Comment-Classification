import torch
import os
import itertools, functools
import pickle
import tqdm
import numpy as np
import pandas as pd
import nltk, re


class DataLoader():
    def __init__(self, path, embeddings_path, embeddings_size, maxlen, batch_size=64, training=True, lemmatize=False, verbose=True):
        self.path = path
        self.emb_path = embeddings_path
        self.emb_size = embeddings_size
        self.training = training
        self.lemm = lemmatize
        self.ind = maxlen
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.load_embeddings()
        self.load()
        self.transform()
    
    def __len__(self):
        return len(self.text)
    
    def load(self):
        if self.verbose is True:
            print("Loading text...")
                 
        
        def edit_text(sentence):
            return ' '.join([elem for elem in re.split('\W+', str.lower(sentence))])
        
        self.text = pd.read_csv(self.path, sep=',')
        self.columns = self.text.columns
        self.id = self.text.id
        
        if self.verbose is True:
            print("Done.")
        
        if self.training is True:
            self.labels = self.text.iloc[:, 2:].values
            
        if self.verbose is True:
            print("Perfoming text editing...")
        
        self.text = list(map(edit_text, self.text.iloc[:, 1].values))
        
        if self.verbose is True:
            print("Done.")
        
        if self.lemm is True:
            self.text = self.lemmatize(self.text)
            
    def load_embeddings(self):
        if self.verbose is True:
            print("Loading embeddings from {}...".format(self.emb_path))
            
        self.emb = []
        self.word_to_ix = {}

        for i, line in enumerate(open(os.getcwd() + '\\' + self.emb_path, encoding='utf-8')):
            if i == 0:
                continue
            if i == self.ind + 1:
                break
            line = line.rstrip('\n').strip().split(' ')
            self.word_to_ix.update({line[0]:i - 1})
            self.emb.append(np.array(list(map(lambda x:float(x), line[1:]))))

        self.emb.append(np.zeros(self.emb_size))
        
        self.emb = torch.from_numpy(np.array(self.emb))
        
        if self.verbose is True:
            print("Done.")
            
    def transform(self):
        if self.verbose is True:
            print("Transforming text into a sequence of indices...")
        
        def get_indices(sentence):
            res = []
            for word in sentence.split(' '):
                ind = self.word_to_ix.get(word)
                if ind is None:
                    continue
                else:
                    res.append(ind)
            return res
    
        if self.verbose is True:
            print("Done.")
        
        self.text = list(map(get_indices, self.text))
  
    def lemmatize(self, text):
        if self.verbose is True:
            print("Perfoming text lemmatization...")
        
        def get_wordnet_pos(tag):
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                return nltk.corpus.wordnet.NOUN
            elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                return nltk.corpus.wordnet.VERB
            elif tag in ['RB', 'RBR', 'RBS']:
                return nltk.corpus.wordnet.ADV
            elif tag in ['JJ', 'JJR', 'JJS']:
                return nltk.corpus.wordnet.ADJ
            else:
                return nltk.corpus.wordnet.NOUN

        def simple_lemmatizer(sentence):
            tokenized_sent = sentence.split()
            pos_taged = [(word, get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(tokenized_sent)]
            return " ".join([lemmatizer.lemmatize(word, tag) for word, tag in pos_taged])
        
        lemmatizer = nltk.WordNetLemmatizer()
        
        if self.verbose is True:
            print("Done.")
        
        return list(map(simple_lemmatizer, text))
    
    def batch_generator(self):
        for i in range(0, len(self.text), self.batch_size):
            batch = self.text[i:i + self.batch_size]
            max_len = max(list(map(lambda x:len(x), batch)))
            if self.training is True:
                yield (torch.LongTensor(list(map(lambda x:x + [self.ind] * (max_len - len(x)), batch))), 
                       torch.FloatTensor(self.labels[i:i + self.batch_size]))
            else:
                yield torch.LongTensor(list(map(lambda x:x + [self.ind] * (max_len - len(x)), batch)))
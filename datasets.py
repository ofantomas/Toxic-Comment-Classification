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
    
    def load(self, df=None):
        if self.verbose is True:
            print("Loading text...")
        
        if df is None:
            assert self.path is not None
            self.text = pd.read_csv(self.path, sep=',')
        else:
            self.text = df
            
        self.columns = self.text.columns
        print(self.text.columns)
        self.id = self.text.id
        
        if self.verbose is True:
            print("Done.")
        
        if self.training is True:
            self.labels = self.text.iloc[:, 2:].values
            
        if self.verbose is True:
            print("Perfoming text editing...")
        
        self.text = list(map(self.preprocess, self.text.iloc[:, 1].values))
        
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
            line = line.rstrip('\n').rstrip().split(' ')
            self.word_to_ix.update({line[0].lower():i - 1})
            self.emb.append(np.array(list(map(lambda x:float(x), line[1:]))))

        self.emb.append(np.zeros(self.emb_size))
        self.emb = torch.from_numpy(np.stack(self.emb, axis=0))
        
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
                
    def preprocess(self, text):
        s_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
        specials = ["’", "‘", "´", "`"]
        p_mapping = {"_":" ", "`":" "}    
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        for s in specials:
            text = text.replace(s, "'")
        
        text = ' '.join([s_mapping[t] if t in s_mapping else t for t in text.lower().split(" ")])
        
        for p in p_mapping:
            text = text.replace(p, p_mapping[p])
        
        return ' '.join([elem for elem in re.split('\W+', text)]).strip('\n')    
import torch
import os
import itertools, functools
import pickle
import tqdm
import numpy as np
import pandas as pd
import nltk, re
import argparse as ap

from models import *
from datasets import *


class ArgParser():
    def __init__(self):
        self.parser = ap.ArgumentParser()
        self.parser.add_argument('train_path', type=str, help='path to training data')
        self.parser.add_argument('test_path', type=str, help='path to test data')
        self.parser.add_argument('save_pred_path', type=str, help='path to save final prediction')
        self.parser.add_argument('--embeddings', nargs='+', type=str, help='list of paths to embedding files')
        self.parser.add_argument('--embeddings_sizes', nargs='+', type=int, help='list of ints corresponding to embeddings sizes')
        self.parser.add_argument('-s', '--max_embeddings', type=int, default=1000000, help='ammount of words to use embeddings for')
        self.parser.add_argument('-l', '--lemmatize', action='store_true', help='lemmatize text')
        self.parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch_size')
        self.parser.add_argument('--num_classes', type=int, help='number of classes for classification')
        self.parser.add_argument('--num_epochs', type=int, help='number of epochs to train')
        self.parser.add_argument('-v', '--verbosity', action='store_true', help='increase verbosity')

    def parse(self):
        args = self.parser.parse_args()
        assert args.embeddings is not None, "List of embeddings' paths must not be empty"
        assert len(args.embeddings) == len(args.embeddings_sizes), "List of embeddings' paths and list of\
                                                                   embedding sizes must be of the same length"
        return args
        
        
def main():
    parser = ArgParser()
    args = vars(parser.parse())
    
    #nltk stuff needed for text processing
    #if args['lemmatize'] is True:
    #    nltk.download('averaged_perceptron_tagger')
    #    nltk.download('wordnet')
    
    hidden_size = 128
    num_layers = 2
    learning_rate = 0.005
    num_classes = args['num_classes']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    
    prediction_list = []
    for emb, emb_size in zip(args['embeddings'], args['embeddings_sizes']):    
        print("Current embedding path: {}".format(emb))
        print("Initializing train loader...")
        TrainLoader = DataLoader(path=args['train_path'], embeddings_path=emb, embeddings_size=emb_size, maxlen=args['max_embeddings'], 
                                 batch_size=args['batch_size'], training=True, lemmatize=args['lemmatize'], verbose=args['verbosity'])

        print("Initializing test loader...")
        TestLoader = DataLoader(path=args['test_path'], embeddings_path=emb, embeddings_size=emb_size, maxlen=args['max_embeddings'], 
                                 batch_size=args['batch_size'], training=False, lemmatize=args['lemmatize'], verbose=args['verbosity'])
        
        print("Initializing the model...")
        model = nnPredictor(emb_size, hidden_size, num_layers, num_classes, TrainLoader.emb)
        print("Training...")
        model.train(TrainLoader, num_epochs, verbose_step=args['batch_size'] * 50)
        print("Done.")
        
        prediction_list.append(model.predict(TestLoader))
        #model.save(...)
        
    print("Saving predictions to {}".format(args['save_pred_path']))
    predictions = torch.mean(torch.stack(prediction_list), 0)
    submission_lem = pd.DataFrame(predictions.numpy(), columns=TrainLoader.columns[2:])
    submission_lem = pd.concat((TestLoader.id, submission_lem), axis=1)
    submission_lem.to_csv(args['save_pred_path'], index=False)
    
    
    
if __name__ == '__main__':
    main()
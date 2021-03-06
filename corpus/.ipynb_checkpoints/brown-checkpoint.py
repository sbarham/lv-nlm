# python
import os
import pickle
import random
import math
from collections import defaultdict

# torch
import torch
from torch.utils.data import Dataset

# torchtext
import torchtext.vocab

# numpy
import numpy as np

# nltk
from nltk.corpus import brown as brown

# tqdm
from tqdm import tqdm_notebook as tqdm

# our code
import util
from util.utils import OrderedCounter
from corpus.base import Corpus
from corpus.util import load_embeddings, get_num_lines      

class Brown(Corpus):
    def __init__(self, split, create_data, name='brown', embeddings=False, train_split=0.8,
                 val_split=0.1, test_split=0.1, random_seed=1111, **kwargs):
        super().__init__(split, create_data, embeddings, **kwargs)
        
        # set the random seed, for reproducibility
        random.seed(random_seed)
        
        # this takes a while; do it now
        self.corpus_length = len(brown.sents())
        
        # compute the directory/path where this split's indices are stored
        # self.data_indices_dir = '.' # os.path.join(util.DATA_DIR, util.BROWN_DIR)
        # self.data_indices_path = os.path.join(self.data_indices_dir, split + '.indices')
        self.data_indices_path = split + '.indices'
        
        # if this is the TRAIN split, generate the split indices
        if split == util.TRAIN:
            self.generate_split(train_split, val_split, test_split)
        
        # load the split indices
        with open(self.data_indices_path, 'rb') as indices:
            self.data_indices = pickle.load(indices)
        
        self.name = name
        
        # load the vocab
        self.vocab_file = name + '.vocab.pickle'
        
        # preprocess the data
        print("Preprocessing Brown Corpus *{}* data:".format(self.split))
        print("------------------------------------------")
        self._create_data()
        
    def generate_split(self, train_split, val_split, test_split):
        # ensure this makes for a valid split
        assert train_split + val_split + test_split == 1.0

        # get a shuffled list of indices
        indices = list(range(self.corpus_length))
        random.shuffle(indices)

        # calculate the break indices
        train_stop = math.floor(train_split * len(indices))
        val_stop = train_stop + math.floor(val_split * len(indices))

        # make the split
        train_indices = indices[:train_stop]
        val_indices = indices[train_stop:val_stop]
        test_indices = indices[val_stop:]

        # save the split
        train_path = os.path.join(util.TRAIN + '.indices') # (self.data_indices_dir, util.TRAIN + '.indices')
        val_path = os.path.join(util.VAL + '.indices') # (self.data_indices_dir, util.VAL + '.indices')
        test_path = os.path.join(util.TEST + '.indices') # (self.data_indices_dir, util.TEST + '.indices')
        with open(train_path, 'wb') as file:
            pickle.dump(train_indices, file)
        with open(val_path, 'wb') as file:
            pickle.dump(val_indices, file)
        with open(test_path, 'wb') as file:
            pickle.dump(test_indices, file)

    # make more compositional
    def _create_data(self):
        if self.split == 'train':
            print("Creating vocab file ...")
            self._create_vocab()
        else:
            print("Loading vocab file ...")
            self._load_vocab()

        print("Creating dataset ...")
        data = dict()
        for i, words in tqdm(enumerate(brown.sents())):
            # filter out sequences longer than self.max_sequence_length
            if len(words) >= self.max_sequence_length:
                continue
                
            # make train/val/test split:
            if i not in self.data_indices:
                continue

            # preprocess
            input, target, length = self.process_sent(words)

            # add the datum to the dataset
            idx = len(data)
            data[idx] = dict()
            data[idx]['input'] = input
            data[idx]['target'] = target
            data[idx]['length'] = length
            
        self.num_sents = len(data)
        self.num_tokens = sum([data[key]['length'] for key in data])
        self.avg_length = self.num_tokens / self.num_sents
            
        print("Dataset created, with:")
        print("\t{} sentences".format(self.num_sents))
        print("\t{} word tokens".format(self.num_tokens))
        print("\t{} avg. sentence length".format(self.avg_length))
        print("\n")
                
        self.data = data

    # TODO make more compositional
    def _create_vocab(self):
        assert self.split == 'train', "Vocabulary can only be created for training file."

        w2c = OrderedCounter()
        i2w = dict()
        w2i = defaultdict(int)

        special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        for tok in special_tokens:
            i2w[len(i2w)] = tok
            w2i[tok] = len(w2i)

        print("\t[Getting word counts]")
        for words in tqdm(brown.sents()):
            w2c.update(words)

        print("\t[Creating dictionaries]")
        for tok, count in tqdm(w2c.items()):
            if count >= self.min_occ and tok not in special_tokens:
                i2w[len(i2w)] = tok
                w2i[tok] = len(w2i)

        assert len(w2i) == len(i2w)

        # load glove embeddings, if desiresd
        if self.embeddings is not None:
            print("\t[Loading pretrained GLOVE embeddings -- this may take a while the first time]")
            load_embeddings()
            glove_embeddings = self._get_embeddings(i2w)
        else:
            glove_embeddings = None
        
        
        # create the vocab object -- w2i, i2w, and optional embeddings
        vocab = dict(w2i=w2i, i2w=i2w, emb=glove_embeddings)
        print("Vocabulary created (%i word types)!" %len(w2i))
        
        # save the vocab object
        with open(self.vocab_file, 'wb') as vocab_file:
            pickle.dump(vocab, vocab_file)

        self._load_vocab()

# python
import os
import pickle
from collections import defaultdict

# torch
import torch
from torch.utils.data import Dataset

# torchtext
import torchtext.vocab

# numpy
import numpy as np

# nltk
from nltk.tokenize import word_tokenize as tokenize

# tqdm
from tqdm import tqdm_notebook as tqdm

# our code
import util
from util.utils import OrderedCounter
import corpus.base.Corpus as Corpus
import corpus.util.load_embeddings as load_embeddings
import corpus.util.num_lines as num_lines        
    
class Wikitext103(Corpus):
    def __init__(self, split, create_data, name='ptb', embeddings=False, **kwargs):
        super().__init__(split, create_data, embeddings, **kwargs)
        
        self.name = name
        self.raw_data_path = os.path.join(self.data_dir, util.WIKITEXT103_DIR, split + '.txt')
        self.vocab_file = name + '.vocab.pickle'
        
        print("Preprocessing Wikitext-103 *{}* data:".format(self.split))
        print("------------------------------------------")
        self._create_data()

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
        with open(self.raw_data_path, 'r') as file:
            for line in tqdm(file, total=get_num_lines(self.raw_data_path)):
                words = tokenize(line)
                
                # filter out sequences longer than self.max_sequence_length
                if len(words) >= self.max_sequence_length:
                    continue

                input, target, length = process_sent(words)

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

        with open(self.raw_data_path, 'r') as file:
            print("\t[Getting word counts]")
            for i, line in tqdm(enumerate(file), total=get_num_lines(self.raw_data_path)):
                words = tokenize(line)
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
        with open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            pickle.dump(vocab, vocab_file)

        self._load_vocab()

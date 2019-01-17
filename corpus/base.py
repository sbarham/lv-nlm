# python
import os
import pickle
import mmap
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

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_embeddings():
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    print('Loaded {} words'.format(len(glove.itos)))

class Corpus(Dataset):
    def __init__(self, split, create_data, embeddings=False, **kwargs):
        super().__init__()
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 1)
        self.embeddings = embeddings

        self.data_dir = util.DATA_DIR
        
    def __len__(self):
        return self.num_sents

    def __getitem__(self, idx):
        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }
    
    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w
    
    def process_sent(self, words):
        # prepend <sos> to inputs, postpend <eos> to targets
        input = ['<sos>'] + words
        target = words + ['<eos>']

        # sanity check
        assert len(input) == len(target), "%i, %i" % (len(input), len(target))
        length = len(input)

        # pad the sentences
        input.extend(['<pad>'] * (self.max_sequence_length - length))
        target.extend(['<pad>'] * (self.max_sequence_length - length))

        # numericalize the sentences
        input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
        target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]
        
        return input, target, length

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)

        self.w2i, self.i2w, self.glove_embeddings = vocab['w2i'], vocab['i2w'], vocab['emb']
        
    def get_embedding(word):
        return glove.vectors[glove.stoi[word]]
        
    def _get_embeddings(self, i2w):
        glove_embeddings = []
        
        print("\t[Mapping vocab to GLOVE embeddings]")
        for word in tqdm(i2w.values()):
            try:
                glove_embeddings.append(get_embedding(word).unsqueeze(0))
            except:
                glove_embeddings.append(torch.randn((1, 300)))

        glove_embeddings = torch.cat(glove_embeddings)
        
        return glove_embeddings
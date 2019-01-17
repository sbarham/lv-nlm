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
from utils import OrderedCounter

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

class PTB(Dataset):
    def __init__(self, data_dir, split, create_data, embeddings=False, **kwargs):
        super().__init__()
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 1)
        self.embeddings = embeddings

        self.data_dir = data_dir
        self.raw_data_path = os.path.join(data_dir, 'ptb.'+ split + '.txt')
        self.vocab_file = 'ptb.vocab.pickle'
        
        print("Preprocessing Penn Treebank *{}* data:".format(self.split))
        print("------------------------------------------")
        self._create_data()

    def __len__(self):
        return self.num_sents

    def __getitem__(self, idx):
        # print("idx: {}".format(idx))
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

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)

        self.w2i, self.i2w, self.glove_embeddings = vocab['w2i'], vocab['i2w'], vocab['emb']

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

# nltk
import nltk

# python
import mmap
import os
import pickle
from collections import OrderedDict

# torchtext
import torchtext

# ours
import util

from corpus.ptb import PTB
from corpus.brown import Brown
from corpus.gutenberg import Gutenberg
from corpus.kjv import Bible
from corpus.wikitext_2 import Wikitext2
from corpus.wikitext_103 import Wikitext103

def create_datasets(args, verbose=True):
    # get the path to the dataset
    dataset_name = ':'.join([
        args.corpus,
        str(args.max_sequence_length),
        str(args.min_occ)
    ]) + '.corpus'
    dataset_path = os.path.join('tmp', dataset_name)
    
    # check if dataset has been saved previously
    if os.path.isfile(dataset_path):
        with open(dataset_path, 'rb') as file:
            return pickle.load(file)
    
    # select correct corpus class
    assert args.corpus in ['ptb', 'bible', 'gutenberg', 'brown', 'wikitext-2', 'wikitext-103']
    if args.corpus == 'ptb':
        corpus_class = PTB
    elif args.corpus == 'kjv' or args.corpus == 'bible':
        nltk.download('gutenberg')
        corpus_class = Bible
    elif args.corpus == 'gutenberg':
        nltk.download('gutenberg')
        corpus_class = Gutenberg
    elif args.corpus == 'brown':
        nltk.download('brown')
        corpus_class = Brown
    elif args.corpus == 'wikitext-2':
        corpus_class = Wikitext2
    elif args.corpus == 'wikitext-103':
        corpus_class = Wikitext103
    
    # prepare for splits
    splits = [util.TRAIN, util.VAL] + ([util.TEST] if args.test else [])
    datasets = OrderedDict()
    datasets.splits = splits
    
    # create train, validation, and possibly test split
    for split in datasets.splits:
        datasets[split] = corpus_class(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ,
            embeddings=args.embeddings
        )
        
    # save the dataset for future reuse
    with open(dataset_path, 'wb') as file:
        pickle.dump(datasets, file)
    
    # return the dataset
    return datasets

def idx2word(sents, i2w, pad_idx):
    sent_str = [str()] * len(sents)

    for sent_idx, sent in enumerate(sents):
        for word_id in sent:
            try:
                word_id = word_id.item()
            except: pass
            
            if word_id == pad_idx:
                break
            
            sent_str[sent_idx] += (i2w[word_id] + " ")

        sent_str[sent_idx] = sent_str[sent_idx].strip()


    return sent_str

def count_sentences(datasets):
    data = datasets['train'] + datasets['valid'] + datasets['test']
    
    return len(data)

def count_words(datasets):
    data = datasets['train'] + datasets['valid'] + datasets['test']
    total = 0
    for sent in data:
        total = total + sent['length']
        
    return total
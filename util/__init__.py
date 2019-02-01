import torch

DATA_DIR = '/floyd/input'
PTB_DIR = '_ptb'
BROWN_DIR = '_brown'
GUTENBERG_DIR = '_gutenberg'
BIBLE_DIR = '_bible'
WIKITEXT2_DIR = '_wikitext2'
WIKITEXT103_DIR = '_wikitext103'

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

MODEL_DIR = 'bin'

FLOYD = True

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
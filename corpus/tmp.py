# python
import mmap

# torchtext
import torchtext

# ours
import util

def load_embeddings():
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    print('Loaded {} words'.format(len(glove.itos)))

def get_num_lines(file_path):
    if util.FLOYD:
        return 0
    
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
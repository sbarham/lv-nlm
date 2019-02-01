# torch
import torch
from torch.utils.data import DataLoader

# numpy
import numpy as np

# nltk
import nltk
nltk.download('punkt')

# python
import mmap
import time
import os
from multiprocessing import cpu_count
from collections import defaultdict

# torchtext
import torchtext
import torchtext.vocab

# ours
import util
from util.utils import to_var, experiment_name
from models.bowman import SentenceVAE

def get_num_lines(file_path):
    if util.FLOYD:
        return 0
    
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_embeddings():
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    print('Loaded {} words'.format(len(glove.itos)))

def create_model(args, datasets):
    model = SentenceVAE(
            vocab_size=datasets[util.TRAIN].vocab_size,
            sos_idx=datasets[util.TRAIN].sos_idx,
            eos_idx=datasets[util.TRAIN].eos_idx,
            pad_idx=datasets[util.TRAIN].pad_idx,
            unk_idx=datasets[util.TRAIN].unk_idx,
            max_sequence_length=args.max_sequence_length,
            embedding_size=args.embedding_size,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            word_dropout=args.word_dropout,
            embedding_dropout=args.embedding_dropout,
            latent_size=args.latent_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional
        )
    if util.USE_CUDA:
        print("sending model to cuda")
        model = model.cuda()
        
    return model

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1.0 / (1.0 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1.0, step / x0)
    elif anneal_function == 'const':
        return 1.0
    
def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0, pad_idx):
    NLL = torch.nn.NLLLoss(reduction='sum', ignore_index=pad_idx)
    
    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))
        
    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

def train(model, datasets, args, verbose=True):
    print(model)
    
    print("\n\n\n-------------------------------------------")
    print("TRAINING NEW MODEL:")
    print("\t- using corpus:\t" + args.corpus)
    print("\t- max seq len:\t" + str(args.max_sequence_length))
    print("\t- hidden size:\t" + str(args.hidden_size))
    print("\t- latent size:\t" + str(args.latent_size))
    print("\t- bidirectional:\t" + str(args.bidirectional))
    print()
    
    timestamp = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    model_name = experiment_name(args, timestamp)
    print("Beginning training at: {}".format(timestamp))
    print("-------------------------------------------")

    # create the directory for saving this model
    args.save_model_path = os.path.join(util.MODEL_DIR, timestamp)
    os.makedirs(args.save_model_path)
    
    # create the optimizer, the tracker, and initialize the step to 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trackers = {split: defaultdict(list) for split in datasets.splits}
    trackers.update({(split + '_mean'): defaultdict(list) for split in datasets.splits})
    step = 0
    num_batches = 0
    best = float('inf')
    
    # get the pad index, for convenience
    pad_idx = datasets['train'].get_w2i()['<pad>']
    # go!
    for epoch in range(args.epochs):
        print("\n")
        for split in datasets.splits:
            num_batches = 0
            print(" EPOCH {}, SPLIT = {}".format((epoch + 1), split))
            print("-------------------------------------------")
            
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):
                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['length'])

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0, pad_idx)

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        % (split.upper(),
                           iteration,
                           len(data_loader) - 1,
                           loss.item(),
                           NLL_loss.item() / batch_size,
                           KL_loss.item() / batch_size,
                           KL_weight))

                trackers[split]['ELBO'].append(loss.item())
                trackers[split]['NLL'].append(NLL_loss.item() / batch_size)
                trackers[split]['KLD'].append(KL_loss.item() / batch_size)
                trackers[split]['KL_weight'].append(KL_weight)
                
                num_batches += 1

            
            """
            END OF SPLIT
            """
            print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (
                split.upper(),
                (epoch + 1),
                args.epochs,
                np.mean(trackers[split]['ELBO'])
            ))
            
            trackers[split + '_mean']['ELBO'].append(np.mean(trackers[split]['ELBO'][-num_batches:]))   
            trackers[split + '_mean']['KLD'].append(np.mean(trackers[split]['KLD'][-num_batches:]))
            trackers[split + '_mean']['NLL'].append(np.mean(trackers[split]['NLL'][-num_batches:]))

            # save checkpoint
            if split == 'train':                
                # save checkpoint
                checkpoint_path = os.path.join(args.save_model_path, "E%i.pytorch" % (epoch + 1))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)
                
                # check if best checkpoint so far
                if trackers[split + '_mean']['ELBO'][-1] < best:
                    best = trackers[split + '_mean']['ELBO'][-1]
                    args.best_epoch = epoch
                    args.load_checkpoint = 'E{}.pytorch'.format(epoch + 1)
                
    print("-------------------------------------------")
    print("\nTraining Complete")
                
    return trackers, model
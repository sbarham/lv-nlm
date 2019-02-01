# python
import os
import re
import random

# torch
import torch

# numpy
import numpy as np

# ours
import util
from util.utils import to_var
from models.utils import create_model as create_model
from corpus.utils import idx2word as idx2word

def test(args, datasets):
    # get the vocabulary
    w2i, i2w = datasets['train'].get_w2i(), datasets['train'].get_i2w()

    # create the model obejct to specification
    model = create_model(args=args, datasets=datasets)

    # find the correct (indicated) checkpoint to use for the weights
    checkpoint_path = os.path.join(args.save_model_path, args.load_checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)  
    
    # load the weights
    if util.USE_CUDA:
        print("loading cuda state")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("loading non-cuda state")
        model.load_state_dict(torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage
        ))
    
    # report success
    print("Model loaded from %s" % (checkpoint_path))
    
    model.eval()

    fname = os.path.join(args.save_model_path, 'samples')
    res = ''
    with open(fname, 'w+') as file:
        # generate random samples
        res += random_samples(model, args, i2w, w2i)
        
        # generate cold interpolations (i.e., not starting from encoded
        # corpus sentences as endpoints)
        res += cold_interpolation(model, args, i2w, w2i)

        # generate warm interpolations (i.e., using encoded corpus sentences
        # as endpoints)
        res += warm_interpolation(model, args, datasets, i2w, w2i)
        
        # test sentence reconstruction
        res += reconstruction(model, args, datasets, i2w, w2i)
        
        print("wrote samples to '{}'".format(fname))
        file.write(res)
        
    return res

def test_cold(path):
    # extract the required model hyperparameters/arguments from the
    # checkpoint's enclosing directory
    with open(path + '/args', 'r') as file:
        args = args_to_dict(file)

    # report the arguments used to build the model, for debugging purposes
    print(vars(args))    
    
    # build the dataset and get the vocabulary
    datasets = create_datasets(args)
    w2i, i2w = datasets['train'].get_w2i(), datasets['train'].get_i2w()

    # create the model obejct to specification
    model = create_model(args=args, datasets=datasets)

    # find the correct (indicated) checkpoint to use for the weights
    checkpoint_path = os.path.join(path, args.load_checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)  
    
    # load the weights
    if util.USE_CUDA:
        print("loading cuda state")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("loading non-cuda state")
        model.load_state_dict(torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage
        ))
    
    # report success
    print("Model loaded from %s" % (checkpoint_path))
    
    model.eval()

    fname = os.path.join(path, 'samples')
    res = ''
    with open(fname, 'w+') as file:
        # generate random samples
        res += random_samples(model, args, i2w, w2i)
        
        # generate cold interpolations (i.e., not starting from encoded
        # corpus sentences as endpoints)
        res += cold_interpolation(model, args, i2w, w2i)

        # generate warm interpolations (i.e., using encoded corpus sentences
        # as endpoints)
        res += warm_interpolation(model, args, datasets, i2w, w2i)
        
        # test sentence reconstruction
        res += reconstructions(model, args, datasets, i2w, w2i)
        
        print("wrote samples to '{}'".format(fname))
        file.write(res)
        
    return res

def interpolate(start, end, steps):
    steps = steps + 2
    
    interpolation = np.zeros((start.shape[0], steps))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps)

    return interpolation.T

def random_samples(model, args, i2w, w2i):
    res = ''
    
    # generate random samples
    samples, z = model.inference(n=args.num_samples)
    
    # create result string
    res += '----------SAMPLES----------'
    for line in idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']):
        line = ''.join(line)
        line = clean_sample(line)
        res += '\n' + line
    
    res += '\n\n'
    
    return res

def cold_interpolation(model, args, i2w, w2i, device=util.DEVICE):
    res = ''
    
    # generate latent code endpoints
    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    
    # create the interpolations
    z = to_var(torch.from_numpy(interpolate(
        start=z1,
        end=z2,
        steps=(args.num_steps - 2)
    ))).float()
    
    # sample sentences from each of the interpolations
    samples, _ = model.inference(z=z)
    
    # create result string
    res += '------- COLD INTERPOLATION --------'
    for line in idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']):
        line = ''.join(line)
        line = clean_sample(line)
        res += '\n' + line
    
    res += '\n\n'
    
    return res

def warm_interpolation(model, args, datasets, i2w, w2i, device=util.DEVICE):
    res = ''
    
    # pick two random sentences
    i = random.randint(0, len(datasets['test']))
    j = random.randint(0, len(datasets['test']))

    # convert the sentences to tensors
    s_i = torch.tensor([datasets['test'][i]['input']], device=device)
    s_i_length = torch.tensor([datasets['test'][i]['length']], device=util.DEVICE)
    s_j = torch.tensor([datasets['test'][j]['input']], device=device)
    s_j_length = torch.tensor([datasets['test'][j]['length']], device=util.DEVICE)

    # encode the two sentences into latent space
    with torch.no_grad():
        _, _, _, z_i = model(s_i, s_i_length)
        _, _, _, z_j = model(s_j, s_j_length)
        z_i, z_j = z_i.cpu(), z_j.cpu()
            
    # create the interpolation
    z1, z2 = z_i.squeeze().numpy(), z_j.squeeze().numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    
    # generate samples from each code point
    samples, _ = model.inference(z=z)
    
    # create the result string
    res += '------- WARM INTERPOLATION --------\n'
    
    res += '(Original 1): '
    line = ''.join(idx2word(
        s_i,
        i2w=i2w,
        pad_idx=w2i['<pad>']
    ))
    line = clean_sample(line)
    res += line + '\n'
    
    for line in idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']):
        line = ''.join(line)
        line = clean_sample(line)
        res += line + '\n'
    
    res += '(Original 2): '
    line = ''.join(idx2word(
        s_j,
        i2w=i2w,
        pad_idx=w2i['<pad>']
    ))
    line = clean_sample(line)
    res += line + '\n\n'
        
    return res

def reconstruction(
    model,
    args,
    datasets,
    i2w,
    w2i,
    device=util.DEVICE
):
    res = ''
    sentences = []
    
    for _ in range(args.num_reconstructions):    
        # pick short sentences
        s_short, s_short_len = get_random_short(datasets, args)
        sentences.append({
            'original': s_short,
            'original_length': s_short_len,
            'type': 'short'
        })
        
    for _ in range(args.num_reconstructions):
        # pick random sentences
        s_rand, s_rand_len = get_random(datasets, args)
        sentences.append({
            'original': s_rand,
            'original_length': s_rand_len,
            'type': 'random'
        })
    
    # pick args.num_reconstructions sentences of each type (short, long, random)
    for _ in range(args.num_reconstructions):    
        # pick long sentences
        s_long, s_long_len = get_random_long(datasets, args)
        sentences.append({
            'original': s_long,
            'original_length': s_long_len,
            'type': 'long'
        })
        
    # reconstruct each sentence
    for i, sentence in enumerate(sentences):
        # get the mean and logvar of this each encoded sentence
        with torch.no_grad():
            _, mean, log_v, _ = model(
                sentence['original'],
                sentence['original_length']
            )
        
        mean, log_v = mean.cpu(), log_v.cpu()
        
        if not args.bidirectional:
            mean = mean.unsqueeze(0)
            log_v = log_v.unsqueeze(0)
        
        stdev = torch.exp(0.5 * log_v)
        
        # decode the mean sentence
        mean_sentence = model.inference(z=mean)[0] # don't save z-value
        # save the mean sentence
        sentence['mean'] = mean
        sentence['mean_sentence'] = mean_sentence
        
        # decode a number of sentences sampled from the latent code
        # (1) sample a latent code
        z = torch.randn([3, args.latent_size])
        z = z * stdev + mean
            
        # (2) decode the latent code
        random_sentences, zs = model.inference(z=z)
        
        # save the random sentences with their z values
        sentence['random_samples'] = []
        for i in range(random_sentences.size(0)):
            sentence['random_samples'].append((
                random_sentences[i].cpu(),
                zs[i].cpu()
            ))
    
    # create the result string
    res += '------- RECONSTRUCTION --------'
    for i, sentence in enumerate(sentences):
        ###############################
        # PRINT THE ORIGINAL SENTENCE #
        ###############################
        res += '\nSentence [{}] ({})\n'.format(i, sentence['type'])
        res += '(Original):\n\t'
        line = ''.join(idx2word(
            sentence['original'],
            i2w=i2w,
            pad_idx=w2i['<pad>']
        ))
        line = clean_sample(line)
        res += line + '\n'
        
        ###################################
        # PRINT THE MEAN DECODED SENTENCE #
        ###################################
        res += '(Mean):\n\t'
        line = ''.join(idx2word(
            sentence['mean_sentence'],
            i2w=i2w,
            pad_idx=w2i['<pad>']
        ))
        line = clean_sample(line)
        res += line + '\n'
        
        ########################################
        # PRINT THE RANDOMLY SAMPLED SENTENCES #
        ########################################
        res += '(Random Samples):\n'
        for sent, z in sentence['random_samples']:
            # get the distance from the mean and print it in parens
            mean = sentence['mean']
            res += '({:.3f} from mean)\n\t'.format(np.linalg.norm(mean - z))
            
            # convert the randomly sampled sentence to a string
            line = ''.join(idx2word(
                sent.unsqueeze(0),
                i2w=i2w,
                pad_idx=w2i['<pad>']
            ))
            
            # clean the string up and add it to the result
            line = clean_sample(line)
            res += line + '\n'
            
        res += '\n'
        
    res += '\n\n'
        
    return res

def get_random(datasets, args, device=util.DEVICE):
    s_rand_idx = random.randint(0, len(datasets['test']))    
    s_rand = torch.tensor([datasets['test'][s_rand_idx]['input']], device=device)
    s_rand_len = torch.tensor([datasets['test'][s_rand_idx]['length']], device=device)
    
    # print("in get_random: found sent with length={}".format(s_rand_len))
    
    return s_rand, s_rand_len

def get_random_short(datasets, args, device=util.DEVICE):
    # warm up -- get a stochastic average of length, and a stochastic max
    running_length = 0
    running_min = 0
    for i in range(args.sample_warmup_period):
        s_rand_idx = random.randint(0, len(datasets['test']))
        s_rand_length = datasets['test'][s_rand_idx]['length']
        running_length += s_rand_length
        if s_rand_length < running_min:
            running_min = s_rand_length
    
    # we want only sentences whose length is roughly in the fourth quartile
    avg_length = running_length / args.sample_warmup_period
    max_length = (avg_length + running_min) / 2
    
    # find an appropriate sentence
    while True:
        s_rand_idx = random.randint(0, len(datasets['test']))
        s_rand_length = datasets['test'][s_rand_idx]['length']
        
        if s_rand_length < max_length:
            s_short = torch.tensor([datasets['test'][s_rand_idx]['input']], device=device)
            s_short_len = torch.tensor([s_rand_length], device=device)
            return s_short, s_short_len
        
def get_random_long(datasets, args, device=util.DEVICE):
    # warm up -- get a stochastic average of length, and a stochastic max
    running_length = 0
    running_max = 0
    for i in range(args.sample_warmup_period):
        s_rand_idx = random.randint(0, len(datasets['test']))
        s_rand_length = datasets['test'][s_rand_idx]['length']
        running_length += s_rand_length
        if s_rand_length > running_max:
            running_max = s_rand_length
    
    # we want only sentences whose length is roughly in the fourth quartile
    avg_length = running_length / args.sample_warmup_period
    min_length = (avg_length + running_max) / 2
    
    # find an appropriate sentence
    while True:
        s_rand_idx = random.randint(0, len(datasets['test']))
        s_rand_length = datasets['test'][s_rand_idx]['length']
        
        if s_rand_length > min_length:
            s_long = torch.tensor([datasets['test'][s_rand_idx]['input']], device=device)
            s_long_len = torch.tensor([s_rand_length], device=device)
            return s_long, s_long_len
        
def clean_sample(line):
    # left and right strip the line
    line = line.strip()
    
    # remove leading or trailing reserved symbol
    if line.startswith('<sos>'):
        line = line[5:]
    if line.endswith('<eos>'):
        line = line[:-5]
        
    # again left and right strip the line
    line = line.strip()
    
    # fix the punctuation
    line = clean_punctuation(line)
    
    return line

def clean_punctuation(line):
    # remove space around colons between numbers
    num_colon = re.compile(r'(\d+)\s+:\s+(\d+)')
    line = num_colon.sub(r"\1:\2", line)
    
    # remove space before commas, colons, and periods
    line = re.sub(r"\s+(,|:|\.)\s+", r"\1 ", line)
    
    # remove space around apostrophes
    line = re.sub(r"\s+'\s+", r"'", line)
    
    # remove space before final periods
    line = re.sub(r"\s+\.", r".", line)
    
    return line
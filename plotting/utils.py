# python
import math
import os

# pyplot
import matplotlib.pyplot as plt

def exponential_smoothing(ys, beta=0.8, ub=math.inf, lb=-math.inf):
    """
    This is ugly, and I should have used a comprehension, but
    it'll get the job done. I made it a function because I suspect
    I may need it later.
    """
    smooth_ys = [ys[0]]
    for y in ys:
        if y > ub or y < lb:
            smooth_ys.append(smooth_ys[-1])
        else:
            smooth_ys.append(beta * smooth_ys[-1] + (1 - beta) * y)
    return smooth_ys[1:]

def plot(ELBO, NLL, KL, title, fname=None, xlabel="Epochs", ylabel="Measurements", hline=None, epochs=None):
    """
    Just a *slight* abstraction over pyplot to ease development a bit.
    """
    xs = list(range(len(ELBO)))
    if epochs is not None:
        xs = [x / len(xs) * epochs for x in xs]
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if hline:
        plt.axhline(y=hline, color='r', linestyle='-')
    
    plt.plot(xs, ELBO, label="ELBO")
    plt.plot(xs, NLL, label="NLL Loss", c='blue')
    plt.plot(xs, KL, label="KL Loss", c='red')
    plt.legend()
    
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
        
    plt.clf()
    
def plot_elbo(ELBO, fname=None, title='ELBO', xlabel="Epochs", ylabel="ELBO", hline=None, epochs=None):
    """
    Just a *slight* abstraction over pyplot to ease development a bit.
    """
    xs = list(range(len(ELBO)))
    if epochs is not None:
        xs = [x / len(xs) * epochs for x in xs]
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(xs, ELBO, label="ELBO")
    plt.legend()
        
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
        
    plt.clf()
    
def graph(trackers, datasets, args):
    for split in datasets.splits:
        # use the full, batch-by-batch metrics, not the epoch-averages
        if split.endswith('_mean'):
            continue
        
        fname = '{}_perf:emb{}-z{}-lstm{}-maxlen{}'.format(
            split,
            args.embedding_size,
            args.latent_size,
            args.hidden_size,
            args.max_sequence_length
        )
        
        fname = os.path.join(args.save_model_path, fname)
        
        plot(
            fname=fname,
            ELBO=exponential_smoothing(trackers[split]['ELBO']),
            KL=exponential_smoothing(trackers[split]['KLD']),
            NLL=exponential_smoothing(trackers[split]['NLL']),
            title='S-VAE *{}* Performance\n(Mikolov\'s Simplified PTB, max length={})'.format(
                split,
                args.max_sequence_length
            ),
            epochs=args.epochs
        )
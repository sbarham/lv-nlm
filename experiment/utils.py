# python
import os
import pickle

class Args: pass

def save_args(args):
    fname = os.path.join(args.save_model_path, 'args')
    with open(fname, 'w+') as file:
        lines = ['{}: {}\n'.format(key, val) for key, val in vars(args).items()]
        file.writelines(lines)
        
def args_to_dict(file):
    args = Args()
    for line in file:
        # print(line)
        line = line.split(":")
        if len(line) > 2:
            continue
        key, val = line[0], line[1]
        key = key.strip()
        val = val.strip()
        setattr(args, key, convert(val))
        
    return args
        
def save_model_printout(args, model):
    fname = os.path.join(args.save_model_path, 'model-printout')
    with open(fname, 'w+') as file:
        file.write(str(model))
        
def save_trackers(trackers, args):
    fname = os.path.join(args.save_model_path, 'trackers')
    with open(fname, 'wb') as file:
        pickle.dump(trackers, file)
        
    fname = os.path.join(args.save_model_path, 'metrics-pretty-print')
    with open(fname, 'w+') as file:
        file.write(pretty_print_trackers(trackers, args))
        
def pretty_print_trackers(trackers, args):
    # variable for the string we'll accumulate
    res = ''
    
    # only pretty-print the mean trackers -- i.e., we want epochal means, not
    # per-batch statistics (that's way too much information)
    mean_trackers = filter(lambda item: item[0].endswith('_mean'), trackers.items())
    
    # pretty-print means for each split, and for each metric in each split
    for (split, split_dict) in mean_trackers:
        split = split[:-5]
        res += ('*' * 35)
        res += '  {}  '.format(split.upper())
        res += ('*' * 35)
        res += '\n'
        
        for metric, metric_list in split_dict.items():
            res += "{} {}:\n{}\n\n".format(
                split.capitalize(),
                metric,
                list_to_str(metric_list, args.best_epoch - 1)
            )
        
        res = res.rstrip()
        res += '\n\n'
        res += ('*' * 79)
        res += '\n\n\n\n'
            
    return res

def convert(s):
    if s.isdigit():
        return int(s)
    elif s == 'True':
        return True
    elif s == 'False':
        return False
    try:
        float(s)
        return float(s)
    except ValueError:
        return s
    
def list_to_str(l, best_idx):
    scale = ['']
    sep = ['']
    res = ['']
    
    # add each element, creating a new line when necessary
    for i, item in enumerate(l):
        if len(res[-1].expandtabs()) > 80:
            scale[-1] = scale[-1].strip()
            res[-1] = res[-1].strip()
            sep[-1] += ('-' * (len(res[-1].expandtabs()) + 3))
            
            scale += ['']
            sep += ['']
            res += ['']
        
        # add the element itself, with its epoch number
        if i == best_idx:
            res[-1] += '**'
            scale[-1] += '**'
        
        res[-1] += '{:.2f}'.format(item)
        scale[-1] += str(i + 1)
        
        if i == best_idx:
            res[-1] += '**'
            scale[-1] += '**'
        
        res[-1] += '\t'
        len_dif = len(res[-1].expandtabs()) - len(scale[-1].expandtabs())
        scale[-1] += ' ' * len_dif
    
    # add final divider and strip final tabs
    scale[-1] = scale[-1].strip()
    res[-1] = res[-1].strip()
    sep[-1] += ('-' * (len(res[-1].expandtabs()) + 3))
    
    # join the lines together into a single string
    final = ''
    for i in range(len(res)):
        final += scale[i] + '\n' + sep[i] + '\n' + res[i] + '\n'
        
    return final
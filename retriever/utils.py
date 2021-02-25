import os
import sys
import numpy as np

import torch


def get_iter_batch(data_iter, data_loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)

    return batch, data_iter

    
def filter_cand_paths(candidates):
    '''
    [('22-rdf-syntax-ns#type', ''), ('notable types', 'm.02kl65p'), ('has value', 'm.0c134hr')]
    Remove duplicate paths, merge answers
    '''
    candidates = [x for x in candidates if x[1] != '']
    paths = list(set(x[0] for x in candidates))
    path_aw_dict = dict()
    for cand in candidates:
        path_aw_dict.setdefault(cand[0], [])
        path_aw_dict[cand[0]].append(cand[1])

    new_cands = [(p, path_aw_dict[p]) for p in paths]

    return paths, new_cands


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def save(model, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,  'saved_pytorch_model_{}.pt'.format(save_prefix))
    print('Save the model at {}'.format(save_path))
    torch.save(model.state_dict(), save_path)
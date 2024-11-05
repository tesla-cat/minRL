import numpy as np
import torch as tc

# general


def start_workers(seed=0):
    # setup_torch
    # seed += 1000 * proc_id
    tc.manual_seed(seed)
    np.random.seed(seed)


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def avg(x):
    return x


# torch


def avg_grads(params):
    pass


def sync_params(params):
    pass

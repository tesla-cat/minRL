import numpy as np

# general


def start_workers():
    pass


def proc_id():
    return 0


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


# torch


def setup_torch():
    pass


def avg_grads(params):
    pass


def sync_params(params):
    pass

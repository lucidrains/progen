import os, errno
from shutil import rmtree

from jax import random, nn, value_and_grad, vmap, pmap, jit, lax
from jax.lax import top_k
import jax.numpy as np

# helper functions

def exists(val):
    return val is not None

def log(t, eps = 1e-20):
    return np.log(t + eps)

def confirm(question):
    while True:
        resp = input(f'{question} (y/n) ')
        lower_resp = resp.lower()
        if lower_resp in ('y', 'n'):
            return lower_resp == 'y'

def clear_directory_(path):
    rmtree(str(path), ignore_errors = True)
    path.mkdir(exist_ok = True, parents = True)

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

# training functions

def cross_entropy(logits, targets, axis = -1, ignore_index = 0):
    logprobs = nn.log_softmax(logits, axis = axis)
    nll = np.take_along_axis(logprobs, np.expand_dims(targets, axis = axis), axis = axis)

    # mask for loss is engineered so that it learns from the first padding token
    # the padding token is reused as end-of-string for simplicity

    mask = (targets != ignore_index)
    eos_mask = (~mask).cumsum(axis = -1) == 1
    mask = mask | eos_mask

    ce = -np.mean(nll[mask])
    return ce

def get_train_loss_fn(model, data_parallel = False):
    map_fn = pmap if data_parallel else vmap
    batch_model_apply = jit(map_fn(model.apply, in_axes = (None, None, 0), out_axes = 0))

    @value_and_grad
    def loss_fn(params, key, data):
        inp, labels = data[:, :-1], data[:, 1:]
        logits = batch_model_apply(params, key, inp)
        return cross_entropy(logits, labels, axis = -1)

    return loss_fn

# sampling functions

def select_top_k(tensor, k):
    values, _ = top_k(tensor, k)
    mask = tensor > values.min()
    return mask, np.where(mask, tensor, 0.)

def gumbel_noise(rng, shape):
    noise = random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
    return -log(-log(noise))

def sample(rng, fn, params, prime, length, top_k = None):
    start_pos = prime.shape[-1]
    seq = np.pad(prime, (0, length - prime.shape[-1]))
    one_hots = np.eye(length, dtype = int)

    for curr_pos in range(start_pos, length):
        logits = fn(params, next(rng), seq)
        logits = logits[curr_pos - 1]

        noise = gumbel_noise(next(rng), logits.shape)

        if exists(top_k):
            mask, logits = select_top_k(logits, top_k)
            noise *= mask

        logits += noise
        sampled_ind = np.argmax(logits, axis = -1)

        one_hot = one_hots[curr_pos]
        seq += one_hot * sampled_ind

    # for now, just set everything after second padding token (eos) to padding
    remove_after_eos_mask = (seq == 0).cumsum(axis = -1) > 1
    seq *= ~remove_after_eos_mask

    return seq

# rng hacks

def hardware_uniform(
    rng_key,
    shape,
    dtype = np.float32,
    minval = np.float32(0),
    maxval = np.float32(1)
):
    del rng_key
    minval = lax.convert_element_type(minval, dtype)
    maxval = lax.convert_element_type(maxval, dtype)
    return lax.rng_uniform(minval, maxval, shape)

def hardware_bernoulli(rng_key, p = np.float32(0.5), shape = None):
    del rng_key
    return lax.rng_uniform(0.0, 1.0, shape) < p

def set_hardware_rng_(jax):
    jax.random.bernoulli = hardware_bernoulli
    jax.random.uniform = hardware_uniform
    jax._src.random.uniform = hardware_uniform

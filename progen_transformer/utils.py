from math import ceil
import os, errno
from shutil import rmtree

import jax
from jax import random, nn, value_and_grad, vmap, pmap, jit, lax
from jax.lax import top_k
import jax.numpy as np

from einops import rearrange

# helper functions

def noop(x):
    return x

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

def masked_mean(t, mask, axis = None):
    return (t * mask).sum(axis = axis) / mask.sum(axis = axis)

def cross_entropy(logits, targets, axis = -1, ignore_index = 0):
    logprobs = nn.log_softmax(logits, axis = axis)

    nll = np.take_along_axis(logprobs, np.expand_dims(targets, axis = axis), axis = axis)
    nll = nll.squeeze(-1)

    # mask for loss is engineered so that it learns from the first padding token
    # the padding token is reused as end-of-string for simplicity

    mask = (targets != ignore_index)
    eos_mask = (~mask).cumsum(axis = -1) == 1
    mask = mask | eos_mask

    ce = -masked_mean(nll, mask, axis = -1)
    return ce

def get_loss_fn(model, data_parallel = False):
    def loss_fn(params, key, data):
        ids, labels = data[:-1], data[1:]
        logits = model.apply(params, key, ids)
        return cross_entropy(logits, labels, axis = -1)

    loss_fn = jit(vmap(loss_fn, in_axes = (None, None, 0), out_axes = 0))

    if data_parallel:
        loss_fn = pmap(loss_fn, in_axes = (None, None, 0), out_axes = 0)

    @value_and_grad
    def batched_loss_fn(params, key, data):
        if not data_parallel:
            values = loss_fn(params, key, data)
            return np.mean(values)

        mask = np.ones((data.shape[0],))

        device_count = jax.local_device_count()
        batch_size = data.shape[0]

        remainder = (batch_size % device_count)
        if remainder != 0:
            padding = device_count - remainder
            data = np.pad(data, ((0, padding), (0, 0)))
            mask = np.pad(mask, ((0, padding)))

        data, mask = map(lambda t: rearrange(t, '(p b) ... -> p b ...', p = device_count), (data, mask))
        values = loss_fn(params, key, data)
        return masked_mean(values, mask)

    return batched_loss_fn

# sampling functions

def select_top_k(tensor, k):
    values, _ = top_k(tensor, k)
    mask = tensor > values.min()
    return mask, np.where(mask, tensor, 0.)

def gumbel_noise(rng, shape):
    noise = random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
    return -log(-log(noise))

def sample(rng, fn, params, prime, length, top_k = None, add_bos = False):
    start_pos = prime.shape[-1]
    pad_right = length - prime.shape[-1]

    padding = (0, pad_right) if not add_bos else (1, pad_right - 1)
    seq = np.pad(prime, padding)

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

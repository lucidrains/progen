from functools import partial

import jax
from jax import random
from jax import nn
import jax.numpy as np

import haiku as hk
from haiku import initializers
from einops import rearrange

from progen.utils import exists

# constants

EPS = 1e-3
ATTN_MASK_VALUE = -1e10

# helpers

LayerNorm = partial(hk.LayerNorm, create_scale = True, create_offset = False, axis = -1)

# classes

class Attention(hk.Module):
    def __init__(
        self,
        *,
        dim_out,
        dim_head
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.to_qkv = hk.Linear(dim_head * 3)
        self.to_out = hk.Linear(dim_out)

    def __call__(self, x):
        n = x.shape[0]

        qkv = self.to_qkv(x)
        q, k, v = np.split(qkv, 3, axis = -1)
        sim = np.einsum('i d, j d -> i j', q, k) * self.scale

        mask = np.triu(np.ones((n, n), dtype = bool), 1)
        sim = np.where(mask, ATTN_MASK_VALUE, sim)

        attn = nn.softmax(sim, axis = -1)
        out = np.einsum('i j, j d -> i d', attn, v)
        return self.to_out(out)

class SGU(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        seq_len
    ):
        super().__init__()
        self.seq_len = seq_len
        self.norm = LayerNorm()
        self.proj_out = hk.Linear(dim_out)

    def __call__(self, x, gate_res = None):
        n = self.seq_len
        x, gate = np.split(x, 2, axis = -1)

        gate = self.norm(gate)

        init_scale = EPS / n
        init_eps = initializers.RandomUniform(minval = -init_scale, maxval = init_scale)

        weights = hk.get_parameter('spatial_weights', shape = (n, n), init = init_eps)
        biases = hk.get_parameter('spatial_biases', shape = (n, 1), init = np.ones)

        mask = np.tril(np.ones((n, n)))
        weights = weights * mask

        gate = np.einsum('n d, m n -> m d', gate, weights)
        gate += biases

        if exists(gate_res):
            gate += gate_res

        x = x * gate
        return self.proj_out(x)

class gMLP(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        name,
        attn_dim = None
    ):
        super().__init__(name = name)
        self.attn = Attention(dim_head = attn_dim, dim_out = dim_ff // 2) if exists(attn_dim) else None
        self.norm = LayerNorm()
        self.proj_in = hk.Linear(dim_ff)
        self.sgu = SGU(dim = dim_ff, dim_out = dim_ff // 2, seq_len = seq_len)
        self.proj_out = hk.Linear(dim)

    def __call__(self, x):
        x = self.norm(x)
        gate_res = self.attn(x) if exists(self.attn) else None

        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.sgu(x, gate_res)
        x = self.proj_out(x)
        return x

class ProGenBase(hk.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        depth,
        heads = 1,
        ff_mult = 4,
        attn_dim = None,
        clamp_gate = True,
        layer_survival_prob = 1.
    ):
        super().__init__()
        self.embed = hk.Embed(num_tokens, dim)
        self.layers = [gMLP(dim = dim, dim_ff = dim * ff_mult, seq_len = seq_len, name = f'gmlp{i}', attn_dim = attn_dim) for i in range(depth)]

        self.to_logits = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_tokens)
        ])

    def __call__(self, x):
        x = self.embed(x)

        for layer in self.layers:
            x += layer(x)

        return self.to_logits(x)

def ProGen(**kwargs):
    @hk.transform
    def inner(seq):
        return ProGenBase(**kwargs)(seq)
    return inner

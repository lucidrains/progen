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

class LocalAttention(hk.Module):
    def __init__(
        self,
        *,
        name,
        dim,
        window_size,
        heads = 8,
        dim_head = 64
    ):
        super().__init__(name = name)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.norm = LayerNorm()
        self.to_qkv = hk.Linear(inner_dim * 3, with_bias = False)
        self.to_out = hk.Linear(dim)

    def __call__(self, x):
        x = self.norm(x)

        n, h, wsz = x.shape[0], self.heads, self.window_size
        assert (n % wsz) == 0, 'sequence length must be divisible by the window size'
        window = n // wsz

        qkv = self.to_qkv(x)
        q, k, v = np.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, '(w n) (h d) -> h w n d', w = window, h = h), (q, k, v))

        k, v = map(lambda t: np.pad(t, ((0, 0), (1, 0), (0, 0), (0, 0)), constant_values = 0.), (k ,v))
        k, v = map(lambda t: np.concatenate((t[:, :-1], t[:, 1:]), axis = 2), (k, v))

        sim = np.einsum('h w i d, h w j d -> h w i j', q, k) * self.scale

        mask = np.tril(np.ones((wsz, wsz * 2)), wsz)
        sim = np.where(mask, sim, ATTN_MASK_VALUE)

        attn = nn.softmax(sim, axis = -1)
        out = np.einsum('h w i j, h w j d -> h w i d', attn, v)
        out = rearrange(out, 'h w n d -> (w n) (h d)')
        return self.to_out(out)

class FeedForward(hk.Module):
    def __init__(
        self,
        *,
        name,
        dim,
        ff_mult = 4
    ):
        super().__init__(name = name)
        self.norm = LayerNorm()
        self.proj_in = hk.Linear(dim * ff_mult)
        self.proj_out = hk.Linear(dim)

    def __call__(self, x):
        x = self.norm(x)
        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.proj_out(x)
        return x

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

    def __call__(self, x):
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
        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.sgu(x)
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
        window_size = 512,
        global_mlp_depth = 2,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dim = None,
        clamp_gate = True
    ):
        super().__init__()
        self.embed = hk.Embed(num_tokens, dim)

        self.layers = []
        for i in range(depth):
            self.layers.extend([
                LocalAttention(name = f'attn{i}', dim = dim, window_size = window_size, heads = heads, dim_head = dim_head),
                FeedForward(name = f'ff{i}', dim = dim, ff_mult = ff_mult)
            ])

        self.layers.extend([gMLP(dim = dim, dim_ff = dim * ff_mult, seq_len = seq_len, name = f'gmlp{i}', attn_dim = attn_dim) for i in range(global_mlp_depth)])

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

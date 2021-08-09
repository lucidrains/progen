from functools import partial

import jax
from jax import random
from jax import nn
import jax.numpy as np
import jmp

import haiku as hk
from haiku import initializers
from einops import rearrange, repeat

from progen_transformer.utils import exists

# constants

ATTN_MASK_VALUE = -1e10

# helpers

LayerNorm = partial(hk.LayerNorm, create_scale = True, create_offset = False, axis = -1)

def fixed_pos_embedding(seq, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, "b n -> b (n r)", r = 2)[None, :, :]
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

def rotate_every_two(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = np.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    rot_dim = sin.shape[-1]
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x = (x * cos) + (rotate_every_two(x) * sin)
    return np.concatenate((x, x_pass), axis = -1)

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

    def __call__(self, x, *, pos_emb):
        x = self.norm(x)

        n, h, wsz = x.shape[0], self.heads, self.window_size
        assert (n % wsz) == 0, 'sequence length must be divisible by the window size'
        window = n // wsz

        qkv = self.to_qkv(x)
        q, k, v = np.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h = h), (q, k, v))

        q, k, v = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'h (w n) d -> h w n d', w = window), (q, k, v))

        k, v = map(lambda t: np.pad(t, ((0, 0), (1, 0), (0, 0), (0, 0)), constant_values = 0.), (k ,v))
        k, v = map(lambda t: np.concatenate((t[:, :-1], t[:, 1:]), axis = 2), (k, v))

        sim = np.einsum('h w i d, h w j d -> h w i j', q, k) * self.scale

        mask = np.tril(np.ones((wsz, wsz * 2)), wsz)
        sim = np.where(mask, sim, ATTN_MASK_VALUE)

        sim = sim - np.amax(sim, axis = -1, keepdims = True)
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
        ff_mult = 4,
        glu = False,
        seq_len = None,
        spatial_gate = False
    ):
        super().__init__(name = name)
        assert not (glu and spatial_gate), 'glu and sgu cannot be turned on at the same time'
        hidden_dim = dim * ff_mult
        hidden_dim *= (1 if not glu else 2)

        self.norm = LayerNorm()
        self.proj_in = hk.Linear(hidden_dim)
        self.proj_out = hk.Linear(dim)

        self.glu = glu
        self.sgu = SGU(dim = hidden_dim, dim_out = hidden_dim // 2, seq_len = seq_len) if spatial_gate else None

    def __call__(self, x):
        x = self.norm(x)
        x = self.proj_in(x)

        if self.glu:
            x, gate = np.split(x, 2, axis = -1)
            x *= nn.gelu(gate)
        else:
            x = nn.gelu(x)

        if exists(self.sgu):
            x = self.sgu(x)

        x = self.proj_out(x)
        return x

class SGU(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        seq_len,
        eps = 1e-3
    ):
        super().__init__()
        self.eps = eps
        self.seq_len = seq_len
        self.norm = LayerNorm()
        self.proj_out = hk.Linear(dim_out)

    def __call__(self, x):
        n = self.seq_len
        x, gate = np.split(x, 2, axis = -1)

        gate = self.norm(gate)

        init_scale = self.eps / n
        init_eps = initializers.RandomUniform(minval = -init_scale, maxval = init_scale)

        weights = hk.get_parameter('spatial_weights', shape = (n, n), init = init_eps)
        biases = hk.get_parameter('spatial_biases', shape = (n, 1), init = np.ones)

        mask = np.tril(np.ones((n, n)))
        weights = weights * mask

        gate = np.einsum('n d, m n -> m d', gate, weights)
        gate += biases

        x = x * gate
        return self.proj_out(x)

class ProGenBase(hk.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        depth,
        window_size = 256,
        global_mlp_depth = 2,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        ff_glu = True,
        attn_dim = None,
        clamp_gate = True
    ):
        super().__init__()
        self.dim_head = dim_head
        self.embed = hk.Embed(num_tokens, dim)

        self.layers = []
        for i in range(depth):
            use_gmlp = (depth - i) <= global_mlp_depth
            use_ff_glu = not use_gmlp and ff_glu

            self.layers.append([
                LocalAttention(name = f'attn{i}', dim = dim, window_size = window_size, heads = heads, dim_head = dim_head),
                FeedForward(name = f'ff{i}', dim = dim, ff_mult = ff_mult, seq_len = seq_len, spatial_gate = use_gmlp, glu = use_ff_glu)
            ])

        self.to_logits = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_tokens)
        ])

    def __call__(self, x):
        n = x.shape[0]
        x = self.embed(x)
        rotary_emb = fixed_pos_embedding(n, self.dim_head)

        for attn, ff in self.layers:
            x += attn(x, pos_emb = rotary_emb)
            x += ff(x)

        return self.to_logits(x)

def ProGen(mixed_precision = False, mixed_precision_policy = dict(params = 'float32', compute = 'float16', output = 'float32'), **kwargs):
    @hk.transform
    def inner(seq):
        if mixed_precision:
            serialized_policy = ','.join([f'{k}={v}' for k, v in mixed_precision_policy.items()])
            policy = jmp.get_policy(serialized_policy)
            hk.mixed_precision.set_policy(ProGenBase, policy)
        return ProGenBase(**kwargs)(seq)
    return inner

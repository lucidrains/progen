from dotenv import load_dotenv
load_dotenv()

import click
import humanize
from pathlib import Path
from omegaconf import OmegaConf

import jax
from jax import nn, random, jit, tree_util, numpy as np

from haiku import PRNGSequence

from progen_transformer import ProGen
from progen_transformer.data import decode_tokens, encode_tokens
from progen_transformer.utils import sample, set_hardware_rng_
from progen_transformer.checkpoint import get_checkpoint_fns

# speedup rng

set_hardware_rng_(jax)

# main functions

@click.command()
@click.option('--seed', default = 42)
@click.option('--checkpoint_path', default = './ckpts')
@click.option('--config_path', default = './configs/model')
@click.option('--model_name', default = 'default')
@click.option('--prime', default = '')
def main(
    seed,
    checkpoint_path,
    config_path,
    model_name,
    prime,
):
    # prepare folders

    _, get_last_checkpoint, _ = get_checkpoint_fns(checkpoint_path)

    last_checkpoint = get_last_checkpoint()

    if last_checkpoint is None:
        exit(f'no checkpoints found at {checkpoint_path}')

    params = last_checkpoint['params']
    num_seqs = max(last_checkpoint['next_seq_index'], 0)

    # setup model and params

    config_folder_path = Path(config_path)
    config_path = config_folder_path / f'{model_name}.yml'
    assert config_path.exists(), f'path to your model config {str(config_path)} does not exist'

    model_kwargs = OmegaConf.load(str(config_path))
    model = ProGen(**model_kwargs)

    model_apply = jit(model.apply)
    rng = PRNGSequence(seed)

    # initialize all states, or load from checkpoint

    num_params = tree_util.tree_reduce(lambda acc, el: acc + el.size, params, 0)
    num_params_readable = humanize.naturalsize(num_params)

    # print

    print(f'params: {num_params_readable}')
    print(f'trained for {num_seqs} sequences')

    # sample with prime

    seq_len = model_kwargs['seq_len']

    prime_tokens = encode_tokens(prime)
    prime_length = len(prime_tokens) + 1
    prime_tensor = np.array(prime_tokens, dtype = np.uint16)

    sampled = sample(rng, jit(model_apply), params, prime_tensor, seq_len, top_k = 25, add_bos = True)
    sampled_str = decode_tokens(sampled[prime_length:])

    print("\n", prime, "\n", "*" * 40, "\n", sampled_str)

if __name__ == '__main__':
    main()

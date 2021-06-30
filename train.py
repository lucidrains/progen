from dotenv import load_dotenv
load_dotenv()

import click
import humanize
import time
from random import randrange
from shutil import rmtree
from pathlib import Path
import tqdm
import gzip
import numpy as np

from omegaconf import OmegaConf
from cloudpickle import pickle

import jax
from jax import nn, random, jit, tree_util
from optax import adam, clip_by_global_norm, chain, apply_updates, apply_every

from haiku import PRNGSequence

from progen_transformer import ProGen
from progen_transformer.data import decode_tokens, iterator_from_tfrecords_folder
from progen_transformer.utils import sample, get_train_loss_fn, set_hardware_rng_

import wandb

# speedup rng

set_hardware_rng_(jax)

# main functions

@click.command()
@click.option('--seed', default = 42)
@click.option('--num_batches', default = int(1e6))
@click.option('--batch_size', default = 4)
@click.option('--grad_accum_every', default = 4)
@click.option('--learning_rate', default = 2e-4)
@click.option('--max_grad_norm', default = 0.5)
@click.option('--validate_every', default = 100)
@click.option('--sample_every', default = 500)
@click.option('--checkpoint_every', default = 1000)
@click.option('--checkpoint_path', default = './ckpts')
@click.option('--config_path', default = './configs/model')
@click.option('--model_name', default = 'default')
@click.option('--prime_length', default = 25)
@click.option('--seq_len', default = 1024)
@click.option('--data_path', default = './train_data')
@click.option('--wandb_project_name', default = 'progen-training')
@click.option('--new', default = False, is_flag = True)
def main(
    seed,
    num_batches,
    batch_size,
    grad_accum_every,
    learning_rate,
    max_grad_norm,
    validate_every,
    sample_every,
    checkpoint_every,
    checkpoint_path,
    config_path,
    model_name,
    prime_length,
    seq_len,
    data_path,
    wandb_project_name,
    new
):
    # prepare folders

    if new:
        rmtree(str(checkpoint_path), ignore_errors = True)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents = True, exist_ok = True)

    # setup model and params

    config_folder_path = Path(config_path)
    config_path = config_folder_path / f'{model_name}.yml'
    assert config_path.exists(), f'path to your model config {str(config_path)} does not exist'

    model_kwargs = OmegaConf.load(str(config_path))
    model = ProGen(**model_kwargs)

    model_apply = jit(model.apply)
    rng = PRNGSequence(seed)
    loss_fn = get_train_loss_fn(model)

    # optimizer

    optim = chain(
        clip_by_global_norm(max_grad_norm),
        adam(learning_rate),
        apply_every(grad_accum_every)
    )

    # initialize all states, or load from checkpoint

    checkpoints = [c for c in checkpoint_path.glob('**/ckpt_*')]
    has_checkpoints = len(checkpoints) > 0

    if has_checkpoints:
        last_checkpoint_timestamp = max(list(map(lambda t: int(t.stem.split('_')[-1]), checkpoints)))
        last_checkpoint_path = checkpoint_path / f'ckpt_{last_checkpoint_timestamp}.pkl'
        with open(str(last_checkpoint_path), 'rb') as f:
            package = pickle.load(f)
            params = package['params']
            optim_state = package['optim_state']
            start_step = package['next_step']
            print(f'restoring from step {start_step}')
    else:
        mock_data = np.zeros((model_kwargs['seq_len'],), dtype = np.uint8)
        params = model.init(next(rng), mock_data)
        optim_state = optim.init(params)
        start_step = 0

    # experiment tracker

    num_params = tree_util.tree_reduce(lambda acc, el: acc + el.size, params, 0)
    num_params_readable = humanize.naturalsize(num_params)
    print(f'params: {num_params_readable}')

    wandb.config.num_params = num_params
    wandb.init(project = wandb_project_name)

    # get tf dataset

    train_loader = iterator_from_tfrecords_folder(
        data_path,
        seq_len = model_kwargs['seq_len'],
        batch_size = batch_size,
        skip = start_step
    )

    # training

    for i in tqdm.tqdm(range(start_step, num_batches), mininterval = 10., desc = 'training'):
        data = next(train_loader)

        loss, grads = loss_fn(params, next(rng), data)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = apply_updates(params, updates)

        if i % grad_accum_every == 0:
            print(f'loss: {loss.item()}')
            wandb.log({'loss': loss.item()})

        if i % checkpoint_every == 0:
            unix_time = int(time.time())
            package = {
                'next_step': i + 1,
                'params': params,
                'optim_state': optim_state
            }
            with open(str(checkpoint_path / f'ckpt_{unix_time}.pkl'), 'wb') as f:
                pickle.dump(package, f)

        if i % sample_every == 0:
            prime = data[0][:prime_length]
            prime_str = decode_tokens(prime)
            print(prime_str, "\n", "*" * 40)

            sampled = sample(rng, model_apply, params, prime, seq_len, top_k = 25)
            sampled_str = decode_tokens(sampled[prime_length:])
            print(sampled_str)

            wandb.log({'samples': wandb.Html(f'<i>{prime_str}</i><br/><br/><div style="overflow-wrap: break-word;">{sampled_str}</div>')})

if __name__ == '__main__':
    main()

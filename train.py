from dotenv import load_dotenv
load_dotenv()

import gzip

import click
import humanize
from jinja2 import Template
from random import randrange
from shutil import rmtree
from pathlib import Path
import tqdm
import numpy as np

from omegaconf import OmegaConf

import jax
from jax import nn, random, jit, tree_util
from optax import adam, clip_by_global_norm, chain, apply_updates, apply_every

from haiku import PRNGSequence

from progen_transformer import ProGen
from progen_transformer.data import decode_tokens, iterator_from_tfrecords_folder
from progen_transformer.utils import sample, get_train_loss_fn, set_hardware_rng_, confirm
from progen_transformer.checkpoint import get_checkpoint_fns

import wandb

# sample html

sample_tmpl = Template("""<i>{{prime_str}}</i><br/><br/><div style="overflow-wrap: break-word;">{{sampled_str}}</div>""")

# speedup rng

set_hardware_rng_(jax)

# main functions

@click.command()
@click.option('--seed', default = 42)
@click.option('--batch_size', default = 4)
@click.option('--grad_accum_every', default = 4)
@click.option('--learning_rate', default = 2e-4)
@click.option('--data_parallel', default = False, is_flag = True)
@click.option('--max_grad_norm', default = 0.5)
@click.option('--validate_every', default = 100)
@click.option('--sample_every', default = 500)
@click.option('--checkpoint_every', default = 1000)
@click.option('--checkpoint_path', default = './ckpts')
@click.option('--checkpoint_keep_n', default = 500)
@click.option('--config_path', default = './configs/model')
@click.option('--model_name', default = 'default')
@click.option('--prime_length', default = 25)
@click.option('--seq_len', default = 1024)
@click.option('--data_path', default = './train_data')
@click.option('--wandb_off', default = False, is_flag = True)
@click.option('--wandb_project_name', default = 'progen-training')
@click.option('--new', default = False, is_flag = True)
def main(
    seed,
    batch_size,
    grad_accum_every,
    learning_rate,
    max_grad_norm,
    validate_every,
    sample_every,
    checkpoint_every,
    checkpoint_path,
    checkpoint_keep_n,
    config_path,
    model_name,
    prime_length,
    seq_len,
    data_path,
    wandb_off,
    wandb_project_name,
    new
):
    # prepare folders

    reset_checkpoint, get_last_checkpoint, save_checkpoint = get_checkpoint_fns(checkpoint_path)

    if new:
        if not confirm('are you sure you want to clear all your checkpoints and restart training?'):
            exit()
        reset_checkpoint()

    # setup model and params

    config_folder_path = Path(config_path)
    config_path = config_folder_path / f'{model_name}.yml'
    assert config_path.exists(), f'path to your model config {str(config_path)} does not exist'

    model_kwargs = OmegaConf.load(str(config_path))
    model = ProGen(**model_kwargs)

    model_apply = jit(model.apply)
    rng = PRNGSequence(seed)
    loss_fn = get_train_loss_fn(model, data_parallel = data_parallel)

    # optimizer

    optim = chain(
        clip_by_global_norm(max_grad_norm),
        adam(learning_rate),
        apply_every(grad_accum_every)
    )

    # initialize all states, or load from checkpoint

    last_checkpoint = get_last_checkpoint()

    if last_checkpoint is not None:
        params = last_checkpoint['params']
        optim_state = last_checkpoint['optim_state']
        start_seq_index = last_checkpoint['next_seq_index']
    else:
        mock_data = np.zeros((model_kwargs['seq_len'],), dtype = np.uint8)
        params = model.init(next(rng), mock_data)
        optim_state = optim.init(params)
        start_seq_index = 0

    # experiment tracker

    seq_len = model_kwargs['seq_len']
    num_params = tree_util.tree_reduce(lambda acc, el: acc + el.size, params, 0)
    num_params_readable = humanize.naturalsize(num_params)

    wandb.config.num_params = num_params

    wandb_kwargs = {'mode': 'disabled'} if wandb_off else {}
    wandb.init(project = wandb_project_name, **wandb_kwargs)

    # get tf dataset

    total_train_seqs, get_train_dataset = iterator_from_tfrecords_folder(data_path, data_type = 'train')
    total_valid_seqs, get_valid_dataset = iterator_from_tfrecords_folder(data_path, data_type = 'valid',)

    assert total_train_seqs > 0, 'no protein sequences found for training'
    assert total_valid_seqs > 0, 'no protein sequences found for validation'

    train_dataset = get_train_dataset(
        seq_len = seq_len,
        batch_size = batch_size,
        skip = start_seq_index
    )

    valid_dataset = get_valid_dataset(
        seq_len = seq_len,
        batch_size = batch_size,
        loop = True
    )

    # print

    print(f'params: {num_params_readable}')
    print(f'sequence length: {seq_len}')
    print(f'num sequences: {total_train_seqs}')
    print(f'starting from sequence {start_seq_index}')

    # training

    effective_batch_size = batch_size * grad_accum_every
    seq_index_ranges = range(start_seq_index, total_train_seqs, effective_batch_size)    

    for i, seq_index in tqdm.tqdm(enumerate(seq_index_ranges), mininterval = 10., desc = 'training', total = len(seq_index_ranges)):
        for _ in range(grad_accum_every):
            data = next(train_dataset)

            loss, grads = loss_fn(params, next(rng), data)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = apply_updates(params, updates)

        print(f'loss: {loss.item()}')
        wandb.log({'loss': loss.item()})

        if i % checkpoint_every == 0:
            package = {
                'next_seq_index': seq_index + effective_batch_size,
                'params': params,
                'optim_state': optim_state
            }

            save_checkpoint(package, checkpoint_keep_n)
            print(f"checkpoint to start at sequence index of {package['next_seq_index']}")

        if i % validate_every == 0:
            valid_data = next(valid_dataset)
            loss, _ = loss_fn(params, next(rng), valid_data)
            print(f'valid_loss: {loss.item()}')
            wandb.log({'valid_loss': loss.item()})

        if i % sample_every == 0:
            valid_data = next(valid_dataset)[0]
            prime = valid_data[:prime_length]
            prime_str = decode_tokens(prime)

            sampled = sample(rng, model_apply, params, prime, seq_len, top_k = 25)
            sampled_str = decode_tokens(sampled[prime_length:])

            print(prime_str, "\n", "*" * 40, "\n", sampled_str)
            wandb.log({'samples': wandb.Html(sample_tmpl.render(prime_str = prime_str, sampled_str = sampled_str))})

if __name__ == '__main__':
    main()

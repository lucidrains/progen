import click
import humanize
from random import randrange
import tqdm
import gzip
import numpy as np

from torch.utils.data import DataLoader, Dataset

import jax
from jax import nn, random, jit, tree_util
from optax import adam, clip_by_global_norm, chain, apply_updates, apply_every

from haiku import PRNGSequence

from progen import ProGen
from progen.utils import sample, get_train_loss_fn

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    if token == 0:
        return ''
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# dataset

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = randrange(0, self.data.shape[0] - self.seq_len - 1)
        return self.data[rand_start: rand_start + self.seq_len + 1]

    def __len__(self):
        return self.data.shape[0] // self.seq_len

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
@click.option('--prime_length', default = 25)
@click.option('--seq_len', default = 1024)
@click.option('--data_path', default = './data/uniref50.sample.gz')
@click.option('--wandb_project_name', default = 'progen-training')
def main(
    seed,
    num_batches,
    batch_size,
    grad_accum_every,
    learning_rate,
    max_grad_norm,
    validate_every,
    sample_every,
    prime_length,
    seq_len,
    data_path,
    wandb_project_name
):
    # prepare enwik8 data

    with gzip.open(data_path) as file:
        all_data = np.fromstring(file.read(), dtype = np.uint8)
        data_len = all_data.shape[0]
        data_train, data_val = np.split(all_data, [int(data_len * 0.95)])

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset   = TextSamplerDataset(data_val, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size))

    # setup model and params

    model = ProGen(
        num_tokens = 256,
        dim = 512,
        seq_len = seq_len,
        depth = 6,
        global_mlp_depth = 2
    )

    model_apply = jit(model.apply)
    rng = PRNGSequence(seed)

    params = model.init(next(rng), train_dataset[0][:-1])
    num_params = tree_util.tree_reduce(lambda acc, el: acc + el.size, params, 0)
    num_params_readable = humanize.naturalsize(num_params)

    print(f'params: {num_params_readable}')

    loss_fn = get_train_loss_fn(model)

    # experiment tracker

    import wandb
    wandb.config.num_params = num_params
    wandb.init(project = wandb_project_name)

    # optimizer

    optim = chain(
        clip_by_global_norm(max_grad_norm),
        adam(learning_rate),
        apply_every(grad_accum_every)
    )

    optim_state = optim.init(params)

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval=10., desc='training'):
        data = next(train_loader).numpy()

        loss, grads = loss_fn(params, next(rng), data)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = apply_updates(params, updates)

        if i % grad_accum_every == 0:
            print(f'loss: {loss.item()}')
            wandb.log({'loss': loss.item()})

        if i % sample_every == 0:
            valid_data = next(val_loader).numpy()
            prime = valid_data[0][:prime_length]
            prime_str = decode_tokens(prime)
            print(prime_str, "\n", "*" * 40)

            sampled = sample(rng, model_apply, params, prime, seq_len, top_k = 25)
            sampled_str = decode_tokens(sampled[prime_length:])
            print(sampled_str)

            wandb.log({'samples': wandb.Html(f'<i>{prime_str}</i><br/><br/><div style="overflow-wrap: break-word;">{sampled_str}</div>')})

if __name__ == '__main__':
    main()

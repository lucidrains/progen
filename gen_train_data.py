import os
import gzip
import click
from itertools import islice
from Bio import SeqIO

from pathlib import Path

from omegaconf import OmegaConf
from dagster import execute_pipeline, pipeline, solid

@solid
def fasta_to_bytes(context):
    config = context.solid_config

    it = SeqIO.parse(config['read_from'], 'fasta')
    it = filter(lambda t: len(t.seq) + len(t.description) + 10 <= config['max_seq_len'], it)
    it = islice(it, 0, config['num_samples'])

    with gzip.open(config['write_to'], 'wb') as f:
        for sample in it:
            data = f' > {sample.description} # {sample.seq}'
            data = data.encode('utf-8')
            f.write(data)

@pipeline
def main_pipeline():
    fasta_to_bytes()

@click.command()
@click.option('--data_dir', default = './configs/data')
@click.option('--name', default = 'default')
def main(
    data_dir,
    name
):
    data_dir = Path(data_dir)
    config_path = data_dir / f'{name}.yml'
    assert config_path.exists(), f'config does not exist at {str(config_path)}'

    config = OmegaConf.load(str(config_path))

    execute_pipeline(
        main_pipeline,
        run_config = dict(
            solids = dict(
                fasta_to_bytes = dict(
                    config = config
                )
            )
        )
    )

if __name__ == '__main__':
    main()

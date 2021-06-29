import os
import gzip
import click
from math import ceil
from itertools import islice
from Bio import SeqIO

from tfrecord import TFRecordWriter
import numpy as np
from shutil import rmtree
from pathlib import Path

from omegaconf import OmegaConf
from dagster import execute_pipeline, pipeline, solid
from progen_transformer.data import write_tfrecord

# constants

TMP_DIR = Path('/tmp') / 'progen-seqs'
NUM_SEQUENCES_PER_FILE = 1000000

# helper functions

def clear_directory_(path):
    rmtree(str(path), ignore_errors = True)
    path.mkdir(exist_ok = True, parents = True)

# DAG functions

@solid
def fasta_to_tmp_files(context):
    config = context.solid_config
    clear_directory_(TMP_DIR)

    it = SeqIO.parse(config['read_from'], 'fasta')
    it = filter(lambda t: len(t.seq) + len(t.description) + 10 <= config['max_seq_len'], it)
    it = islice(it, 0, config['num_samples'])

    for index, sample in enumerate(it):
        seq = sample.seq
        annotation = f'[{sample.description}]'
        data = f'{annotation} # {seq}'
        data = data.encode('utf-8')

        filename = str(TMP_DIR / str(index))
        with gzip.open(filename, 'wb') as f:
            f.write(data)
    return

@solid
def files_to_tfrecords(context):
    config = context.solid_config
    permuted_sequences = np.random.permutation(config['num_samples'])

    write_to_path = Path(config['write_to'])
    clear_directory_(write_to_path)

    num_split = ceil(config['num_samples'] / NUM_SEQUENCES_PER_FILE)
    for file_index, indices in enumerate(np.array_split(permuted_sequences, num_split)):
        writer = TFRecordWriter(str(write_to_path / f'./{file_index}.tfrecord'))

        for index in indices:
            filename = str(TMP_DIR / str(index))
            with gzip.open(filename, 'rb') as f:
                write_tfrecord(writer, f.read())

        writer.close()

@pipeline
def main_pipeline():
    fasta_to_tmp_files()
    files_to_tfrecords()

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
                fasta_to_tmp_files = dict(
                    config = config
                ),
                files_to_tfrecords = dict(
                    config = config
                )
            )
        )
    )

if __name__ == '__main__':
    main()

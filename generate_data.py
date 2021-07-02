import os
import gzip
import click
from math import ceil
from itertools import islice
from Bio import SeqIO

from tfrecord import TFRecordWriter
import numpy as np
from random import random
from pathlib import Path

from omegaconf import OmegaConf
from dagster import execute_pipeline, pipeline, solid

from progen_transformer.data import write_tfrecord
from progen_transformer.utils import clear_directory_

# constants

TMP_DIR = Path('/tmp') / 'progen-seqs'
NUM_SEQUENCES_PER_FILE = 1000000

# DAG functions

@solid
def fasta_to_tmp_files(context):
    config = context.solid_config
    clear_directory_(TMP_DIR)

    it = SeqIO.parse(config['read_from'], 'fasta')
    it = filter(lambda t: len(t.seq) + len(t.description) + 10 <= config['max_seq_len'], it)
    it = islice(it, 0, config['num_samples'])

    for index, sample in enumerate(it):
        seq = str(sample.seq)
        annotation = f'[{sample.description}]'
        data = (annotation, seq)

        if random() <= config['prob_invert_seq_annotation']:
            data = tuple(reversed(data))

        data = ' # '.join(data)
        data = data.encode('utf-8')

        filename = str(TMP_DIR / str(index))
        with gzip.open(filename, 'wb') as f:
            f.write(data)
    return

@solid
def files_to_tfrecords(context):
    config = context.solid_config
    num_samples = len([*TMP_DIR.glob('**/*')])
    num_valids = int(config['fraction_valid_data'] * num_samples)

    # split out validation sequences

    permuted_sequences = np.random.permutation(num_samples)
    valid_seqs, train_seqs = np.split(permuted_sequences, [num_valids])

    # clear directory to write to

    write_to_path = Path(config['write_to'])
    clear_directory_(write_to_path)

    # loop and write all train and valid files to tfrecords

    for (seq_type, seqs) in (('train', train_seqs), ('valid', valid_seqs)):
        num_split = ceil(seqs.shape[0] / NUM_SEQUENCES_PER_FILE)

        for file_index, indices in enumerate(np.array_split(seqs, num_split)):
            num_sequences = len(indices)

            writer = TFRecordWriter(str(write_to_path / f'./{file_index}.{num_sequences}.{seq_type}.tfrecord'))

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

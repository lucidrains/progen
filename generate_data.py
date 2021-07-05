import os
import gzip
import click
from math import ceil
from itertools import islice
from Bio import SeqIO

import numpy as np
from random import random
from pathlib import Path

from omegaconf import OmegaConf
from google.cloud import storage
from dagster import execute_pipeline, pipeline, solid

from progen_transformer.data import with_tfrecord_writer
from progen_transformer.utils import clear_directory_

# constants

GCS_WRITE_TIMEOUT = 60 * 30
TMP_DIR = Path('./.tmp')

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

    num_sequences_per_file = config['num_sequences_per_file']

    # split out validation sequences

    permuted_sequences = np.random.permutation(num_samples)
    valid_seqs, train_seqs = np.split(permuted_sequences, [num_valids])

    # clear directory to write to

    write_to = config['write_to']
    upload_gcs = write_to.startswith('gs://')

    if upload_gcs:
        write_to = write_to[5:]
        client = storage.Client()
        bucket_name = write_to

        bucket = client.get_bucket(bucket_name)
        bucket.delete_blobs(list(bucket.list_blobs()))

    write_to_path = Path(write_to)
    clear_directory_(write_to_path)

    # loop and write all train and valid files to tfrecords

    for (seq_type, seqs) in (('train', train_seqs), ('valid', valid_seqs)):
        num_split = ceil(seqs.shape[0] / num_sequences_per_file)

        for file_index, indices in enumerate(np.array_split(seqs, num_split)):
            num_sequences = len(indices)
            tfrecord_filename = f'{file_index}.{num_sequences}.{seq_type}.tfrecord.gz'
            tfrecord_path = str(write_to_path / tfrecord_filename)

            with with_tfrecord_writer(tfrecord_path) as write:
                for index in indices:
                    filename = str(TMP_DIR / str(index))
                    with gzip.open(filename, 'rb') as f:
                        write(f.read())

            if upload_gcs:
                blob = bucket.blob(tfrecord_filename)
                blob.upload_from_filename(tfrecord_path, timeout = GCS_WRITE_TIMEOUT)

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

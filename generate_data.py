import os
import gzip
import click
import re
import random
from math import ceil
from functools import partial
from itertools import islice, chain
from operator import itemgetter

from pyfaidx import Faidx

import numpy as np
from random import random
from pathlib import Path

from omegaconf import OmegaConf
from google.cloud import storage

from prefect import Parameter, task, Flow

from progen_transformer.data import with_tfrecord_writer
from progen_transformer.utils import clear_directory_

# constants

GCS_WRITE_TIMEOUT = 60 * 30
TMP_DIR = Path('./.tmp')

# functions

def order_dict_by(d, fn):
    keys = fn(d.keys())
    return dict(tuple(map(lambda k: (k, d[k]), keys)))

def get_annotations_from_description(config, description):
    taxonomy_matches = re.findall(r'Tax=([a-zA-Z\s]*)\s[a-zA-Z\=]', description)
    annotations = dict()

    if len(taxonomy_matches) > 0:
        annotations['tax'] = taxonomy_matches[0]

    return annotations

def fasta_row_to_sequence_strings(config, fa, uid):
    seq_len = fa.index[uid].rlen
    seq = str(fa.fetch(uid, 1, seq_len))
    description = fa.get_long_name(uid)

    sequences = []
    annotations = get_annotations_from_description(config, description)
    # todo: gather annotations from GO

    if len(annotations) > 0:
        sort_annot_by = random.shuffle if not config['sort_annotations'] else sorted
        annotations = order_dict_by(annotations, sort_annot_by)

        annotation_str = [f"[{annot_name}={annot}]" for annot_name, annot in annotations.items()]
        annotation_str = ' '.join(annotation_str)

        seq_annot_pair = (annotation_str, seq)

        if random() <= config['prob_invert_seq_annotation']:
            seq_annot_pair = tuple(reversed(seq_annot_pair))

        sequence = ' # '.join(seq_annot_pair)
        sequence = sequence.encode('utf-8')
        sequences.append(sequence)

    sequence = f'# {seq}'
    sequence = sequence.encode('utf-8')
    sequences.append(sequence)

    return sequences

def process_and_write_to_tmp_file(i, seq_str):
    filename = TMP_DIR / str(i)
    with gzip.open(str(filename), 'wb') as f:
        f.write(seq_str)

def foreach(fn, it):
    for el in it:
        fn(*el)

# DAG functions

@task
def fasta_to_tmp_files(config):
    clear_directory_(TMP_DIR)

    print('reading from fasta')
    fa = Faidx(config['read_from'], sequence_always_upper = True)

    print('filtering by length')
    it = iter(fa.index.items())
    it = filter(lambda el: el[1].rlen <= config['max_seq_len'], it)

    print('parallel processing to tmp files')
    it = islice(it, 0, config['num_samples'])
    it = map(itemgetter(0), it)

    fasta_to_seq_fn = partial(fasta_row_to_sequence_strings, config, fa)
    it = map(fasta_to_seq_fn, it)
    it = enumerate(chain.from_iterable(it))
    foreach(process_and_write_to_tmp_file, it)

@task
def files_to_tfrecords(config):
    filenames = [*TMP_DIR.glob('**/*')]
    num_samples = len(filenames)
    num_valids = ceil(config['fraction_valid_data'] * num_samples)

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
                    filename = filenames[index]
                    with gzip.open(filename, 'rb') as f:
                        write(f.read())

            if upload_gcs:
                blob = bucket.blob(tfrecord_filename)
                blob.upload_from_filename(tfrecord_path, timeout = GCS_WRITE_TIMEOUT)

with Flow('parse-fasta') as flow:
    config = Parameter('config', required = True)
    fasta_to_tmp_files(config = config)
    files_to_tfrecords(config = config)

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
    flow.run(config = config)

if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
from functools import partial
from pathlib import Path
from contextlib import contextmanager

# writing tfrecords

def write(writer, values):
    record_bytes = tf.train.Example(features = tf.train.Features(feature={
        'seq': tf.train.Feature(bytes_list = tf.train.BytesList(value=[values]))
    })).SerializeToString()

    writer.write(record_bytes)

@contextmanager
def with_tfrecord_writer(path):
    options = tf.io.TFRecordOptions(compression_type = 'GZIP')

    with tf.io.TFRecordWriter(path, options = options) as writer:
        yield partial(write, writer)

# reading tfrecords

def parse_fn(sample):
    return tf.io.parse_single_example(sample, {
        'seq': tf.io.FixedLenFeature([], tf.string)
    })

def collate_fn(batch, pad_length, offset = 0):
    tensors = [np.frombuffer(el, dtype = np.uint8).astype(np.uint16) for el in batch.numpy()]
    tensors = map(lambda t: t[..., :pad_length], tensors)
    tensors = map(lambda t: t + offset, tensors)
    padded_tensors = map(lambda t: np.pad(t, (0, pad_length - t.shape[-1])), tensors)
    return np.stack(list(padded_tensors))

def iterator_from_tfrecords_folder(folder, data_type = 'train'):
    is_gcs_path = folder.startswith('gs://')

    if is_gcs_path:
        filenames = tf.io.gfile.glob(f'{folder}/*.{data_type}.tfrecord.gz')
    else:
        folder = Path(folder)
        filenames = [str(p) for p in folder.glob(f'**/*.{data_type}.tfrecord.gz')]

    num_seqs = sum(map(lambda t: int(t.split('.')[-4]), filenames))

    def iter_fn(
        seq_len,
        batch_size,
        skip = 0,
        loop = False
    ):
        dataset = tf.data.TFRecordDataset(filenames, compression_type = 'GZIP')

        dataset = dataset.skip(skip)
        dataset = dataset.map(parse_fn)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if loop:
            dataset = dataset.repeat()

        for batch in dataset:
            seq = batch['seq']
            batch_size = seq.shape[0]
            seq = collate_fn(seq, pad_length = seq_len, offset = 1)
            bos = np.zeros((batch_size, 1), dtype = np.uint16)
            seq = np.concatenate((bos, seq), axis = 1)
            yield seq

    return num_seqs, iter_fn

# tokenization

def encode_token(token):
    return ord(token) + 1

def decode_token(token):
    if token < 0:
        return ''
    return str(chr(token))

def encode_tokens(tokens):
    return list(map(encode_token, tokens))

def decode_tokens(tokens, offset = 1):
    return ''.join(list(map(decode_token, tokens.astype(np.int16) - offset)))

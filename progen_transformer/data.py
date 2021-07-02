import tfrecord
import tensorflow as tf
import numpy as np
from pathlib import Path

# writing tfrecords

def write_tfrecord(writer, datum):
    writer.write({'seq': (datum, 'byte')})

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

def iterator_from_tfrecords_folder(folder, *, seq_len, batch_size, data_type = 'train', skip = 0, loop = False):
    folder = Path(folder)
    filenames = [str(p) for p in folder.glob(f'**/*.{data_type}.tfrecord')]
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.skip(skip)
    dataset = dataset.map(parse_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    while True:
        for batch in dataset:
            seq = batch['seq']
            batch_size = seq.shape[0]
            seq = collate_fn(seq, pad_length = seq_len, offset = 1)
            bos = np.zeros((batch_size, 1), dtype = np.uint16)
            seq = np.concatenate((bos, seq), axis = 1)
            yield seq

        if not loop:
            break

# tokenization

def encode_token(token):
    return token.encode('utf-8') + 1

def decode_token(token):
    if token < 0:
        return ''
    return str(chr(token))

def encode_tokens(tokens):
    return list(map(encode_token, tokens))

def decode_tokens(tokens, offset = 1):
    return ''.join(list(map(decode_token, tokens.astype(np.int16) - offset)))

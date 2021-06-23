import gzip
from itertools import islice
from Bio import SeqIO

MAX_SEQ_LEN = 2048
NUM_SAMPLES = 25000

it = SeqIO.parse('./uniref50.fasta', 'fasta')
it = filter(lambda t: len(t.seq) + len(t.description) + 10 <= MAX_SEQ_LEN, it)
it = islice(it, 0, NUM_SAMPLES)

with gzip.open('./data/uniref50.sample.gz', 'wb') as f:
    for sample in it:
        data = f' > {sample.description} # {sample.seq}'
        data = data.encode('utf-8')
        f.write(data)

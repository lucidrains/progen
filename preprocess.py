import os
import gzip
import click
from itertools import islice
from Bio import SeqIO

@click.command()
@click.option('--read_from', default = './uniref50.fasta')
@click.option('--write_to', default = './data/uniref50.sample.gz')
@click.option('--num_samples', default = 25000)
@click.option('--max_seq_len', default = 2048)
def main(
    read_from,
    write_to,
    num_samples,
    max_seq_len
):
    it = SeqIO.parse(read_from, 'fasta')
    it = filter(lambda t: len(t.seq) + len(t.description) + 10 <= max_seq_len, it)
    it = islice(it, 0, num_samples)

    with gzip.open(write_to, 'wb') as f:
        for sample in it:
            data = f' > {sample.description} # {sample.seq}'
            data = data.encode('utf-8')
            f.write(data)

if __name__ == '__main__':
    main()

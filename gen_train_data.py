import os
import gzip
import click
from itertools import islice
from Bio import SeqIO

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
@click.option('--read_from', default = './uniref50.fasta')
@click.option('--write_to', default = './data/uniref50.sample.gz')
@click.option('--num_samples', default = 25000)
@click.option('--max_seq_len', default = 2048)
def main(**config):
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

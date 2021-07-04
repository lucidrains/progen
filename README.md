## ProGen - (wip)

Implementation and replication of <a href="https://arxiv.org/abs/2004.03497">ProGen</a>, Language Modeling for Protein Generation, in Pytorch and Jax (the weights will be made easily transferrable between the two). You can think of this as GPT for proteins sequences.

## Install

```bash
$ pip install progen-transformer
```

## Usage

```python
from jax import random
from haiku import PRNGSequence
from progen_transformer import ProGen

model = ProGen(
    num_tokens = 256,
    dim = 512,
    seq_len = 1024,
    window_size = 256,       # local attention window size
    depth = 12,              # depth
    heads = 8,               # attention heads
    dim_head = 64,           # dimension per head
    ff_glu = True,           # use GLU in feedforward, from Noam's paper
    global_mlp_depth = 2     # last N global gmlp layers
)

rng = PRNGSequence(42)
seq = random.randint(next(rng), (1024,), 0, 256)

params = model.init(next(rng), seq)
logits = model.apply(params, next(rng), seq) # (1024, 256)
```

## Training

Download Uniref50 from <a href="https://www.uniprot.org/downloads">UniProt</a> and place `uniref50.fasta` in the root directory

```bash
$ python generate_data.py
```

You should see a lot of green if everything succeeds. Then


```bash
$ python train.py
```

By default, the script will checkpoint and resume automatically, but if you wish to clear your progress and restart, just add a `--new` flag

```bash
$ python train.py --new
```

Model checkpoints will be saved periodically to `./ckpts`

Finally, to sample from your checkpoint, just do

```bash
$ python sample.py
```

You can pass a prime with `--prime`. You can either pass the annotations, followed by `#`, to get the generated sequence, or pass the sequence (also followed by `#`) and get the generated annotations

```bash
$ python sample.py --prime "[Tax=Mammalia] #"
```

## Mixed Precision

To use mixed precision training, you'll need to install the latest Haiku with the following command

```bash
$ pip install git+https://github.com/deepmind/dm-haiku
```

Then make sure to set the `--mixed_precision` flag when invoking the training script

```bash
$ python train.py --mixed_precision
```

## Todo

- [ ] model parallelism with pjit
- [ ] join in GO annotations with pandas dataframe
- [ ] setup annotation -> template string system, all configuration driven, find easy way to test. offer two types of annotations, one parsed from uniref descriptions, the other from GO annotation presence
- [ ] add multiple data sources (check out trembl)
- [ ] when sampling, prime with entire sequence prior to the pound sign (intersection of sequence and annotation)
- [ ] utilize all cores when processing data
- [ ] save all training settings in the checkpoints too
- [x] bfloat16 on xla
- [x] resume from correct place in tfrecord even if batch size is changed inbetween runs, display number of sequences processed
- [x] train compressed gzip tfrecords from google cloud storage path
- [x] remove tfrecord package and just use tfrecordwriter with gzip
- [x] generate validation tfrecords
- [x] checkpoint and resume from a google cloud storage path
- [x] use jinja2 for wandb html sample logging
- [x] manage experimental tracker state, and also allow ability to turn it off by piping to noop
- [x] add a confirmation before clearing a folder for --new run
- [x] engineer mask in cross entropy loss so that padding can be reused as end-of-string token
- [x] flip seq # annotation order with prob set in config
- [x] keep N last checkpoints

## Acknowledgements

Many thanks goes out to <a href="https://github.com/kingoflolz">Ben Wang</a>, who showed this type of large-scale training can be achieved with <a href="https://github.com/kingoflolz/mesh-transformer-jax">GPT-J</a>

## Citations

```bibtex
@misc{madani2020progen,
    title   = {ProGen: Language Modeling for Protein Generation}, 
    author  = {Ali Madani and Bryan McCann and Nikhil Naik and Nitish Shirish Keskar and Namrata Anand and Raphael R. Eguchi and Po-Ssu Huang and Richard Socher},
    year    = {2020},
    eprint  = {2004.03497},
    archivePrefix = {arXiv},
    primaryClass = {q-bio.BM}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

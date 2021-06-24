## ProGen - (wip)

Implementation and replication of <a href="https://arxiv.org/abs/2004.03497">ProGen</a>, Language Modeling for Protein Generation, in Pytorch and Jax (the weights will be made easily transferrable between the two)

## Install

```bash
$ pip install progen
```

## Usage

```python
from jax import random
from haiku import PRNGSequence
from progen import ProGen

model = ProGen(
    num_tokens = 256,
    dim = 512,
    seq_len = 1024,
    window_size = 256,       # local attention window size
    depth = 12,              # depth
    heads = 8,               # attention heads
    dim_head = 64,           # dimension per head
    global_mlp_depth = 2     # last N global gmlp layers
)

rng = PRNGSequence(42)
seq = random.randint(next(rng), (1024,), 0, 256)

params = model.init(next(rng), seq)
logits = model.apply(params, next(rng), seq) # (1024, 256)
```

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

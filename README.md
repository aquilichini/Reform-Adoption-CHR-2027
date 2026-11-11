# [Paper code] The Speed of Reform Adoption: A Computational Study of 20th-Century Norwegian Newspapers

This repository provides code to execute the experiments in the paper:

> Quilichini et al. *The Speed of Reform Adoption: A Computational Study of 20th-Century Norwegian Newspapers.* Anthology of Computers and the Humanities, 2027.

If you use this code, kindly cite the paper:
```bibtex
@article{quilichini_2027_speed,
  title   = {The Speed of Reform Adoption: A Computational Study of 20th-Century Norwegian Newspapers},
  author  = {Amélie Quilichini and David Nabergoj and Antoine Doucet and Marko Robnik Šikonja},
  journal = {Anthology of Computers and the Humanities},
  year    = {2027},
  note    = {To appear}
}
```

Conference website: https://2027.computational-humanities-research.org/.

## Repository organization
Plots that appear in the paper can be found in [`notebooks/plots.ipynb`](notebooks/plots.ipynb).
The [`src`](src) directory contains text processing pipelines, including CoNLL-U parsing and language classification.
The [`tests`](tests) directory contains unit tests for key pieces of code.
The [`data`](data) directory contains word pairs for each reform.
A list of websites providing details on individual Norwegian language reforms can be found [here](REFORM_WEBSITES.md).

## Setup the environment

First, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and environment dependencies:
```bash
uv venv
.venv\Scripts\activate
uv sync
```

Alternatively, install inside a virtualenv (editable mode via pip):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Reproducing the experiments

The newspaper articles can be obtained from the [National Library of Norway](https://www.nb.no/).
After acquiring the articles, run text processing via:
```bash
uv run src/main.py
```

Finally, execute the [`plots.ipynb` notebook](notebooks/plots.ipynb), which displays adoption curve figures and stores them as `.pdf` files to the same directory.
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
* Plots that appear in the paper can be found in [`notebooks/plots.ipynb`](notebooks/plots.ipynb).
* The [`src`](src) directory contains text processing pipelines, including CoNLL-U parsing and language classification.
* The [`tests`](tests) directory contains unit tests for key pieces of code.
* The [`data`](data) directory contains word pairs for each reform ([`rules.csv`](data/rules.csv)) and supporting statistics.
* Run settings and paths are defined in [`src/config.py`](src/config.py).
* See the [list of websites](REFORM_WEBSITES.md) providing details on individual Norwegian language reforms.

## Setup the environment

The code requires Python 3.12.
[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then install the dependencies:
```bash
uv sync
```
This creates a `.venv` virtual environment with Python 3.12 and all dependencies.
Run commands through `uv run` (as below), so there is no need to activate the environment.
You can run the tests with `uv run pytest`.

## Reproducing the experiments

While we cannot publish the newspaper articles due to licensing restricitons, they can be obtained from the [National Library of Norway](https://www.nb.no/).
The pipeline expects the articles as a CoNLL-U corpus, with one `.conllu` file per newspaper issue, organized by newspaper and year:
```
<corpus_dir>/
└── klassekampen/
    └── 1922/
        └── 19220729_klassekampen.conllu
```
Set `CORPUS_DIR` in [`src/config.py`](src/config.py) to the root of your corpus directory, then run text processing via:
```bash
uv run src/main.py
```
Intermediate results (preprocessed and language-classified corpus caches) are written to the `output` directory.  
Finally, execute [`plots.ipynb`](notebooks/plots.ipynb), which displays adoption curve figures and stores them as `.pdf` files to the same directory.
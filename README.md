# Graph Utilities and Data - [GUD](https://github.com/jackd/gud)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python package for machine learning on graphs.

See also [graph-tfds](https://github.com/jackd/graph-tfds) for [tensorflow-datasets](https://github.com/tensorflow/datasets) implementations of larger datasets.

## Data

[Single graph data](gud/data/single.py) from

- [EdisonLeeeee/GraphData](https://github.com/EdisonLeeeee/GraphData)
- [klicperajo/ppnp](https://github.com/klicperajo/ppnp)

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```

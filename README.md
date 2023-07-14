Sempervirens Phylogenetic Tree Reconstructor
===
Reconstructs phylogenetic trees from noisy data.

## Commandline Usage

Simple usage is of the form:
```bash
python reconstructor.py noisy_matrix_filename fpr fnr mer
```

The output file defaults to "noisy_matrix_filename.CFMatrix", and can be specified with the optional "-o" flag:
```bash
python reconstructor.py noisy_matrix_filename fpr fnr mer -o output_filename
```

Help information can be found by running `python reconstructor.py -h`.

### Example Usage
```bash
python reconstructor.py noisy_data.SC 0.001 0.2 0.05 -o reconstructed_data.SC.CFMatrix
```

## Library Usage

The `reconstructor.py` file can be imported and used as a library in other Python code.
Example usage is as follows.

```python
from reconstructor import reconstruct
...
reconstruction = reconstruct(noisy_mat, fpr, fnr, mer)
...
```

When used as a library, the only dependency is `numpy`.

## Files and Directories:
`sempervirens`:
* `reconstructor.py`: All code for reconstruction.
* `metrics.py`: Functions for metrics; e.g. ancestor-descendant score.

`tests`: Code for evaluating algorithms on datasets and comparing them. 

`data`: Datasets for testing algorithms.
* Unzip `data.zip` to get this folder.

`metrics`: Results of running algorithms on various datasets.

## Setup

Use either `poetry` or the `requirements.txt` (by running `pip install -r requirements.txt`), tested with Python 3.10.

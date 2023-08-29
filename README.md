Sempervirens Phylogenetic Tree Reconstructor
===
Reconstructs phylogenetic trees from noisy data.

## Commandline Usage

The following reconstructs the noisy matrix in `noisy_matrix_filename` given false positive rate `fpr`, false negative rate `fnr`, and missing entry rate `mer`. The reconstructed matrix is written to `noisy_matrix_filename.CFMatrix`.
```bash
python reconstructor.py noisy_matrix_filename fpr fnr mer
```

The output file can be specified with the optional `-o` flag:
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

An example of using Pandas to read and write matrices of files can be seen at the bottom of `reconstructor.py`.

## Input/Output Format

The input and output to the `reconstruct` function is a NumPy matrix with `n` rows and `m` columns. This describes `n` objects in terms of `m` characters. Element `(i, j)` of the matrix can be either `0`, `1`, or `3`. If it is `0`, then object `i` does not have character `j`. If it is `1`, then object `i` has character `j`. If it is `3`, then it is unknown whether object `i` has character `j` (the output of `reconstruct` will not have any `3`s).

Below is an example file used for input. The input file must be a tab separated value (TSV) file. The first row and column of the file are used as labels of the rows and columns respectively. The rest of the TSV must represent a matrix with the format described above. The output file will be of the same format as the input file, reusing the input file's row and column labels.
```
cellID/mutID  mut0  mut1  mut2  mut3  mut4  mut5  mut6  mut7
cell0         0     0     3     0     0     0     0     0
cell1         0     3     1     0     0     0     1     1
cell2         0     0     1     0     0     0     1     1
cell3         1     1     0     0     0     0     0     0
cell4         0     0     1     0     3     0     0     0
cell5         1     0     0     0     0     0     0     0
cell6         0     0     1     0     0     0     1     1
cell7         0     0     1     0     0     0     0     0
cell8         3     0     0     0     3     0     3     1
cell9         0     1     0     0     0     0     0     0
```

## Installation

Sempervirens requires Python and only two Python packages: NumPy and Pandas. Pandas can be omitted if only using Sempervirens as a library. 

First, make sure Python is installed. Then the dependencies can by installed as follows.
```bash
$ pip install numpy
$ pip install pandas
```

Download `reconstructor.py` from the repository.

Finally, test the script with **TODO**.

All code has been tested with Python 3.10.

## Files and Directories:
`sempervirens`:
* `reconstructor.py`: All code for reconstruction.
* `metrics.py`: Functions for metrics; e.g. ancestor-descendant score.

`tests`: Code for evaluating algorithms on datasets and comparing them. 

`data`: Datasets for testing algorithms.
* Unzip `data.zip` to get this folder.

`metrics`: Results of running algorithms on various datasets.

## Citing

If you found this code useful, consider citing the associated paper:

**TODO**


## Contact Information

For questions, please contact [Neelay Junnarkar](mailto:neelay.junnarkar@berkeley.edu) and [Can Kizilkale](mailto:cankizilkale@lbl.gov).
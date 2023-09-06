# Sempervirens: error correction for phylogenetic tree matrices.
# Written by: Neelay Junnarkar (neelay.junnarkar@berkeley.edu) and Can Kizilkale (cankizilkale@berkeley.edu).
# Last update: Sep 5, 2023.

# Can be used from the commandline or as a library from other code.
#
# Commandline usage:
# python reconstructor.py noisy_matrix_filename fpr fnr mer
# The output file defaults to noisy_matrix_filename.CFMatrix.
# Can specify output file with the "-o" flag: python reconstructor.py noisy_matrix_filename fpr fnr mer -o output_filename.
# To get help information, run: python reconstructor.py -h 
# Dependencies: numpy, and pandas.
#
# Example:
# python reconstructor.py noisy_data.SC 0.001 0.2 0.05 -o reconstructed_data.SC.CFMatrix
#
#
# Library usage:
# Import file and use the reconstruct function.
# Dependencies: numpy.
#
# Example:
# from reconstructor import reconstruct
# reconstruction = reconstruct(noisy_mat, fpr, fnr, mer)


###### Reconstructor code ######

import numpy as np

def correct_pivot_subtree_columns(pivot_col, cols_sorted, mat, fpr, fnr):
    """Finds all columns of mat that ocurr in the subtree of the pivot_col column,
    assuming that each entry of mat is subject to fpr and fnr, and pivot_col is correct.
    """
    mat_cols = mat[:, cols_sorted]
    K_11 = pivot_col @ mat_cols
    K_01 = (1 - pivot_col) @ mat_cols
    cols_mask = K_11 >= K_01
    return cols_sorted[cols_mask]

def noisy_pivot_subtree_columns(pivot_col, cols_sorted, mat, fpr, fnr):
    """Finds all columns of mat that ocurr in the same subtree as the pivot_col column,
    assuming that each entry of mat and pivot_col is subject to fpr and fnr.
    """
    mat_cols = mat[:, cols_sorted]
    K_11 = pivot_col @ mat_cols
    K_01 = (1 - pivot_col) @ mat_cols
    K_10 = pivot_col @ (1 - mat_cols)
    if fnr >= fpr:
        c0 = np.log((1-fnr)/fpr)
        c1 = np.log((1-fpr)/fnr)
        cols_mask = np.logical_or(
            K_11 * c0 >= K_01 * c1,
            K_11 * c0 >= K_10 * c1,
        )
    else:
        cols_mask = np.logical_or(
            K_11 >= K_01,
            K_11 >= K_10,
        )
    return cols_sorted[cols_mask]

def reconstruct_root(cols_in_subtree, mat, fnr):
    """Reconstructs the root row for the subtree consisting of the columns in cols_in_subtree."""
    counts = np.sum(mat[:, cols_in_subtree], axis = 0)
    root = np.zeros(mat.shape[1])
    root[cols_in_subtree] = (counts >= np.max(counts)*(1 - fnr)).astype(int)
    return root

def reconstruct_pivot(cols_in_subtree, cols_sorted, mat):
    """Reconstructs the pivot column based on the columns in its subtree."""
    # Count the number of times a row occurs in the subtree.
    subtree_in = mat[:, cols_in_subtree].sum(axis = 1)

    # Count the number of times a row occurs outside of the subtree but in non-reconstructed columns.
    out_set = np.setdiff1d(np.array(cols_sorted), np.array(cols_in_subtree), assume_unique = True)
    subtree_out = mat[:, out_set].sum(axis = 1) 

    # Set a row of the reconstruction to 1 if the row occurs more in the subtree than out of it.
    reconstructed_pivot = (subtree_in >= subtree_out).astype(int) * (subtree_in > 0).astype(int)
    return reconstructed_pivot

def count_based_max_likelihood(pivot, col_set, mat):
    """Returns the index of the column in mat (of columns in col_set) that is most similar to pivot."""
    return col_set[np.argmax(pivot.T @ mat[:, col_set])]

def column_max_likelihood_refinement(reconstructed_mat, noisy_mat, fpr, fnr, mer):
    """Create and return a refined matrix from the columns in reconstructed_mat by, 
    for each column in the noisy matrix, selecting the column in reconstructed_mat 
    that maximizes the likelihood of measuring the noisy column.
    Note that noisy_mat may contain 3s.
    """
    noisy0T = (noisy_mat == 0).astype(int).T
    noisy1T = (noisy_mat == 1).astype(int).T
    K_00 = noisy0T @ (1 - reconstructed_mat)
    K_01 = noisy0T @ reconstructed_mat
    K_10 = noisy1T @ (1 - reconstructed_mat)
    K_11 = noisy1T @ reconstructed_mat
    log_probs = np.log(1 - fpr - mer) * K_00 \
                + np.log(fnr) * K_01 \
                + np.log(fpr) * K_10 \
                + np.log(1 - fnr - mer) * K_11
    refinement = reconstructed_mat[:, log_probs.argmax(axis = 1)]
    return refinement

def split_part(part, mat, fpr, fnr):
    """Finds the largest maximal subtree in a set of columns, determines and removes the pivot,
    enforces column relationships, and returns the subtree, remaining columns, pivot,
    and non-reconstructed column most similar to the pivot.
    """
    # Partition part into maximal subtrees and find the largest one.
    partition = []
    temp_part = part.copy()
    while temp_part.size > 0:
        candidate_pivot_col_i = temp_part[0]
        candidate_pivot = mat[:, candidate_pivot_col_i]
        candidate_cols_in_subtree = noisy_pivot_subtree_columns(candidate_pivot, temp_part, mat, fpr, fnr)
        partition.append((candidate_pivot_col_i, candidate_cols_in_subtree))
        temp_part = np.setdiff1d(temp_part, candidate_cols_in_subtree, assume_unique = True)
    part_lengths = [part[1].size for part in partition]
    assert np.sum(part_lengths) == part.size
    part_i = np.argmax(part_lengths)
    pivot_col_i, cols_in_subtree = partition.pop(part_i)

    # Adjust mat to ensure that rows which should be in the subtree are supersets of the subtree root.
    root = reconstruct_root(cols_in_subtree, mat, fnr)
    rows_in_subtree_mask2 = mat @ root > np.sum(root) / 2
    rows_in_subtree2 = np.flatnonzero(rows_in_subtree_mask2)
    mat[rows_in_subtree2, :] = np.logical_or(mat[rows_in_subtree2, :], root).astype(int)

    # Re-find pivot based on adjusted mat.
    reconstructed_pivot = reconstruct_pivot(cols_in_subtree, part, mat)
    
    # Find columns in subtree of the reconstructed pivot assuming the reconstructed pivot is correct.
    final_cols_in_subtree = correct_pivot_subtree_columns(reconstructed_pivot, part, mat, fpr, fnr)

    if final_cols_in_subtree.size > 0:
        # Find the columns which haven't been reconstructed yet, but are outside the reconstructed pivot's subtree.
        unreconstructed_cols_out_subtree = np.setdiff1d(np.array(part), np.array(final_cols_in_subtree), assume_unique = True)

        # Go through all mat, enforcing column structure with pivot_reconstructed.
        # Any column in the subtree of the reconstructed pivot must be a subset of the reconstructed pivot.
        mat[:, final_cols_in_subtree] *= reconstructed_pivot.reshape((-1, 1))
        # Any undecided column outside of the subtree must be disjoint with the reconstructed pivot.
        mat[:, unreconstructed_cols_out_subtree] *= 1 - reconstructed_pivot.reshape((-1, 1))

        # Find which column of mat should be replaced with the reconstructed pivot.
        col_placement_i = count_based_max_likelihood(reconstructed_pivot, final_cols_in_subtree, mat)

        # Setup split and return.
        # Delete col_placement_i wherever it is.
        split_a = np.delete(final_cols_in_subtree, np.flatnonzero(final_cols_in_subtree == col_placement_i))
        split_b = unreconstructed_cols_out_subtree
        # split_b = np.delete(unreconstructed_cols_out_subtree, np.flatnonzero(unreconstructed_cols_out_subtree == col_placement_i))

        assert split_a.size + split_b.size + 1 == part.size

        return ([split_a, split_b], reconstructed_pivot, col_placement_i)
    else:
        # Default to replacing the column the pivot started as.
        col_placement_i = pivot_col_i
        split_b = np.delete(part, np.flatnonzero(part == col_placement_i)) # split_a is empty
        return ([split_b], reconstructed_pivot, col_placement_i)


def reconstruct(noisy, fpr, fnr, mer):
    """Reconstructs a phylogenetic tree from the noisy matrix.

    Args:
        noisy: N x M numpy array. The noisy matrix to be corrected. 
            Elements are of values 0, 1, or 3. 
            Zero represents mutation absent, 1 represents mutation present,
            and 3 represents missing entry.
        fpr: scalar. False positive rate.
        fnr: scalar. False negative rate.
        mer: scalar. Missing entry rate.
    
    Returns:
      N x M matrix that represents the reconstructed phylogenetic tree.
      In this matrix, for any two columns, either one is a subset of the other or they are disjoint.
    """
    assert np.all(np.logical_or(noisy == 0, np.logical_or(noisy == 1, noisy == 3))), \
        "noisy matrix must contain only 0s, 1s, and 3s."
    assert 0 <= fpr <= 1 and 0 <= fnr <= 1 and 0 <= mer <= 1, "fpr, fnr, and mer must be in [0, 1]."

    mat = noisy.copy()

    # Set all missing elements to 0s.
    mat[mat == 3] = 0

    col_set = np.array(range(mat.shape[1]))
    col_sums = np.sum(mat[:, col_set], axis = 0)
    assert np.all(col_sums.shape == (mat.shape[1],))
    cols_sorted = col_set[np.argsort(-col_sums)] # cols_sorted used as a sorted col_set from here on.
    
    partition = [cols_sorted] # Start with all columns in one part

    while len(partition) > 0:
        # Take a part and split into two
        part = partition.pop(0)
        subpartition, reconstructed_pivot, col_i = split_part(part, mat, fpr, fnr)
        # Place reconstructed pivot in matrix
        mat[:, col_i] = reconstructed_pivot
        # Replace part in partition with the new parts
        for subpart in subpartition:
            if subpart.size > 0:
                partition.append(subpart)

    # Row maximum likelihood
    mat = column_max_likelihood_refinement(mat.T, noisy.T, fpr, fnr, mer).T
    # Column maximum likelihood
    mat = column_max_likelihood_refinement(mat, noisy, fpr, fnr, mer)

    assert np.all(np.logical_or(mat == 0, mat == 1))
    return mat


###### Running reconstructor as command line program ######

def main():
    import argparse
    import pandas as pd

    def float_closed_unit_interval(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid float value: {arg}")
        if f < 0.0 or f > 1.0:
            raise argparse.ArgumentTypeError(f"invalid float value {arg}: argument must be in [0.0, 1.0]")
        return f

    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type = str, help = "Input file to read noisy matrix from.")
    parser.add_argument("fpr", type = float_closed_unit_interval, help = "False positive rate.")
    parser.add_argument("fnr", type = float_closed_unit_interval, help = "False negative rate.")
    parser.add_argument("mer", type = float_closed_unit_interval, help = "Missing entry rate.")
    parser.add_argument("-o", "--out_file", type = str, help = "Output file to write conflict-free matrix to. Defaults to IN_FILE.CFMatrix.")
    parser.add_argument("-v", "--verbose", action = "store_true")
    args = parser.parse_args()

    assert args.fpr + args.mer <= 1.0, "fpr + mer must be in [0.0, 1.0]"
    assert args.fnr + args.mer <= 1.0, "fnr + mer must be in [0.0, 1.0]"
    if args.out_file is None:
        args.out_file = args.in_file + ".CFMatrix"

    if args.verbose:
        print(f"Reading noisy matrix from {args.in_file} with false positive rate {args.fpr}, false negative rate {args.fnr}, and missing entry rate {args.mer}.")
        print(f"Conflict-free matrix will be written to {args.out_file}.")

    noisy_df = pd.read_csv(args.in_file, sep = '\t', index_col = 0)

    reconstructed_mat = reconstruct(noisy_df.to_numpy(), args.fpr, args.fnr, args.mer)

    reconstructed_df = pd.DataFrame(reconstructed_mat)
    reconstructed_df.columns = noisy_df.columns
    reconstructed_df.index = noisy_df.index
    reconstructed_df.index.name = "cellIDxmutID"
    reconstructed_df.to_csv(args.out_file, sep = "\t")
    if args.verbose:
        print(f"Output written.")

if __name__ == '__main__':
    main()

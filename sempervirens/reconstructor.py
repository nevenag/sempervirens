import numpy as np

def column_in_subtree(col_a, col_b):
    """Returns whether col_a ocurrs in the subtree of col_b."""
    # Checks whether more of column a lies in the subtree of column b or outside of it.
    return np.sum(col_a * col_b) >= np.sum(col_a * (1 - col_b))

def find_subtree_columns(pivot_col_i, cols_sorted, mat):
    """Finds all columns that ocurr in the subtree of mat's pivot_col_i'th column."""
    cols_in_subtree = []
    for col_i in cols_sorted:
        if column_in_subtree(mat[:, col_i], mat[:, pivot_col_i]):
            cols_in_subtree.append(col_i)
    return cols_in_subtree

def reconstruct_root(cols_in_subtree, mat, fnr):
    """Reconstructs the root row for the subtree consisting of the columns in cols_in_subtree."""
    counts = np.sum(mat[:, cols_in_subtree], axis = 0)
    root = np.zeros(mat.shape[1])
    root[cols_in_subtree] = (counts >= np.max(counts)*(1 - fnr)).astype(int)
    return root

def row_in_subtree(row_a, row_b):
    """Returns whether row_a is in the subtree of row_b."""
    return np.sum(row_a * row_b) >= np.sum(row_b) * (1/2)

def reconstruct_pivot(cols_in_subtree, cols_sorted, mat):
    """Reconstructs the pivot column based on the columns in its subtree."""
    # Count the number of times a row occurs in the subtree.
    subtree_in = mat[:, cols_in_subtree].sum(axis = 1)

    # Count the number of times a row occurs outside of the subtree but in undecided columns.
    out_set = list(set(cols_sorted) - set(cols_in_subtree))
    subtree_out = mat[:, out_set].sum(axis = 1) 

    # Set a row of the reconstruction to 1 if the row occurs more in the subtree than out of it.
    reconstructed_pivot = np.logical_and(subtree_in >= subtree_out, subtree_in > 0).astype(int)
    return reconstructed_pivot

def count_based_max_likelihood(pivot, col_set, mat):
    """Returns the index of the column in mat (of columns in col_set) that is most similar to pivot."""
    return col_set[(pivot.T @ mat[:, col_set]).argmax()]

def column_max_likelihood_refinement(reconstructed_mat, noisy_mat, fpr, fnr):
    """
    Create and return a refined matrix from the columns in reconstructed_mat by, 
    for each column in the noisy matrix, selecting the column in reconstructed_mat 
    that maximizes the likelihood of measuring the noisy column.
    """
    refinement = np.zeros_like(noisy_mat)
    for i in range(refinement.shape[1]):
        noisy_col = noisy_mat[:, i].reshape((noisy_mat.shape[0], 1))
        log_probs = np.log(fpr) * np.sum(noisy_col > reconstructed_mat, axis = 0) + \
                    np.log(fnr) * np.sum(noisy_col < reconstructed_mat, axis = 0)
        refinement[:, i] = reconstructed_mat[:, np.argmax(log_probs)]
    return refinement

def reconstruct(measured, fpr, fnr, mer):
    """
    Reconstructs a phylogenetic tree from the measured matrix.

    Args:
        measured: N x M np array. The measured matrix to be corrected. 
            Elements are of values 0, 1, or 3. 
            Zero represents measured absent, 1 represents measured present,
            and 3 represents missing entry.
        fpr: scalar. False positive rate.
        fnr: scalar. False negative rate.
        mer: scalar. Missing entry rate.
    
    Returns:
      Matrix that represents the reconstructed phylogenetic tree.
    """
    
    assert np.all(np.logical_or(measured == 0, np.logical_or(measured == 1, measured == 3))), \
        "measured must contain only 0s, 1s, and 3s."
    assert 0 <= fpr <= 1 and 0 <= fnr <= 1 and 0 <= mer <= 1, "fpr, fnr, and mer must be in [0, 1]."

    mat = measured.copy()
    reconstruction = np.zeros_like(mat)

    # Set all missing elements to 0s.
    mat[mat == 3] = 0

    col_set = np.array(range(mat.shape[1]))
    col_sums = np.sum(mat[:, col_set], axis = 0)
    assert np.all(col_sums.shape == (mat.shape[1],))
    cols_sorted = col_set[np.argsort(-col_sums)]
    # cols_sorted used as a sorted col_set from here on.

    while cols_sorted.size > 0:
        col_sums = np.sum(mat[:, cols_sorted], axis = 0)
        cols_sorted = cols_sorted[np.argsort(-col_sums)] # Sort from highest to lowest column sum.
        pivot_col_i = cols_sorted[0] # Get the column with the most number of ones.

        # # Find the columns that occur in the subtree of the pivot column.
        # reconstructed_pivot = mat[:, pivot_col_i]
        # cols_in_subtree = []
        # for (i, col_i) in enumerate(cols_sorted):
        #     if column_in_subtree(mat[:, col_i], reconstructed_pivot):
        #         cols_in_subtree.append(col_i)
        #         # Construct a new pivot restricting the view to undecided columns we've seen so far.
        #         new_pivot = reconstruct_pivot(cols_in_subtree, cols_sorted[:(i+1)], mat)
        #         # Make pivot be increasing by taking the union.
        #         reconstructed_pivot = np.logical_or(reconstructed_pivot, new_pivot).astype(int)
        # # Enforce that all rows in the subtree are supersets of the subtree root.
        # root = reconstruct_root(cols_in_subtree, mat, fnr)
        # for row_i in range(mat.shape[0]):
        #     if row_in_subtree(mat[row_i, :], root):
        #         mat[row_i, :] = np.logical_or(mat[row_i, :], root).astype(int)
        # # Reconstruct the pivot based on the columns in the subtree.
        # reconstructed_pivot = reconstruct_pivot(cols_in_subtree, cols_sorted, mat)

        def get_new_pivot(pivot_col_i, mat, cols_sorted, fnr):
            cols_in_subtree = find_subtree_columns(pivot_col_i, cols_sorted, mat)
            root = reconstruct_root(cols_in_subtree, mat, fnr)
            for row_i in range(mat.shape[0]):
                if row_in_subtree(mat[row_i, :], root):
                    mat[row_i, :] = np.logical_or(mat[row_i, :], root).astype(int)
                # else:
                    # mat[row_i, cols_in_subtree] = 0
            # Reconstruct the pivot based on the columns in the subtree.
            reconstructed_pivot = reconstruct_pivot(cols_in_subtree, cols_sorted, mat)
            return reconstructed_pivot, cols_in_subtree

        prev_reconstructed_pivot = mat[:, pivot_col_i]
        reconstructed_pivot, cols_in_subtree = get_new_pivot(pivot_col_i, mat, cols_sorted, fnr)
        while not np.all(prev_reconstructed_pivot == reconstructed_pivot):
            prev_reconstructed_pivot = reconstructed_pivot.copy()
            reconstructed_pivot, cols_in_subtree = get_new_pivot(pivot_col_i, mat, cols_sorted, fnr)

        if cols_in_subtree == []: # TODO: do we need this
            print('cols in subtree is empty')
            reconstructed_pivot = mat[:, pivot_col_i]

        # Go through all columns, enforcing structure with pivot_reconstructed
        final_cols_in_subtree = []
        for col_i in cols_sorted:
            if column_in_subtree(mat[:, col_i], reconstructed_pivot):
                mat[:, col_i] *= reconstructed_pivot
                final_cols_in_subtree.append(col_i)
            else:
                mat[:, col_i] *= 1 - reconstructed_pivot

        if final_cols_in_subtree == []:
            col_placement_i = pivot_col_i
            print('Empty final set')
        else:
            col_placement_i = count_based_max_likelihood(reconstructed_pivot, final_cols_in_subtree, mat)
        
        reconstruction[:, col_placement_i] = reconstructed_pivot

        cols_sorted = np.delete(cols_sorted, np.flatnonzero(cols_sorted == col_placement_i))

    # Row maximum likelihood
    reconstruction = column_max_likelihood_refinement(reconstruction.T, measured.T, fpr, fnr).T
    # Column maximum likelihood
    reconstruction = column_max_likelihood_refinement(reconstruction, measured, fpr, fnr)

    assert np.all(np.logical_or(reconstruction == 0, reconstruction == 1))
    return reconstruction

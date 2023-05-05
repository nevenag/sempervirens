import numpy as np

def column_in_subtree(col_a, col_b, fpr, fnr):
    """Returns whether col_a ocurrs in the subtree of col_b."""
    # Tries to determine if col_a is a subset of col_b.

    # k_00 = np.sum((1 - col_b) * (1 - col_a))
    k_01 = np.sum((1 - col_b) * col_a)
    # k_10 = np.sum(col_b * (1 - col_a))
    k_11 = np.sum(col_b * col_a)

    # True if col b is more likely to be a superset of col a than a subset of col a. 
    # supset_gt_subset = k_10 >= k_01
    # True if col b is more likely to be a superset of col a than disjoint from col a.
    # supset_gt_disjoint = k_11 * np.log((1-fnr)/fpr) >= k_01 * np.log((1-fpr)/fnr)

    old_decision = k_11 >= k_01
    # new_decision = supset_gt_subset and supset_gt_disjoint

    # if old_decision and not supset_gt_subset:
        # print("col b more likely to be subset of col a")
    return old_decision
    # return np.sum(col_a * col_b) >= np.sum(col_a * (1 - col_b))

def column_in_subtree2(col, rows_in_subtree, fpr, fnr):
    """Returns if an undecided column col is in in the subtree defined by rows_in_subtree."""
    k_00 = np.sum((1 - rows_in_subtree) * (1 - col))
    k_01 = np.sum((1 - rows_in_subtree) * col)
    k_10 = np.sum(rows_in_subtree * (1 - col))
    k_11 = np.sum(rows_in_subtree * col)
    decision = ((k_00 - k_10)*np.log((1-fpr)/(1-fnr)) + (k_11 - k_01)*np.log((1-fnr)/fnr)) >= 0
    # return decision and k_11 > 0
    return k_11 >= k_01 and k_11 > 0

def column_in_subtree4(col_a, col_b, fpr, fnr):
    """Returns whether col_a ocurrs in the subtree of col_b."""
    # Tries to determine if col_a is a subset of col_b.

    # k_00 = np.sum((1 - col_b) * (1 - col_a))
    k_01 = np.sum((1 - col_b) * col_a)
    k_10 = np.sum(col_b * (1 - col_a))
    k_11 = np.sum(col_b * col_a)

    # True if col b is more likely to be a superset of col a than a subset of col a. 
    supset_gt_subset = k_10 >= k_01
    # True if col b is more likely to be a superset of col a than disjoint from col a.
    supset_gt_disjoint = k_11 * np.log((1-fnr)/fpr) >= k_01 * np.log((1-fpr)/fnr)

    new_decision = supset_gt_subset and supset_gt_disjoint
    return new_decision

def find_subtree_columns(pivot_col, cols_sorted, mat, fpr, fnr):
    """Finds all columns that ocurr in the subtree of mat's pivot_col_i'th column."""
    cols_in_subtree = []
    for col_i in cols_sorted:
        col = mat[:, col_i]
        if column_in_subtree(col, pivot_col, fpr, fnr) or column_in_subtree(pivot_col, col, fpr, fnr):
            cols_in_subtree.append(col_i)
    return cols_in_subtree

def find_subtree_columns2(rows_in_subtree, cols_sorted, mat, fpr, fnr):
    """Finds all columns that ocurr in the subtree of mat's pivot_col_i'th column."""
    cols_in_subtree = []
    for col_i in cols_sorted:
        col = mat[:, col_i]
        if column_in_subtree2(col, rows_in_subtree, fpr, fnr):
            cols_in_subtree.append(col_i)
    return cols_in_subtree

def find_subtree_columns3(pivot_col, cols_sorted, mat, fpr, fnr):
    """Finds all columns that ocurr in the subtree of mat's pivot_col_i'th column."""
    mat_cols = mat[:, cols_sorted]
    K_11 = pivot_col @ mat_cols
    K_01 = (1 - pivot_col) @ mat_cols
    K_10 = pivot_col @ (1 - mat_cols)
    cols_mask = K_11 >= np.minimum(K_01, K_10)
    return np.array(cols_sorted)[cols_mask]

def find_subtree_columns4(pivot_col, cols_sorted, mat, fpr, fnr):
    """Finds all columns that ocurr in the subtree of mat's pivot_col_i'th column."""
    cols_in_subtree = []
    for col_i in cols_sorted:
        col = mat[:, col_i]
        if column_in_subtree4(col, pivot_col, fpr, fnr) or column_in_subtree4(pivot_col, col, fpr, fnr):
            cols_in_subtree.append(col_i)
    return cols_in_subtree

def reconstruct_root(cols_in_subtree, mat, fnr):
    """Reconstructs the root row for the subtree consisting of the columns in cols_in_subtree."""
    counts = np.sum(mat[:, cols_in_subtree], axis = 0)
    root = np.zeros(mat.shape[1])
    root[cols_in_subtree] = (counts >= np.max(counts)*(1 - fnr)).astype(int)
    return root
#
#def row_in_subtree(row_a, row_b):
#    """Returns whether row_a is in the subtree of row_b."""
#    return np.sum(row_a * row_b) >= np.sum(row_b) * (1/2)

def reconstruct_pivot(cols_in_subtree, cols_sorted, mat):
    """Reconstructs the pivot column based on the columns in its subtree."""
    # Count the number of times a row occurs in the subtree.
    subtree_in = mat[:, cols_in_subtree].sum(axis = 1)

    # Count the number of times a row occurs outside of the subtree but in undecided columns.
    out_set = np.setdiff1d(np.array(cols_sorted), np.array(cols_in_subtree), assume_unique = True)
    subtree_out = mat[:, out_set].sum(axis = 1) 

    # Set a row of the reconstruction to 1 if the row occurs more in the subtree than out of it.
    reconstructed_pivot = np.logical_and(subtree_in >= subtree_out, subtree_in > 0).astype(int)
    return reconstructed_pivot


def count_based_max_likelihood(pivot, col_set, mat):
    """Returns the index of the column in mat (of columns in col_set) that is most similar to pivot."""
    return col_set[np.argmax(pivot.T @ mat[:, col_set])]

def column_max_likelihood_refinement(reconstructed_mat, noisy_mat, fpr, fnr, mer):
    """
    Create and return a refined matrix from the columns in reconstructed_mat by, 
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


# TODO: true is only for testing. Remove after.
def reconstruct(measured, fpr, fnr, mer, true):
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
    orig = measured.copy()
    reconstruction = np.zeros_like(mat)
    mask = np.ones_like(mat)

    # Set all missing elements to 0s.
    mat[mat == 3] = 0
    orig[orig == 3] = 0

    col_set = np.array(range(mat.shape[1]))
    col_sums = np.sum(mat[:, col_set], axis = 0)
    assert np.all(col_sums.shape == (mat.shape[1],))
    cols_sorted = col_set[np.argsort(-col_sums)]
    # cols_sorted used as a sorted col_set from here on.

    while cols_sorted.size > 0:
        col_sums = np.sum(mat[:, cols_sorted], axis = 0)
        cols_sorted = cols_sorted[np.argsort(-col_sums)] # Sort from highest to lowest column sum.
        orig_pivot_col_i = cols_sorted[0] # Get the column with the most number of ones.
        pivot_col_i = orig_pivot_col_i

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

        def get_new_pivot(pivot, mat, cols_sorted, fpr, fnr):
            cols_in_subtree = find_subtree_columns3(pivot, cols_sorted, mat, fpr, fnr)
            rows_in_subtree = reconstruct_pivot(cols_in_subtree, cols_sorted, mat)
            root = reconstruct_root(cols_in_subtree, mat, fnr)
            for row_i in range(mat.shape[0]):
                if rows_in_subtree[row_i] == 1:
                    mat[row_i, :] = np.logical_or(mat[row_i, :], root).astype(int)
                else:
                    mat[row_i, cols_in_subtree] = 0
            # mat[np.ix_(np.where(1 - rows_in_subtree)[0], cols_in_subtree)] = 0
            reconstructed_pivot = reconstruct_pivot(cols_in_subtree, cols_sorted, mat)
            return reconstructed_pivot, cols_in_subtree

        prev_reconstructed_pivot = mat[:, pivot_col_i]
        pivot = mat[:, pivot_col_i]
        reconstructed_pivot, cols_in_subtree = get_new_pivot(pivot, mat, cols_sorted, fpr, fnr)
        while not np.all(prev_reconstructed_pivot == reconstructed_pivot):
            prev_reconstructed_pivot = reconstructed_pivot.copy()
            col_sums = np.sum(mat[:, cols_sorted], axis = 0)
            cols_sorted = cols_sorted[np.argsort(-col_sums)] # Sort from highest to lowest column sum.
            pivot_col_i = cols_sorted[0] # Get the column with the most number of ones.
            pivot = mat[:, pivot_col_i]
            reconstructed_pivot, cols_in_subtree = get_new_pivot(pivot, mat, cols_sorted, fpr, fnr)

        if cols_in_subtree == []: # TODO: do we need this
            print('cols in subtree is empty')
            reconstructed_pivot = mat[:, pivot_col_i]

        # Go through all columns, enforcing structure with pivot_reconstructed
        final_cols_in_subtree = find_subtree_columns3(reconstructed_pivot, cols_sorted, mat, fpr, fnr)
        undecided_cols_out_subtree = np.setdiff1d(np.array(cols_sorted), np.array(final_cols_in_subtree), assume_unique = True)
        mat[:, final_cols_in_subtree] *= reconstructed_pivot.reshape((-1, 1))
        mat[:, undecided_cols_out_subtree] *= 1 - reconstructed_pivot.reshape((-1, 1))

        if final_cols_in_subtree == []:
            col_placement_i = pivot_col_i
            print('Empty final set')
        else:
            col_placement_i = count_based_max_likelihood(reconstructed_pivot, final_cols_in_subtree, mat)

        reconstruction[:, col_placement_i] = reconstructed_pivot
        mask[:, col_placement_i] = reconstruction[:, col_placement_i]

        # TODO: remove testing.
        # TESTING
        # Assuming the input is the ground truth, compare the final_cols_in_subtree to the true cols in subtree.
#        true_cols_in_subtree_mask = np.zeros(true.shape[1])
#        true_cols_in_subtree = []
#        for col_i in cols_sorted:
#            if _true_subset(true[:, col_i], true[:, orig_pivot_col_i]):
#                true_cols_in_subtree.append(col_i)
#                true_cols_in_subtree_mask[col_i] = 1
#        final_cols_in_subtree_mask = np.zeros_like(true_cols_in_subtree_mask)
#        final_cols_in_subtree_mask[final_cols_in_subtree] = 1
#        completeness_ratio_cols = np.sum(final_cols_in_subtree_mask * true_cols_in_subtree_mask) / np.sum(true_cols_in_subtree_mask)
#        excess_ratio_cols = len(set(final_cols_in_subtree) - set(true_cols_in_subtree)) / len(final_cols_in_subtree)
#        true_rows_in_subtree = (np.sum(true[:, true_cols_in_subtree], 1) > 0).astype(int)
#        completeness_ratio_rows = np.sum(reconstructed_pivot * true_rows_in_subtree) / np.sum(true_rows_in_subtree)
#        excess_ratio_rows = len(set(np.where(reconstructed_pivot)[0]) - set(np.where(true_rows_in_subtree)[0])) / np.sum(reconstructed_pivot)
#        if completeness_ratio_cols < 1.0-1e-3 or excess_ratio_cols > 1e-3 or completeness_ratio_rows < 1.0-1e-3 or excess_ratio_rows > 1e-3:
#            # print(f'Cols in subtree size: {len(true_cols_in_subtree)},    row size: {np.sum(true_rows_in_subtree)}')
#            # print(f'cols: completeness (bad [0, 1] good): {completeness_ratio_cols},    excess (good [0, 1] bad): {excess_ratio_cols}')
#            # print(f'rows: completeness (bad [0, 1] good): {completeness_ratio_rows},    excess (good [0, 1] bad): {excess_ratio_rows}')
#            # print(f'rows: missing: {np.sum(true_rows_in_subtree) - np.sum(true_rows_in_subtree * reconstructed_pivot)}')
#            # missing_rows_mask = true_rows_in_subtree * (1 - reconstructed_pivot)
##            print(true[np.ix_(np.where(missing_rows_mask)[0], cols_sorted)])
##            print(true[np.ix_(np.where(missing_rows_mask)[0], true_cols_in_subtree)])
##            print(np.sum(true[np.ix_(np.where(missing_rows_mask)[0], true_cols_in_subtree)], 1))
##            print(np.sum(true[np.ix_(np.where(missing_rows_mask)[0], list(set(cols_sorted) - set(true_cols_in_subtree)))], 1))
#            # print(np.where(missing_rows_mask)[0])
#            # exit()

        # END TESTING

        cols_sorted = np.delete(cols_sorted, np.flatnonzero(cols_sorted == col_placement_i))

    # Row maximum likelihood
    reconstruction = column_max_likelihood_refinement(reconstruction.T, measured.T, fpr, fnr, mer).T
    # Column maximum likelihood
    reconstruction = column_max_likelihood_refinement(reconstruction, measured, fpr, fnr, mer)

    assert np.all(np.logical_or(reconstruction == 0, reconstruction == 1))
    return reconstruction

def _true_subset(a, b):
    """Checks that 'a' is a subset of 'b'"""
    return np.all(a * b == a)

def _true_disjoint(a, b):
    """Checks that 'a' and 'b' are disjoint"""
    return np.all(a * b == 0)


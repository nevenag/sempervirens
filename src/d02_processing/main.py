import numpy as np
import scipy.stats

# import modin.pandas as pd
# import ray
# ray.init()
# print('done')
import pandas as pd


import metrics

def read_data(filename):     # reads a matrix from file and returns it in BOOL type
    ds = pd.read_csv(filename, sep='\t', index_col=0)
    M = ds.values
    return M 

def prob_superset(p_i, p):
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(1-(1-p_i)*p, axis=0)

def prob_equal(p_i, p):
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(p_i*p + (1-p_i)*(1-p), axis=0)

def prob_subset(p_i, p):
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(1-(1-p)*p_i, axis=0)

def prob_empty_intersect(p_i, p):
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(1-p_i*p, axis=0)

def pick_pivot(prob_mat):
    # Return (pivot, index in prob_mat)
    
    # This picks column that maximizes the expected number of 1s
    expected_num_ones = np.sum(prob_mat, axis=0)
    pivot_idx = np.argmax(expected_num_ones)
    pivot = prob_mat[:, pivot_idx].reshape((prob_mat.shape[0], 1))
    return (pivot, pivot_idx)

def forward_pass(pivot, prob_mat):
    disjoints = np.zeros((1, prob_mat.shape[1]))
    num_rows = prob_mat.shape[0]
    column_map = np.array(range(prob_mat.shape[1])) # column_map[i] gives index of column in original prob_mat

    # union_count = 0
    while prob_mat.shape[1] > 0: # While columns remaining.
        p_eq = prob_equal(pivot, prob_mat)
        p_sup = prob_superset(pivot, prob_mat) - p_eq
        p_sub = prob_subset(pivot, prob_mat) - p_eq
        p_empty_intersect = prob_empty_intersect(pivot, prob_mat)
        denom = p_eq + p_sup + p_sub + p_empty_intersect
        p_eq /= denom
        p_sup /= denom
        p_sub /= denom
        p_empty_intersect /= denom
        joined = np.vstack((p_sup, p_eq, p_sub, p_empty_intersect))
        assert np.all(joined <= 1) and np.all(0 <= joined)
        assert np.all(np.isclose(np.sum(joined, axis=0), 1))
        entropy = scipy.stats.entropy(joined, base=4, axis=0)
        min_idx = np.argmin(entropy)
        max_prob_idx = np.argmax(joined[:, min_idx])
        min_entropy_col = prob_mat[:, min_idx].reshape((num_rows, 1))
        if max_prob_idx == 0: # sup
            pass
        elif max_prob_idx == 1: # eq # ISSUE: this never seems to be largest...
            pivot = 1 - (1-pivot)*(1-min_entropy_col)
        elif max_prob_idx == 2: # sub
            pivot = min_entropy_col
        else: # empty intersection
            disjoints[0, column_map[min_idx]] = 1
        assert pivot.shape == (prob_mat.shape[0], 1)

        # p_empty_intersect = prob_empty_intersect(pivot, prob_mat)
        # p_other = 1 - p_empty_intersect
        # joined = np.vstack((p_empty_intersect, p_other))
        # entropy = scipy.stats.entropy(joined, base=2, axis=0)
        # min_idx = np.argmin(entropy)
        # if joined[0, min_idx] > joined[1, min_idx]: # Probability of intersection being empty is greater.
        #     disjoints[0, column_map[min_idx]] = 1
        # else:
        #     # union_count += 1
        #     min_entropy_col = prob_mat[:, min_idx].reshape((num_rows, 1))
        #     pivot = 1 - (1-pivot)*(1-min_entropy_col) # Probabilistic union
        #     # pivot = pivot*min_entropy_col
        #     assert pivot.shape == (prob_mat.shape[0], 1)
        prob_mat = np.delete(prob_mat, min_idx, 1)
        column_map = np.delete(column_map, min_idx)
    subsets = 1 - disjoints
    return disjoints, subsets, pivot

def determine_col(disjoints, subsets, prob_mat, pivot_union, fpr, fnr, true_prior):
    assert disjoints.shape == (1, prob_mat.shape[1])
    assert subsets.shape == disjoints.shape
    assert np.all(disjoints == 1 - subsets)
    # print('determine col')
    # print(f"disjoints: {disjoints}, subsets: {subsets}")
    # print(f"prob mat: \n{prob_mat}")

    # prob_change_disjoints = 1 - np.prod(1 - prob_mat*disjoints, axis=1)
    # prob_change_subsets   = 1 - np.prod(1 - prob_mat*subsets, axis=1)
    # print('prob change disjoints', prob_change_disjoints)
    # print('prob change subsets', prob_change_subsets)

    # p10 = fnr*true_prior[1] / (fnr*true_prior[1] + (1-fpr)*true_prior[0])
    # p11 = 1 - (fpr*true_prior[0])/(fpr*true_prior[0]+(1-fnr)*true_prior[1])
    # all0s_p = 1 - np.power(1 - p10, np.sum(subsets) + 1)
    # all1s_p = 1 - np.power(1 - p11, np.sum(subsets) + 1)
    # print(f'p10: {p10}, p11: {p11}, {(all0s_p + all1s_p)/2}, {all0s_p}, {1/2}')
    # thresh = all1s_p
    # thresh = (all0s_p + all1s_p)/2
    # thresh = all0s_p
    thresh = 1/2

    pr1_d = np.prod(1 - prob_mat * disjoints, axis=1)
    pr1_s = pivot_union.reshape((len(pivot_union),))
    true_col = (pr1_d*pr1_s > thresh).reshape((prob_mat.shape[0], 1))

    # true_col = (prob_change_disjoints <= prob_change_subsets).reshape((prob_mat.shape[0], 1))
    # print('true col', true_col)
    return true_col.astype(int)
    
def backward_pass(prob_mat, true_col, disjoints, subsets):
    assert true_col.shape == (prob_mat.shape[0], 1)
    assert disjoints.shape == (1, prob_mat.shape[1])
    assert subsets.shape == disjoints.shape

    prob_mat = prob_mat*disjoints*(1-true_col) + prob_mat*subsets*true_col
    return prob_mat

def prob_measure(x, Y, fpr, fnr):
    # Probability of measuring vector x given true vector y_i with Y = [y_1 ...]
    assert x.shape[1] == 1
    p00 = 1-fpr
    p01 = fpr
    p10 = fnr
    p11 = 1-fnr
    element_probs = (1-x)*(1-Y)*p00 + x*(1-Y)*p01 + (1-x)*Y*p10 + x*Y*p11
    assert element_probs.shape == Y.shape
    probs = np.prod(element_probs, axis=0)
    assert probs.shape == (Y.shape[1],)
    return probs

def reconstruct(measured, fpr, fnr, true_prior):
    measured_prior = np.array([(1-fpr)*true_prior[0] + fnr*true_prior[1], 0])
    measured_prior[1] = 1 - measured_prior[0]

    prob_mat = measured*((1-fnr)*true_prior[1]/measured_prior[1]) + (1-measured)*(1-(1-fpr)*true_prior[0]/measured_prior[0])

    completed_cols = np.empty_like(measured)
    completed_cols_i = 0

    while prob_mat.shape[1] > 1: # While more than 1 column remaining.
        pivot, pivot_idx = pick_pivot(prob_mat)
        prob_mat = np.delete(prob_mat, pivot_idx, 1)

        # print(f'pivot: \n{pivot}')
        disjoints, subsets, pivot_union = forward_pass(pivot, prob_mat)

        # print(f'pivot after unions: \n{pivot_union}')
        true_col = determine_col(disjoints, subsets, prob_mat, pivot_union, fpr, fnr, true_prior)
        # print(f'01d pivot: \n{true_col}')
        prob_mat = backward_pass(prob_mat, true_col, disjoints, subsets)

        completed_cols[:, completed_cols_i] = true_col.reshape((true_col.shape[0],))
        completed_cols_i += 1

    completed_cols[:, -1] = np.around(prob_mat).reshape((prob_mat.shape[0],))

    reconstructed = np.empty_like(measured)
    for i in range(measured.shape[1]):
        prob_measured = prob_measure(measured[:, i].reshape((measured.shape[0], 1)), completed_cols, fpr, fnr)
        max_idx = np.argmax(prob_measured)
        reconstructed[:, i] = completed_cols[:, max_idx]
        completed_cols = np.delete(completed_cols, max_idx, 1)

    return reconstructed


# Test with actual data
file_name  = "simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000"
# file_name = "simNo_10-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000"
true_data_filename = "data/" + file_name + ".SC.before_FP_FN_NA"
measured_data_filename = "data/" + file_name + ".SC"
true_data = read_data(true_data_filename)
measured_data = read_data(measured_data_filename)
# Set missing entries in measured data to true values (only considering false positives and negatives right now).
missing_entries = measured_data == 3
measured_data[missing_entries] = true_data[missing_entries]

fpr = 0.001 # False positive rate.
fnr = 0.2 # False negative rate.
# Priors
true_prior = np.array([1/2, 0])
true_prior[1] = 1 - true_prior[0]

reconstructed = reconstruct(measured_data, fpr, fnr, true_prior)

print(f"Number of elements in matrix: {true_data.shape[0] * true_data.shape[1]}")
metrics.isPtree(reconstructed)
metrics.compareAD(true_data, reconstructed)
metrics.compareDF(true_data, reconstructed)
print(f"Number of differences: {metrics.num_diffs(true_data, reconstructed)}")

# # for basic testing

# rng = np.random.default_rng()

# def corrupt(M, fpr, fnr):
#     corrupted = np.copy(M)
#     randoms = rng.uniform(size=M.shape)
#     for i in range(M.shape[0]):
#         for j in range(M.shape[1]):
#             if M[i,j] == 0:
#                 if randoms[i,j] <= fpr:
#                     corrupted[i,j] = 1
#             else:
#                 if randoms[i,j] <= fnr:
#                     corrupted[i,j] = 0
#     return corrupted

# N = 5
# M = 2*N

# fpr = 0.000 # False positive rate.
# fnr = 0.2  # False negative rate.

# # Priors
# true_prior = np.array([1/2, 0])
# true_prior[1] = 1 - true_prior[0]

# # true_data = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
# # true_data = np.zeros((N, M))
# # true_data[0, M-1] = 1
# true_data = np.triu(np.ones((N, N)))
# # true_data = np.array(np.bmat([[true_data, np.zeros((N, N))], [np.zeros((N, N)), true_data]]))
# # true_data = np.hstack((true_data, true_data))
# # true_data = np.hstack((true_data, true_data))
# # measured = true_data

# # measured = np.array([
# #     [1, 1, 1, 0, 0, 0], 
# #     [0, 1, 1, 0, 0, 0],
# #     [0, 0, 1, 0, 0, 0],
# #     [0, 0, 0, 1, 0 ,0],
# #     [0, 0, 0, 0, 1, 1],
# #     [0, 0, 0, 0, 0, 1.0]
# # ])
# measured = corrupt(true_data, fpr, fnr)

# print(f"Measured: \n{measured}")

# reconstructed = reconstruct(measured, fpr, fnr, true_prior)
# print(f"Reconstructed: \n{reconstructed}")

# # print(metrics.isPtree(reconstructed_cols))

# print(f"Number of elements in matrix: {true_data.shape[0] * true_data.shape[1]}")
# metrics.isPtree(reconstructed)
# metrics.compareAD(true_data, reconstructed)
# metrics.compareDF(true_data, reconstructed)
# print(f"Number of differences: {metrics.num_diffs(true_data, reconstructed)}")
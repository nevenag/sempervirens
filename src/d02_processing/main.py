from calendar import c
import numpy as np
import scipy.stats

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
    # TODO: pick pivot 
    # Return (pivot, index in prob_mat)
    
    # This picks column that maximizes the expected number of 1s
    expected_num_ones = np.sum(prob_mat, axis=0)
    pivot_idx = np.argmax(expected_num_ones)
    pivot = prob_mat[:, pivot_idx].reshape((prob_mat.shape[0], 1))
    return (pivot, pivot_idx)

def forward_pass(pivot, prob_mat):
    disjoints = np.zeros((1, prob_mat.shape[1]))
    num_rows = prob_mat.shape[0]

    while prob_mat.shape[1] > 0: # While columns remaining.
        p_empty_intersect = prob_empty_intersect(pivot, prob_mat)
        p_other = 1 - p_empty_intersect
        joined = np.vstack((p_empty_intersect, p_other))
        entropy = scipy.stats.entropy(joined, base=2, axis=0)
        min_idx = np.argmin(entropy)
        if joined[0, min_idx] > joined[1, min_idx]: # Probability of intersection being empty is greater.
            disjoints[0, min_idx] = 1
        else:
            min_entropy_col = prob_mat[:, min_idx].reshape((num_rows, 1))
            pivot = 1 - (1-pivot)*(1-min_entropy_col) # Probabilistic union
            assert pivot.shape == (prob_mat.shape[0], 1)
        prob_mat = np.delete(prob_mat, min_idx, 1)

    subsets = 1 - disjoints
    return disjoints, subsets

def determine_col(disjoints, subsets, prob_mat):
    assert disjoints.shape == (1, prob_mat.shape[1])
    assert subsets.shape == disjoints.shape

    prob_all_disjoints_0 = np.prod(1 - prob_mat*disjoints, axis=1)
    prob_all_subsets_0   = np.prod(1 - prob_mat*subsets,   axis=1)

    true_col = (prob_all_disjoints_0 > prob_all_subsets_0).reshape((prob_mat.shape[0], 1))
    return true_col.astype(int)
    
def backward_pass(prob_mat, true_col, disjoints, subsets):
    assert true_col.shape == (prob_mat.shape[0], 1)
    prob_mat = prob_mat*disjoints*(1-true_col) + prob_mat*subsets*true_col
    return prob_mat

def reconstruct(measured, fpr, fnr, true_prior):
    measured_prior = np.array([(1-fpr)*true_prior[0] + fnr*true_prior[1], 0])
    measured_prior[1] = 1 - measured_prior[0]

    prob_mat = measured*((1-fnr)*true_prior[1]/measured_prior[1]) + (1-measured)*(1-(1-fpr)*true_prior[0]/measured_prior[0])

    completed_cols = []
    
    while prob_mat.shape[1] > 1: # While more than 1 column remaining.
        print(prob_mat)
        pivot, pivot_idx = pick_pivot(prob_mat)
        prob_mat = np.delete(prob_mat, pivot_idx, 1)

        disjoints, subsets = forward_pass(pivot, prob_mat)

        true_col = determine_col(disjoints, subsets, prob_mat)

        prob_mat = backward_pass(prob_mat, true_col, disjoints, subsets)

        completed_cols += [true_col]

    print(prob_mat)
    # TODO: last column
    completed_cols.append(np.around(prob_mat))

    return completed_cols

# for basic testing
rng = np.random.default_rng()

N = 3
M = 3

fpr = 0.01 # False positive rate.
fnr = 0.01 # False negative rate.

# Priors
true_prior = np.array([1/2, 0])
true_prior[1] = 1 - true_prior[0]

measured = rng.integers(low=0, high=1, endpoint=True, size=(N, M))

reconstructed_cols = reconstruct(measured, fpr, fnr, true_prior)
print(np.hstack(reconstructed_cols))
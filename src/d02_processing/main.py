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
    column_map = np.array(range(prob_mat.shape[1])) # column_map[i] gives index of column in original prob_mat

    # print("Forward Pass")
    # print(f"Prob mat: \n{prob_mat}")
    # print(f"Pivot: \n{pivot}")

    while prob_mat.shape[1] > 0: # While columns remaining.
        p_empty_intersect = prob_empty_intersect(pivot, prob_mat)
        p_other = 1 - p_empty_intersect
        joined = np.vstack((p_empty_intersect, p_other))
        entropy = scipy.stats.entropy(joined, base=2, axis=0)
        min_idx = np.argmin(entropy)
        if joined[0, min_idx] > joined[1, min_idx]: # Probability of intersection being empty is greater.
            disjoints[0, column_map[min_idx]] = 1
        else:
            min_entropy_col = prob_mat[:, min_idx].reshape((num_rows, 1))
            pivot = 1 - (1-pivot)*(1-min_entropy_col) # Probabilistic union
            assert pivot.shape == (prob_mat.shape[0], 1)
            # print(f'Min entropy col: {min_idx}, {min_entropy_col}')
        prob_mat = np.delete(prob_mat, min_idx, 1)
        column_map = np.delete(column_map, min_idx)

        # print('disjoints, column map, pivot', disjoints, column_map, pivot)
    subsets = 1 - disjoints
    return disjoints, subsets

def determine_col(disjoints, subsets, prob_mat):
    assert disjoints.shape == (1, prob_mat.shape[1])
    assert subsets.shape == disjoints.shape
    assert np.all(disjoints == 1 - subsets)
    # print('determine col')
    # print(f"disjoints: {disjoints}, subsets: {subsets}")
    # print(f"prob mat: \n{prob_mat}")

    prob_change_disjoints = 1 - np.prod(1-prob_mat*disjoints, axis=1)
    prob_change_subsets   = 1 - np.prod(1 - prob_mat*subsets,   axis=1)
    # print('prob change disjoints', prob_change_disjoints)
    # print('prob change subsets', prob_change_subsets)

    true_col = (prob_change_disjoints <= prob_change_subsets).reshape((prob_mat.shape[0], 1))
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
    # print(x)
    # print(Y)
    # print('element probs')
    # print(element_probs)
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

        disjoints, subsets = forward_pass(pivot, prob_mat)

        true_col = determine_col(disjoints, subsets, prob_mat)

        prob_mat = backward_pass(prob_mat, true_col, disjoints, subsets)

        completed_cols[:, completed_cols_i] = true_col.reshape((true_col.shape[0],))
        completed_cols_i += 1

    # TODO: last column
    completed_cols[:, -1] = np.around(prob_mat).reshape((prob_mat.shape[0],))
    print('Completed Cols')
    print(completed_cols)

    reconstructed = np.empty_like(measured)
    for i in range(measured.shape[1]):
        prob_measured = prob_measure(measured[:, i].reshape((measured.shape[0], 1)), completed_cols, fpr, fnr)
        # print(measured[:, i].reshape((measured.shape[0], 1)))
        # print(completed_cols)
        # print(prob_measured)
        max_idx = np.argmax(prob_measured)
        reconstructed[:, i] = completed_cols[:, max_idx]
        completed_cols = np.delete(completed_cols, max_idx, 1)

    return reconstructed

# for basic testing
rng = np.random.default_rng()

N = 4
M = 4

fpr = 0.01 # False positive rate.
fnr = 0.01 # False negative rate.

# Priors
true_prior = np.array([1/2, 0])
true_prior[1] = 1 - true_prior[0]

measured = rng.integers(low=0, high=1, endpoint=True, size=(N, M))
# measured = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
# measured = np.array([[1, 1], [1, 1], [0, 0]])

print(f"Measured: \n{measured}")

reconstructed_cols = reconstruct(measured, fpr, fnr, true_prior)
print(f"Reconstructed: \n{reconstructed_cols}")
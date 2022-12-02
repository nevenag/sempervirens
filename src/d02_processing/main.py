from tracemalloc import start
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

def prob_equal(p_i, p): 
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(p_i*p + (1-p_i)*(1-p), axis=0)
    # return np.exp(np.sum(np.log(p_i*p + (1-p_i)*(1-p)), axis=0))

def prob_superset(p_i, p): # supseteq
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(1-(1-p_i)*p, axis=0)
    # return np.exp(np.sum(np.log(1-(1-p_i)*p), axis=0))

def prob_subset(p_i, p): # subseteq
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(1-(1-p)*p_i, axis=0)
    # return np.exp(np.sum(np.log(1-(1-p)*p_i), axis=0))

def prob_disjoint(p_i, p):
    assert p_i.shape[1] == 1 and p_i.shape[0] == p.shape[0]
    return np.prod(1-p_i*p, axis=0)
    # return np.exp(np.sum(np.log(1-p_i*p), axis=0))

def pick_pivot(prob_mat, available_cols):
    # Return (pivot, index in prob_mat)
    
    # This picks column that maximizes the expected number of 1s of the available columns
    # Probably a smarter way to ensure that when matrix is all 0s, available columns are not picked
    prob_mat = prob_mat * available_cols  - np.ones_like(prob_mat)*(1-available_cols)
    expected_num_ones = np.sum(prob_mat, axis=0)

    # Of the ones that have the same number of expected ones, pick one that is a superset of others.
    max_val = np.max(expected_num_ones)
    start_idx = np.argmax(expected_num_ones) # Get the first index with the most expected ones.
    assert available_cols[start_idx] == 1
    pivot_candidate_idx = start_idx
    for i in range(start_idx + 1, prob_mat.shape[1]):
        if expected_num_ones[i] != max_val or not available_cols[i]:
            pass
        pivot = prob_mat[:, pivot_candidate_idx].reshape((prob_mat.shape[0], 1))
        new = prob_mat[:, i].reshape((prob_mat.shape[0], 1))

        p_eq = prob_equal(pivot, new)
        p_strict_sup = prob_superset(pivot, new) - p_eq
        p_strict_sub = prob_subset(pivot, new) - p_eq
        p_dis = prob_disjoint(pivot, new)
        denom = p_eq + p_strict_sup + p_strict_sub + p_dis
        p_eq /= denom
        p_strict_sup /= denom
        p_strict_sub /= denom
        p_dis /= denom

        p_sub = p_strict_sub + p_eq
        if p_strict_sup < p_sub and p_dis < p_sub:
            pivot_candidate_idx = i
    pivot_idx = pivot_candidate_idx

    # pivot_idx = np.argmax(expected_num_ones)
    pivot = prob_mat[:, pivot_idx].reshape((prob_mat.shape[0], 1))
    return (pivot, pivot_idx)


# def forward_pass(pivot, prob_mat):
#     disjoints = np.zeros((1, prob_mat.shape[1]))
#     num_rows = prob_mat.shape[0]
#     column_map = np.array(range(prob_mat.shape[1])) # column_map[i] gives index of column in original prob_mat

#     subtree = np.zeros((1, prob_mat.shape[1]))

#     # union_count = 0
#     while prob_mat.shape[1] > 0: # While columns remaining.
#         p_eq = prob_equal(pivot, prob_mat)
#         p_strict_sup = prob_superset(pivot, prob_mat) - p_eq
#         p_strict_sub = prob_subset(pivot, prob_mat) - p_eq
#         p_dis = prob_disjoint(pivot, prob_mat)
#         denom = p_eq + p_strict_sup + p_strict_sub + p_dis
#         p_eq /= denom
#         p_strict_sup /= denom
#         p_strict_sub /= denom
#         p_dis /= denom
#         joined = np.vstack((p_strict_sup  + p_eq, p_strict_sub, p_dis))
#         assert np.all(joined <= 1) and np.all(0 <= joined)
#         assert np.all(np.isclose(np.sum(joined, axis=0), 1))

#         entropy = scipy.stats.entropy(joined, base=3, axis=0)
#         min_idx = np.argmin(entropy)
#         min_entropy_col = prob_mat[:, min_idx].reshape((num_rows, 1))
#         max_prob_idx = np.argmax(joined[:, min_idx])
#         if max_prob_idx == 0: # sup
#             subtree[0, column_map[min_idx]] = 1
#         elif max_prob_idx == 1: # strict sub
#             # pivot = 1 - (1-pivot)*(1-min_entropy_col)

#         elif max_prob_idx == 2: # sub
#             pivot = min_entropy_col
#         else: # empty intersection
#             disjoints[0, column_map[min_idx]] = 1
#         assert pivot.shape == (prob_mat.shape[0], 1)
#         prob_mat = np.delete(prob_mat, min_idx, 1)
#         column_map = np.delete(column_map, min_idx)
#     subsets = 1 - disjoints
#     return disjoints, subsets, pivot

# def determine_col(disjoints, subsets, prob_mat, pivot_union, fpr, fnr, true_prior):
#     assert disjoints.shape == (1, prob_mat.shape[1])
#     assert subsets.shape == disjoints.shape
#     assert np.all(disjoints == 1 - subsets)
#     thresh = 1/2

#     pr1_d = np.prod(1 - prob_mat * disjoints, axis=1)
#     pr1_s = pivot_union.reshape((len(pivot_union),))
#     true_col = (pr1_d*pr1_s > thresh).reshape((prob_mat.shape[0], 1))

#     # true_col = (prob_change_disjoints <= prob_change_subsets).reshape((prob_mat.shape[0], 1))
#     # print('true col', true_col)
#     return true_col.astype(int)

def find_subtree(pivot, prob_mat, available_cols):
    subtree = np.zeros(prob_mat.shape[1])
    changed_pivot = True
    while changed_pivot:
        changed_pivot = False

        p_eq = prob_equal(pivot, prob_mat)
        p_strict_sup = prob_superset(pivot, prob_mat) - p_eq
        p_strict_sub = prob_subset(pivot, prob_mat) - p_eq
        p_dis = prob_disjoint(pivot, prob_mat)
        denom = p_eq + p_strict_sup + p_strict_sub + p_dis
        p_eq /= denom
        p_strict_sup /= denom
        p_strict_sub /= denom
        p_dis /= denom
        joined = np.vstack((p_strict_sup  + p_eq, p_strict_sub, p_dis))
        assert np.all(joined <= 1) and np.all(0 <= joined)
        assert np.all(np.isclose(np.sum(joined, axis=0), 1))

        entropy = scipy.stats.entropy(joined, base=3, axis=0)
        for (i, _) in sorted(enumerate(entropy), key=lambda x: x[1]):
            if subtree[i] == 1 or not available_cols[i]:
                continue
            prob_max = np.argmax(joined[:, i])
            if prob_max == 0: # sup
                subtree[i] = 1
            elif prob_max == 1: # strict sub
                subtree[i] = 1
                pivot = prob_mat[:, i].reshape((prob_mat.shape[0], 1))
                changed_pivot = True
                break
            else: # disjoint
                pass
    return subtree

def reconstruct_root(pivot, subtree, prob_mat, available_cols):
    pr1 = np.log(1 - pivot).reshape((pivot.shape[0],))
    pr1 += np.sum(np.log(1 - prob_mat*subtree), axis=1)
    pr1 = 1 - np.exp(pr1)

    disjoints = (1 - subtree)*available_cols
    assert np.all(disjoints.shape == subtree.shape)
    pr0 = 1 - np.exp(np.sum(np.log(1 - prob_mat*disjoints), axis=1))
    
    # print(' ')
    # print(pr1)
    # print(pr0)
    # root = (pr1 >= pr0).reshape((prob_mat.shape[0], 1)).astype(int)
    # root = (pr1*(1-pr0) >= (1-pr1)*pr0).reshape((prob_mat.shape[0], 1)).astype(int) # issue: remember discarding some probs
    root = np.logical_and(pr1*(1-pr0) >= (1-pr1)*pr0, pr1*(1-pr0) >= (1-pr1)*(1-pr0)).reshape((prob_mat.shape[0], 1)).astype(int) # issue: remember discarding some probs
    # root = np.logical_and(pr1 >= pr0, pr1*(1-pr0) >= 1/2).reshape((prob_mat.shape[0], 1)).astype(int)
    # what if do some ands of different conditions
    return root

def backward_pass(prob_mat, true_col, subtree, available_cols):
    assert true_col.shape == (prob_mat.shape[0], 1)
    assert subtree.shape == (prob_mat.shape[1],)

    prob_mat = prob_mat*(1-subtree)*available_cols*(1-true_col) + prob_mat*subtree*true_col
    return prob_mat

def prob_measure(x, Y, fpr, fnr): 
    # Log of probability of measuring vector x given true vector y_i with Y = [y_1 ...]
    assert x.shape[1] == 1
    p00 = 1-fpr
    p01 = fpr
    p10 = fnr
    p11 = 1-fnr
    element_probs = (1-x)*(1-Y)*p00 + x*(1-Y)*p01 + (1-x)*Y*p10 + x*Y*p11
    assert element_probs.shape == Y.shape
    # probs = np.prod(element_probs, axis=0)
    probs = np.sum(np.log(element_probs), axis=0)
    assert probs.shape == (Y.shape[1],)
    return probs

def mlrefinementcol(M,M_noisy,p01,p10):  # this is for non missing element case M is reconstructed matrix, M_noisy is the noisy input
    M_output=M_noisy.copy()
    # print(M,M_noisy)
    for i in range(M_output.shape[1]):
        p=[]
        for j in range(M_output.shape[1]):
            #print(i,j,np.sum(M_output[:,i]>M[:,j])*np.log(p01)+np.sum(M_output[:,i]<M[:,j])*np.log(p10))
            p.append(np.sum(M_output[:,i]>M[:,j])*np.log(p01)+np.sum(M_output[:,i]<M[:,j])*np.log(p10))
    #print(M_output[:,i],M[:,np.argmax(p)], " are matched" )
        # print(i,np.argmax(p), " are matched" )
    
        M_output[:,i]=M[:,np.argmax(p)]
    
    return M_output

def reconstruct(measured, fpr, fnr):
    # measured_frac1 = np.sum(measured, axis=0) / measured.shape[0]
    # pi1o1 = measured_frac1*(1-fnr)
    # pi1o0 = measured_frac1*fnr
    # pi0o1 = (1-measured_frac1)*fpr
    # pi0o0 = (1-measured_frac1)*(1-fpr)
    # prob_mat = measured * (pi1o1/(pi1o1 + pi0o1)) + (1 - measured) * (pi1o0/(pi1o0 + pi0o0))
    # # print(prob_mat)

    # Estimate prior column by column
    # measured_num0 = np.sum(1 - measured, axis=0)
    # measured_num1 = np.sum(measured, axis=0)
    # est_true_num1 = 1/(fpr*fnr - (1-fpr)*(1-fnr)) * (fpr*measured_num0 - (1-fpr)*measured_num1)
    # est_true_num0 = 1/(1-fpr) * (measured_num0 - fnr*est_true_num1)
    # prob_mat = measured * ((1-fnr)*est_true_num1/measured_num1) + (1-measured)*(1-(1-fpr)*est_true_num0/measured_num0)
    # assert not np.any(np.isnan(prob_mat))
    # prob_mat = np.clip(prob_mat, 0, 1)

    # Estimate prior over whole matrix (TODO: double check equiv to Can's code)
    measured_num0 = np.sum(1-measured)
    measured_num1 = np.sum(measured)
    est_true_num1 = 1/(fpr*fnr - (1-fpr)*(1-fnr)) * (fpr*measured_num0 - (1-fpr)*measured_num1)
    est_true_num0 = 1/(1-fpr) * (measured_num0 - fnr*est_true_num1)
    prob_mat = measured * ((1-fnr)*est_true_num1/measured_num1) + (1-measured)*(1-(1-fpr)*est_true_num0/measured_num0)
    assert not np.any(np.isnan(prob_mat))
    prob_mat = np.clip(prob_mat, 0, 1)
    
    # print(prob_mat)
    assert np.all(prob_mat >= 0)
    assert np.all(prob_mat <= 1)

    # measured_prior = np.array([(1-fpr)*true_prior[0] + fnr*true_prior[1], 0])
    # measured_prior[1] = 1 - measured_prior[0]
    # prob_mat = measured*((1-fnr)*true_prior[1]/measured_prior[1]) + (1-measured)*(1-(1-fpr)*true_prior[0]/measured_prior[0])

    completed_cols = np.empty_like(measured)
    completed_cols_i = 0
    available_cols = np.ones(prob_mat.shape[1])

    # while prob_mat.shape[1] > 1: # While more than 1 column remaining.
    while np.sum(available_cols) > 1:
        pivot, pivot_idx = pick_pivot(prob_mat, available_cols)
        # prob_mat = np.delete(prob_mat, pivot_idx, 1) # figure out which column to remove / replace based on the reconstructed pivot
        assert available_cols[pivot_idx]
        # available_cols[pivot_idx] = 0

        # print(f'pivot: {pivot_idx}, \n{pivot}')
        subtree = find_subtree(pivot, prob_mat, available_cols)

        # print(f'subtree after unions: \n{subtree}')
        reconstructed_root = reconstruct_root(pivot, subtree, prob_mat, available_cols)
        # print(f'01d pivot: \n{reconstructed_root}')

        p00 = 1-fpr
        p01 = fpr
        p10 = fnr
        p11 = 1-fnr
        element_probs = (1-reconstructed_root)*(1-measured)*p00 + (1-reconstructed_root)*measured*p01 + reconstructed_root*(1-measured)*p10 + reconstructed_root*measured*p11
        probs = np.exp(np.sum(np.log(element_probs), axis=0))
        probs = probs*available_cols
        max_idx = np.argmax(probs)
        assert available_cols[max_idx]
        # available_cols[max_idx] = 0

        available_cols[pivot_idx] = 0

        prob_mat = backward_pass(prob_mat, reconstructed_root, subtree, available_cols)

        completed_cols[:, completed_cols_i] = reconstructed_root.reshape((reconstructed_root.shape[0],))
        completed_cols_i += 1

    remaining_col_idx = np.argmax(available_cols)
    completed_cols[:, -1] = np.around(prob_mat[:, remaining_col_idx]).reshape((prob_mat.shape[0],))

    reconstructed = np.empty_like(measured)
    for i in range(measured.shape[1]):
        prob_measured = prob_measure(measured[:, i].reshape((measured.shape[0], 1)), completed_cols, fpr, fnr)
        max_idx = np.argmax(prob_measured)
        reconstructed[:, i] = completed_cols[:, max_idx]
        # completed_cols = np.delete(completed_cols, max_idx, 1)
    return reconstructed

    # refined = mlrefinementcol(completed_cols, measured, fpr, fnr)
    # return refined


def main():

    # Test with actual data
    file_name = "simNo_10-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000"
    # file_name = "simNo_10-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000"
    # file_name  = "simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000"
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

    reconstructed = reconstruct(measured_data, fpr, fnr)

    print(f"Number of elements in matrix: {true_data.shape[0] * true_data.shape[1]}")
    is_ptree = metrics.isPtree(reconstructed)
    print(f"Is PTree: {is_ptree}")
    (_, _, _, ad_score) = metrics.compareAD(true_data, reconstructed)
    print(f"AD Score: {ad_score}")
    (_, _, df_score) = metrics.compareDF(true_data, reconstructed)
    print(f"DF Score: {df_score}")
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
    # N = 50
    # M = 2*N
    # fpr = 0.00 # False positive rate.
    # fnr = 0.1  # False negative rate.
    # true_data = np.triu(np.ones((N, N)))
    # true_data = np.array(np.bmat([[true_data, np.zeros((N, N))], [np.zeros((N, N)), true_data]]))
    # # true_data = np.array([[1, 1], [1, 1], [1, 0], [1, 1], [0, 1]])
    # print(f"True: \n{true_data}")
    # measured = corrupt(true_data, fpr, fnr)
    # print(f"Measured: \n{measured}")
    # reconstructed = reconstruct(measured, fpr, fnr)
    # print(f"Reconstructed: \n{reconstructed}")
    # print(f"Number of elements in matrix: {true_data.shape[0] * true_data.shape[1]}")
    # metrics.isPtree(reconstructed)
    # metrics.compareAD(true_data, reconstructed)
    # metrics.compareDF(true_data, reconstructed)
    # print(f"Number of differences: {metrics.num_diffs(true_data, reconstructed)}")


if __name__ == '__main__':
    main()

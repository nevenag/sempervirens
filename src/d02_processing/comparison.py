"""
Comparing this project's algorithm to HUNTRESS
"""

import numpy as np
import pandas as pd
import multiprocessing as mp

import main
import huntress
import metrics


def read_data(filename):     # reads a matrix from file and returns it in BOOL type
    ds = pd.read_csv(filename, sep='\t', index_col=0)
    M = ds.values
    return M 


def load_file(file_prefix: str):
    true_data_filename = "data/" + file_prefix + ".SC.before_FP_FN_NA"
    measured_data_filename = "data/only_fpfn/" + file_prefix + ".SC"
    true_data = read_data(true_data_filename)
    measured_data = read_data(measured_data_filename)
    assert np.sum(measured_data == 3) == 0
    # # Set missing entries in measured data to true values (only considering false positives and negatives right now).
    # missing_entries = (measured_data == 3)
    # measured_data[missing_entries] = true_data[missing_entries]
    return true_data, measured_data

def huntress_reconstruct(file_prefix, fpr, fnr):
    measured_data_filename = "data/only_fpfn/" + file_prefix + ".SC"
    output_name = "temp/" + file_prefix + ".SC"
    fn_fpratio = 51
    fp_coeff = fpr
    fn_coeff = fnr
    fn_conorm=0.1
    fp_conorm=fn_conorm*fp_coeff/fn_coeff
    huntress.Reconstruct(
        measured_data_filename, 
        output_name, 
        Algchoice="FPNA", 
        n_proc=mp.cpu_count(),
        fnfp=fn_fpratio,
        post_fn=fn_conorm,
        post_fp=fp_conorm
    )
    reconstruction = read_data(output_name + ".CFMatrix")
    return reconstruction

def compute_metrics(true_data, reconstruction):
    is_ptree = metrics.isPtree(reconstruction)
    (n_adpairs, error_pairs, num_errors, ad_score) = metrics.compareAD(true_data, reconstruction)
    (num_diff_pairs, num_errors, df_score) = metrics.compareDF(true_data, reconstruction)
    num_differences = metrics.num_diffs(true_data, reconstruction)
    return np.array([is_ptree, ad_score, df_score, num_differences/true_data.size])


file_prefixes = [
    ("simNo_10-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000", 0.001, 0.05),
    # ("simNo_5-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    ("simNo_10-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",  0.001, 0.2),
    ("simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    # ("simNo_6-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    ("simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2),
    # ("simNo_6-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2),
    # ("simNo_7-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    ("simNo_2-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2),
    # ("simNo_7-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2),
    # ("simNo_3-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    # ("simNo_8-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    # ("simNo_3-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2),
    # ("simNo_8-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2),
    # ("simNo_4-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    # ("simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  0.001, 0.05),
    # ("simNo_4-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2), 
    # ("simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   0.001, 0.2),
]

np.set_printoptions(precision=3)
for (file_prefix, fpr, fnr) in file_prefixes:
    print(f"Running file: {file_prefix}")
    true_data, measured_data = load_file(file_prefix)

    print(f"Reconstructing using this method...")
    this_reconstruction = main.reconstruct(measured_data, fpr, fnr)

    print(f"Reconstructing using huntress...")
    huntress_reconstruction = huntress_reconstruct(file_prefix, fpr, fnr)

    print(f"Computing this method's metrics...")
    m = compute_metrics(true_data, this_reconstruction)
    print(m)

    print(f"Computing huntress's metrics...")
    hm = compute_metrics(true_data, huntress_reconstruction)
    print(hm)

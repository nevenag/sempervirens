"""
Comparing this project's algorithm to HUNTRESS
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime
import time

import main
import huntress
import CountBasedReconstructor
import metrics
import file_prefixes
import row_reconstructor


def read_data(filename):     # reads a matrix from file and returns it in BOOL type
    ds = pd.read_csv(filename, sep='\t', index_col=0)
    M = ds.values
    return M 


def load_file(file_prefix: str, only_fpfn = False):
    true_data_filename = "data/" + file_prefix + ".SC.before_FP_FN_NA"
    if only_fpfn:
        measured_data_filename = "data/only_fpfn/" + file_prefix + ".SC"
    else:
        measured_data_filename = "data/" + file_prefix + ".SC"
    true_data = read_data(true_data_filename)
    measured_data = read_data(measured_data_filename)
    # assert np.sum(measured_data == 3) == 0
    # # Set missing entries in measured data to true values (only considering false positives and negatives right now).
    # missing_entries = (measured_data == 3)
    # measured_data[missing_entries] = true_data[missing_entries]
    return true_data, measured_data

def huntress_reconstruct(file_prefix, fpr, fnr, only_fpfn = False):
    if only_fpfn:
        measured_data_filename = "data/only_fpfn/" + file_prefix + ".SC"
    else:
        measured_data_filename = "data/" + file_prefix + ".SC"
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

# file_prefixes = [
    # ("simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  300, 300, 1e-3, 0.05, 0.05),
#     ("simNo_6-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   300, 300, 1e-3, 0.2,  0.05),
    # ("simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   300, 300, 1e-3, 0.2,  0.05),
# ]

# file_prefixes = file_prefixes.file_prefixes_300x300s
# file_prefixes = file_prefixes.file_prefixes_1000x300s
file_prefixes = file_prefixes.file_prefixes_300x1000s
# file_prefixes = file_prefixes.file_prefixes_1000x1000s


curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
data_file_name = f"data/metrics/metrics_countbased_all_300x1000s_{curr_time}.csv"
metrics_df = pd.DataFrame(columns=['file', 'n', 'm', 'fpr', 'fnr', 'nar', 'time', 'is_ptree', 'ad_score', 'dl_score', 'fraction_diffs'])

metrics_mat = np.zeros((len(file_prefixes), 4))
time_start = time.perf_counter()

np.set_printoptions(precision=5)
for i, (file_prefix, n, m, fpr, fnr, nar) in enumerate(file_prefixes):
    print(f"\nRunning file: {file_prefix}")
    true_data, measured_data = load_file(file_prefix)


    # submat = measured_data[:, 50:100]
    # submat[submat == 3] = 0
    # col_i = 2
    # print(submat[:, col_i] == 1)
    # hist = np.sum(submat[submat[:, col_i] == 1, :], axis = 0)
    # print(hist)
    # exit()

    # print(f"Reconstructing using this method...")
    # this_reconstruction = main.reconstruct(measured_data, fpr, fnr)
    # # print(f"Computing this method's metrics...")
    # m = compute_metrics(true_data, this_reconstruction)
    # print(m)
    # metrics_mat[i, :] = m

    # print(f"Reconstructing using huntress...")
    # huntress_reconstruction = huntress_reconstruct(file_prefix, fpr, fnr)
    # print(f"Computing huntress's metrics...")
    # hm = compute_metrics(true_data, huntress_reconstruction)
    # print(hm)

    # print(f"Reconstructing using count based...")
    ## Col
    t0 = time.perf_counter()
    separator_reconstruction = CountBasedReconstructor.main_Rec(measured_data, fpr, fnr, nar)
    # sm = compute_metrics(true_data, separator_reconstruction)
    # print(f'Col:                 {sm}')
    separator_reconstruction = CountBasedReconstructor.mlrefinementcol(separator_reconstruction, measured_data, fpr, fnr)
    t1 = time.perf_counter()
    sm_after_ml = compute_metrics(true_data, separator_reconstruction)
    # print(f'Col & ml:            {sm_after_ml}')
    metrics_mat[i, :] = sm_after_ml
    new_df = pd.DataFrame([[file_prefix, n, m, fpr, fnr, nar, t1 - t0, sm_after_ml[0], sm_after_ml[1], sm_after_ml[2], sm_after_ml[3]]], columns=metrics_df.columns)
    metrics_df = pd.concat((metrics_df, new_df), ignore_index = True)

    # print(f'Col:')
    # print(f'0 truth, 1 recons: {np.sum((1-true_data)*separator_reconstruction)}')
    # print(f'1 truth, 0 recons: {np.sum(true_data*(1-separator_reconstruction))}')
    # print(f'FP, still false: {np.sum((1-true_data)*measured_data*separator_reconstruction*(measured_data != 3))}')
    # print(f'FN, still false: {np.sum(true_data*(1-measured_data)*(1-separator_reconstruction)*(measured_data != 3))}')

    ## Row
    # t0 = time.perf_counter()
    # base = compute_metrics(true_data, measured_data)
    # # print(f'Base:     {base}')
    # separator_reconstruction = row_reconstructor.row_rec(measured_data, fpr, fnr, nar)
    # sm = compute_metrics(true_data, separator_reconstruction)
    # # print(f'Row:      {sm}')
    # separator_reconstruction = CountBasedReconstructor.mlrefinementcol(separator_reconstruction, measured_data, fpr, fnr)
    # t1 = time.perf_counter()
    # sm_after_ml = compute_metrics(true_data, separator_reconstruction)
    # # print(f'Row & ml: {sm_after_ml}')
    # metrics_mat[i, :] = sm_after_ml
    # new_df = pd.DataFrame([[file_prefix, n, m, fpr, fnr, nar, t1 - t0, sm_after_ml[0], sm_after_ml[1], sm_after_ml[2], sm_after_ml[3]]], columns=metrics_df.columns)
    # metrics_df = pd.concat((metrics_df, new_df), ignore_index = True)

    # print(f'Row:')
    # print(sm_after_ml)
    # print(f'0 truth, 1 recons: {np.sum((1-true_data)*separator_reconstruction)}')
    # print(f'1 truth, 0 recons: {np.sum(true_data*(1-separator_reconstruction))}')
    # print(f'FP, still false: {np.sum((1-true_data)*measured_data*separator_reconstruction*(measured_data != 3))}')
    # print(f'FN, still false: {np.sum(true_data*(1-measured_data)*(1-separator_reconstruction)*(measured_data != 3))}')

   
    ## Row -> Col
    # t0 = time.perf_counter()
    # separator_reconstruction = row_reconstructor.row_rec(measured_data, fpr, fnr, nar)
    # sm = compute_metrics(true_data, separator_reconstruction)
    # print(f'Row:                 {sm}')
    # separator_reconstruction = CountBasedReconstructor.mlrefinementcol(separator_reconstruction, measured_data, fpr, fnr)
    # sm_after_ml = compute_metrics(true_data, separator_reconstruction)
    # print(f'Row & ml:            {sm_after_ml}')
    # separator_reconstruction = CountBasedReconstructor.main_Rec(separator_reconstruction, fpr, fnr, nar)
    # sm = compute_metrics(true_data, separator_reconstruction)
    # print(f'Row & ml & Col:      {sm}')
    # separator_reconstruction = CountBasedReconstructor.mlrefinementcol(separator_reconstruction, measured_data, fpr, fnr)
    # sm_after_ml = compute_metrics(true_data, separator_reconstruction)
    # print(f'Row & ml & Col & ml: {sm_after_ml}')
    # t1 = time.perf_counter()
    # new_df = pd.DataFrame([[file_prefix, n, m, fpr, fnr, nar, t1 - t0, sm_after_ml[0], sm_after_ml[1], sm_after_ml[2], sm_after_ml[3]]], columns=metrics_df.columns)
    # metrics_df = pd.concat((metrics_df, new_df), ignore_index = True)

time_end = time.perf_counter()
print(f"Took {time_end - time_start} seconds total")

print(f"{data_file_name}:")
metrics_df.to_csv(data_file_name)
# np.save(data_file_name, metrics_mat)

np.set_printoptions(precision=3)
print(f'* FNR: 0.2')
df_fnr0_2 = metrics_df[np.isclose(metrics_df.fnr, 0.2)][["is_ptree", "ad_score", "dl_score", "fraction_diffs"]]
times_fnr0_2 = metrics_df[np.isclose(metrics_df.fnr, 0.2)][["time"]]
print(f'    * Average: {list(df_fnr0_2.mean())}')
print(f'    * Std Dev: {list(df_fnr0_2.std())}')
print(f'    * Min:     {list(df_fnr0_2.min())}')
print(f'    * Max:     {list(df_fnr0_2.max())}')
print(f'    * Time: {list(times_fnr0_2.mean())[0]} average, {list(times_fnr0_2.std())[0]} std')

print(f'* FNR: 0.05')
df_fnr0_05 = metrics_df[np.isclose(metrics_df.fnr, 0.05)][["is_ptree", "ad_score", "dl_score", "fraction_diffs"]]
times_fnr0_05 = metrics_df[np.isclose(metrics_df.fnr, 0.05)][["time"]]
print(f'    * Average: {list(df_fnr0_05.mean())}')
print(f'    * Std Dev: {list(df_fnr0_05.std())}')
print(f'    * Min:     {list(df_fnr0_05.min())}')
print(f'    * Max:     {list(df_fnr0_05.max())}')
print(f'    * Time: {list(times_fnr0_05.mean())[0]} average, {list(times_fnr0_05.std())[0]} std')
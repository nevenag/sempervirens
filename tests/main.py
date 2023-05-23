import sys
import datetime
import time
import re
import multiprocessing as mp
import numpy as np
import pandas as pd

import tests.huntress as huntress
# import scphylo
import tests.freq_based as freq_based
import tests.TestingNewpivot as tnp

import sempervirens

from tests.file_prefixes import *

def read_df(filename):
    """Reads dataframe from filename and returns it."""
    df = pd.read_csv(filename, sep = '\t', index_col = 0)
    return df

def load_file(file_prefix, only_fpfn = False):
    """Reads and returns true dataframe and measured data frame corresponing to file_prefix."""
    true_data_filename = "data/" + file_prefix + ".SC.before_FP_FN_NA"
    if only_fpfn:
        measured_data_filename = "data/only_fpfn/" + file_prefix + ".SC"
    else:
        measured_data_filename = "data/" + file_prefix + ".SC"
    true_df = read_df(true_data_filename)
    measured_df = read_df(measured_data_filename)
    return true_df, measured_df

def compute_metrics(true_data, reconstruction):
    """Computes [ptree, AD score, DL score, fraction errors] for the reconstruction."""
    is_ptree = sempervirens.metrics.isPtree(reconstruction)
    (n_adpairs, error_pairs, num_errors, ad_score) = sempervirens.metrics.compareAD(true_data, reconstruction)
    (num_diff_pairs, num_errors, df_score) = sempervirens.metrics.compareDL(true_data, reconstruction)
    num_differences = sempervirens.metrics.num_diffs(true_data, reconstruction)
    return np.array([is_ptree, ad_score, df_score, num_differences/true_data.size])

def print_metrics(metrics_df):
    """Takes in a dataframe and computes and prints metrics"""
    # metrics_df should be a pandas dataframe with the following columns:
    # ['file', 'n', 'm', 'fpr', 'fnr', 'mer', 'time', 'is_ptree', 'ad_score', 'dl_score', 'fraction_diffs']

    np.set_printoptions(formatter={'float_kind': '{:.3f}'.format})

    # Print metrics by each combination of false positive rate, false negative rate, and missing entry rate.
    # TODO: this is bad because these are arrays of floats. Should have some sort of tolerance.
    fprs = np.unique(metrics_df.fpr)
    fnrs = np.unique(metrics_df.fnr)
    mers = np.unique(metrics_df.mer)
    match_count = 0;
    for fpr in fprs:
        for fnr in fnrs:
            for mer in mers:
                fpr_match = np.isclose(metrics_df.fpr, fpr)
                fnr_match = np.isclose(metrics_df.fnr, fnr)
                mer_match = np.isclose(metrics_df.mer, mer)
                matches = np.logical_and(fpr_match, np.logical_and(fnr_match, mer_match))
                match_count += np.sum(matches)
                df = metrics_df[matches][['is_ptree', 'ad_score', 'dl_score', 'fraction_diffs']]
                times = metrics_df[matches][['time']]
                print(f'* fpr: {fpr}    fnr: {fnr}    mer: {mer}')
                print(f'    * Average: {df.mean().to_numpy()}')
                print(f'    * Std Dev: {df.std().to_numpy()}')
                print(f'    * Min:     {df.min().to_numpy()}')
                print(f'    * Max:     {df.max().to_numpy()}')
                print(f'    * Time: {times.mean().to_numpy()[0]} average, {times.std().to_numpy()[0]} std')
    assert match_count == metrics_df.shape[0]

def run(data_file_prefix, file_prefixes, args = sys.argv[1:]):
    assert len(args) in [0, 1]
    if len(args) == 0:
        algorithm = 'sempervirens'
    else:
        algorithm = args[0]
        assert(algorithm in ['sempervirens', 'huntress', 'scistree', 'freq-based', 'tnp'])
    print(f"\nUsing algorithm: {algorithm}.")
    np.set_printoptions(formatter={'float_kind': '{:.3f}'.format})

    curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    data_file_name = f"metrics/metrics_{algorithm}_{data_file_prefix}_{curr_time}.csv"
    metrics_df = pd.DataFrame(columns=['file', 'n', 'm', 'fpr', 'fnr', 'mer', 'time', 'is_ptree', 'ad_score', 'dl_score', 'fraction_diffs'])

    time_start = time.perf_counter()

    for i, (file_prefix, n, m, fpr, fnr, mer) in enumerate(file_prefixes):
        print(f"Running file: {file_prefix}")
        true_df, measured_df = load_file(file_prefix)

        t0 = time.perf_counter()
        if algorithm == 'sempervirens':
            # TODO: fix. TESTING
            reconstruction = sempervirens.reconstruct(measured_df.to_numpy(), fpr, fnr, mer, true_df.to_numpy())
            # End testing.
            # reconstruction = sempervirens.reconstruct(measured_df.to_numpy(), fpr, fnr, mer)
        elif algorithm == "huntress":
            reconstruction = huntress_reconstruct(file_prefix, fpr, fnr, mer)
        elif algorithm == "scistree":
            reconstruction = scphylo.tl.scistree(measured_df, alpha=fpr, beta=fnr, n_threads=mp.cpu_count(), experiment=True)[0].to_numpy()
        elif algorithm == "freq-based":
            measured = measured_df.to_numpy()
            m_in = measured.copy()
            m_in[m_in == 3] = 0
            reconstruction = freq_based.main_Rec(m_in, fpr, fnr)
            reconstruction = freq_based.mlrefinementcol(reconstruction, m_in, fpr, fnr)
            reconstruction = freq_based.mlrefinementrow(reconstruction, m_in, fpr, fnr)
        elif algorithm == "tnp":
            measured = measured_df.to_numpy()
            m_in = measured.copy()
            m_in[m_in == 3] = 0
            reconstruction = tnp.main_Rec10(m_in, fpr, fnr)
            reconstruction = tnp.mlrefinementcol(reconstruction, m_in, fpr, fnr)
            reconstruction = tnp.mlrefinementrow(reconstruction, m_in, fpr, fnr)

        t1 = time.perf_counter()

        metrics = compute_metrics(true_df.to_numpy(), reconstruction)
        print(metrics)
        new_df = pd.DataFrame([[file_prefix, n, m, fpr, fnr, mer, t1 - t0, metrics[0], metrics[1], metrics[2], metrics[3]]], columns = metrics_df.columns)
        metrics_df = pd.concat((metrics_df, new_df), ignore_index = True)

    time_end = time.perf_counter()
    print(f"Took {time_end - time_start} seconds total")

    print(f"{data_file_name}:")
    metrics_df.to_csv(data_file_name)

    print_metrics(metrics_df)


def huntress_reconstruct(file_prefix, fpr, fnr, mer):
    measured_data_filename = "data/" + file_prefix + ".SC"
    output_name = "/tmp/" + file_prefix + ".SC"
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
    reconstruction = read_df(output_name + ".CFMatrix")
    return reconstruction.to_numpy()


def run_300x300s_0_05fnr(alg = sys.argv[1:]):
    run("300x300s_0_05fnr", file_prefixes_300x300s_0_05fnr, alg)
def run_300x300s_0_2fnr(alg = sys.argv[1:]):
    run("300x300s_0_2fnr", file_prefixes_300x300s_0_2fnr, alg)
def run_300x300s_0_001fpr(alg = sys.argv[1:]):
    run("300x300s_0_001fpr", file_prefixes_300x300s_0_001fpr, alg)
def run_300x300s_0_003fpr(alg = sys.argv[1:]):
    run("300x300s_0_003fpr", file_prefixes_300x300s_0_003fpr, alg)
def run_300x300s_0_01fpr(alg = sys.argv[1:]):
    run("300x300s_0_01fpr", file_prefixes_300x300s_0_01fpr, alg)
def run_300x300s_0_03fpr(alg = sys.argv[1:]):
    run("300x300s_0_03fpr", file_prefixes_300x300s_0_03fpr, alg)
def run_300x300s_0_05fpr(alg = sys.argv[1:]):
    run("300x300s_0_05fpr", file_prefixes_300x300s_0_05fpr, alg)

def run_1000x300s_0_05fnr(alg = sys.argv[1:]):
    run("1000x300s_0_05fnr", file_prefixes_1000x300s_0_05fnr, alg)
def run_1000x300s_0_2fnr(alg = sys.argv[1:]):
    run("1000x300s_0_2fnr", file_prefixes_1000x300s_0_2fnr, alg)
def run_1000x300s_0_001fpr(alg = sys.argv[1:]):
    run("1000x300s_0_001fpr", file_prefixes_1000x300s_0_001fpr, alg)

def run_300x1000s_0_05fnr(alg = sys.argv[1:]):
    run("300x1000s_0_05fnr", file_prefixes_300x1000s_0_05fnr, alg)
def run_300x1000s_0_2fnr(alg = sys.argv[1:]):
    run("300x1000s_0_2fnr", file_prefixes_300x1000s_0_2fnr, alg)
def run_300x1000s_0_001fpr(alg = sys.argv[1:]):
    run("300x1000s_0_001fpr", file_prefixes_300x1000s_0_001fpr, alg)
def run_300x1000s_0_01fpr(alg = sys.argv[1:]):
    run("300x1000s_0_01fpr", file_prefixes_300x1000s_0_01fpr, alg)
def run_300x1000s_0_03fpr(alg = sys.argv[1:]):
    run("300x1000s_0_03fpr", file_prefixes_300x1000s_0_03fpr, alg)
def run_300x1000s_0_05fpr(alg = sys.argv[1:]):
    run("300x1000s_0_05fpr", file_prefixes_300x1000s_0_05fpr, alg)
def run_300x1000s_0_1fpr(alg = sys.argv[1:]):
    run("300x1000s_0_1fpr", file_prefixes_300x1000s_0_1fpr, alg)

def run_1000x1000s_0_05fnr(alg = sys.argv[1:]):
    run("1000x1000s_0_05fnr", file_prefixes_1000x1000s_0_05fnr, alg)
def run_1000x1000s_0_2fnr(alg = sys.argv[1:]):
    run("1000x1000s_0_2fnr", file_prefixes_1000x1000s_0_2fnr, alg)
def run_1000x1000s_0_001fpr(alg = sys.argv[1:]):
    run("1000x1000s_0_001fpr", file_prefixes_1000x1000s_0_001fpr, alg)
def run_1000x1000s_0_01fpr(alg = sys.argv[1:]):
    run("1000x1000s_0_01fpr", file_prefixes_1000x1000s_0_01fpr, alg)
def run_1000x1000s_0_1fpr(alg = sys.argv[1:]):
    run("1000x1000s_0_1fpr", file_prefixes_1000x1000s_0_1fpr, alg)

def test(alg = sys.argv[1:]):
    # files = [
         # ("simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000", 300, 300, 0.001, 0.05, 0.05),
#         ("simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000", 300, 300, 0.001, 0.2, 0.05),
#         ("fp_001/simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.01-fn_0.05-na_0.05-d_0-l_1000000",  300, 300, 0.01, 0.05, 0.05),
#         ("fp_001/simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.01-fn_0.2-na_0.05-d_0-l_1000000",   300, 300, 0.01, 0.2,  0.05)
##        ("simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",  300, 1000, 1e-3, 0.05, 0.05),
##        ("simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",   300, 1000, 1e-3, 0.2,  0.05),
##        ("fp_001/simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.01-fn_0.05-na_0.05-d_0-l_1000000",  300, 1000, 0.01, 0.05, 0.05),
##        ("fp_001/simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.01-fn_0.2-na_0.05-d_0-l_1000000",   300, 1000, 0.01, 0.2,  0.05),
    # ]
    # run("300x300s_test", files, alg)
    

    run_300x300s_0_001fpr(alg)
    run_300x300s_0_003fpr(alg)
    run_300x300s_0_01fpr(alg)

    run_300x1000s_0_001fpr(alg)
    run_300x1000s_0_01fpr(alg)

    # run_1000x1000s_0_001fpr(alg)
    # run_1000x1000s_0_01fpr(alg)


    # for alg in ['scistree']:
        # arr = [alg]
        # run_300x1000s_0_03fpr(arr)
        # run_300x1000s_0_05fpr(arr)

#   files = [
#       'metrics_scistree_300x1000s_0_001fpr_2023-03-03_01:59.csv',
#       'metrics_scistree_300x1000s_0_01fpr_2023-03-03_00:26.csv',
#       'metrics_scistree_300x1000s_0_1fpr_2023-03-02_20:03.csv',
#       'metrics_300x300s_0_001fpr_2023-03-02_15:50.csv',
#       'metrics_300x300s_0_03fpr_2023-03-02_14:34.csv',
#       'metrics_300x300s_0_07fpr_2023-03-02_12:52.csv',
#       'metrics_300x300s_0_1fpr_2023-03-02_11:06.csv',
#   ]
#     files = [
#         'metrics_scistree_300x1000s_0_05fpr_2023-03-12_13:11.csv',
#         'metrics_scistree_300x1000s_0_03fpr_2023-03-12_11:04.csv',
# #       'metrics_huntress_300x1000s_0_05fpr_2023-03-12_22:30.csv',
# #       'metrics_huntress_300x1000s_0_03fpr_2023-03-11_23:58.csv',
# #       'metrics_sempervirens_300x1000s_0_05fpr_2023-03-11_23:43.csv',
# #       'metrics_sempervirens_300x1000s_0_03fpr_2023-03-11_23:28.csv',
# #       'metrics_scistree_300x300s_0_05fpr_2023-03-11_18:30.csv',
# #       'metrics_scistree_300x300s_0_03fpr_2023-03-11_17:54.csv',
# #       'metrics_scistree_300x300s_0_01fpr_2023-03-11_17:30.csv',
# #       'metrics_scistree_300x300s_0_001fpr_2023-03-11_17:10.csv',
# #       'metrics_huntress_300x300s_0_05fpr_2023-03-11_16:17.csv',
# #       'metrics_huntress_300x300s_0_03fpr_2023-03-11_15:26.csv',
# #       'metrics_huntress_300x300s_0_01fpr_2023-03-11_14:35.csv',
# #       'metrics_huntress_300x300s_0_001fpr_2023-03-11_12:54.csv',
# #       'metrics_sempervirens_300x300s_0_05fpr_2023-03-11_12:52.csv',
# #       'metrics_sempervirens_300x300s_0_03fpr_2023-03-11_12:50.csv',
# #       'metrics_sempervirens_300x300s_0_01fpr_2023-03-11_12:49.csv',
# #       'metrics_sempervirens_300x300s_0_001fpr_2023-03-11_12:47.csv',
#     ]
# 
#     for file in reversed(files):
#         file = 'data/metrics/' + file
#         metrics_df = pd.read_csv(file)
#         print(file)
#         print_metrics(metrics_df)
# 
    # corrupt_data()

def corrupt_data():
    # Generating some extra fpr data

    # Take the true matrices of these and corrupt them.
    prefixes = file_prefixes_300x1000s_0_05fnr + file_prefixes_300x1000s_0_2fnr

    rates = [prefix[-3:] for prefix in prefixes]
    for i in range(len(rates)):
        rates[i] = (0.05, rates[i][1], rates[i][2]) # Set fpr to 0.05

    rng = np.random.default_rng()

    for (prefix, rate) in zip(prefixes, rates):
        name = prefix[0]
        new_name = re.sub("-fp_0.001-", "-fp_0.05-", name)
        new_name = "fp_005/" + new_name

        true_mat = read_df("data/" + name + ".SC.before_FP_FN_NA").to_numpy()

        fpr = rate[0]
        fnr = rate[1]
        mer = rate[2]
        # print(name, fpr, fnr, mer)

        corrupted_mat = true_mat.copy()

        num_fp_needed = int(np.round(fpr * np.sum(true_mat == 0)))
        num_fn_needed = int(np.round(fnr * np.sum(true_mat == 1)))
        num_na_needed = int(np.round(mer * np.size(true_mat)))

        zero_coords = np.argwhere(true_mat == 0)
        one_coords = np.argwhere(true_mat == 1)
        assert len(zero_coords) + len(one_coords) == np.size(true_mat)

        fp_coords = rng.choice(zero_coords, num_fp_needed, replace = False)
        assert fp_coords.shape[0] == num_fp_needed
        fn_coords = rng.choice(one_coords, num_fn_needed, replace = False)
        assert fn_coords.shape[0] == num_fn_needed

        remaining_zero_coords = set(map(tuple, zero_coords)) - set(map(tuple, fp_coords))
        remaining_zero_coords = np.array(list(map(list, remaining_zero_coords)))
        remaining_one_coords = set(map(tuple, one_coords)) - set(map(tuple, fn_coords))
        remaining_one_coords = np.array(list(map(list, remaining_one_coords)))
        remaining_coords = np.vstack((remaining_zero_coords, remaining_one_coords))
        na_coords = rng.choice(remaining_coords, num_na_needed, replace = False)
        assert na_coords.shape[0] == num_na_needed

        for x, y in fp_coords:
            assert corrupted_mat[x, y] == 0
            corrupted_mat[x, y] = 1
        for x, y in fn_coords:
            assert corrupted_mat[x, y] == 1
            corrupted_mat[x, y] = 0
        for x, y in na_coords:
            assert corrupted_mat[x, y] != 3
            corrupted_mat[x, y] = 3

        num_fp = np.sum(np.logical_and(true_mat == 0, corrupted_mat == 1))
        num_fn = np.sum(np.logical_and(true_mat == 1, corrupted_mat == 0))
        num_me = np.sum(corrupted_mat == 3)
        size = true_mat.shape[0] * true_mat.shape[1]
        # print(rate)
        # print(f'empirical fpr: {np.round(num_fp/np.sum(true_mat == 0), 3)}    fnr: {np.round(num_fn/np.sum(true_mat == 1), 3)}    mer: {np.round(num_me/size, 3)}')

        df = pd.DataFrame(corrupted_mat)
        # df.to_csv("data/" + new_name + ".SC", sep='\t')
        # print("data/" + new_name + ".SC")

        true_df = pd.DataFrame(true_mat)
        # true_df.to_csv("data/" + new_name + ".SC.before_FP_FN_NA", sep='\t')
        # print("data/" + new_name + ".SC.before_FP_FN_NA")

        print(f"(\"{new_name}\", 300, 1000, {fpr}, {fnr}, {mer}),")


if __name__ == '__main__':
    test()

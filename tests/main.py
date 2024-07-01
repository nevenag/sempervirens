import sys
import datetime
import time
import re
import multiprocessing as mp
import numpy as np
import pandas as pd
import scphylo
import subprocess
import argparse
import enum
from typing import Optional

import tests.huntress as huntress

import sempervirens

from tests.file_prefixes import *


class Algorithm(enum.Enum):
    SEMPERVIRENS = enum.auto()
    SEMPERVIRENS_RS = enum.auto()
    HUNTRESS = enum.auto()
    SCISTREEP = enum.auto()
    SCISTREE = enum.auto()
    SPHYR = enum.auto()
    SCITE = enum.auto()

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Algorithm[s.upper()]
        except KeyError:
            return s


def read_df(filename):
    """Reads dataframe from filename and returns it."""
    df = pd.read_csv(filename, sep="\t", index_col=0)
    return df


def load_file(file_prefix, only_fpfn=False):
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
    (n_adpairs, error_pairs, num_errors, ad_score) = sempervirens.metrics.compareAD(
        true_data, reconstruction
    )
    (num_diff_pairs, num_errors, df_score) = sempervirens.metrics.compareDL(
        true_data, reconstruction
    )
    num_differences = sempervirens.metrics.num_diffs(true_data, reconstruction)
    return np.array([is_ptree, ad_score, df_score, num_differences / true_data.size])


def print_metrics(metrics_df):
    """Takes in a dataframe and computes and prints metrics"""
    # metrics_df should be a pandas dataframe with the following columns:
    # This uses 'fpr', 'fnr', and 'mer' for backwards compatibility instead of 'fpp', 'fnp', and 'mep'
    # ['file', 'n', 'm', 'fpr', 'fnr', 'mer', 'time', 'is_ptree', 'ad_score', 'dl_score', 'fraction_diffs', 'rf_score']

    np.set_printoptions(formatter={"float_kind": "{:.3f}".format})

    # Print metrics by each combination of false positive rate, false negative rate, and missing entry rate.
    # TODO: this is bad because these are arrays of floats. Should have some sort of tolerance.
    fprs = np.unique(metrics_df.fpr)
    fnrs = np.unique(metrics_df.fnr)
    mers = np.unique(metrics_df.mer)
    match_count = 0
    for fpr in fprs:
        for fnr in fnrs:
            for mer in mers:
                fpr_match = np.isclose(metrics_df.fpr, fpr)
                fnr_match = np.isclose(metrics_df.fnr, fnr)
                mer_match = np.isclose(metrics_df.mer, mer)
                matches = np.logical_and(
                    fpr_match, np.logical_and(fnr_match, mer_match)
                )
                match_count += np.sum(matches)
                df = metrics_df[matches][
                    ["is_ptree", "ad_score", "dl_score", "fraction_diffs", "rf_score"]
                ]
                times = metrics_df[matches][["time"]]
                print(f"* fpr: {fpr}    fnr: {fnr}    mer: {mer}")
                print(f"    * Average: {df.mean().to_numpy()}")
                print(f"    * Std Dev: {df.std().to_numpy()}")
                print(f"    * Min:     {df.min().to_numpy()}")
                print(f"    * Max:     {df.max().to_numpy()}")
                print(
                    f"    * Time: {times.mean().to_numpy()[0]} average, {times.std().to_numpy()[0]} std"
                )
    assert match_count == metrics_df.shape[0]


def run(
    data_file_prefix,
    file_prefixes,
    algorithm: Algorithm,
    fpp_override: Optional[float],
    fnp_override: Optional[float],
    mep_override: Optional[float],
    quiet = False
):
    print(f"\nUsing algorithm: {algorithm}.")
    np.set_printoptions(formatter={"float_kind": "{:.3f}".format})

    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    data_file_name = f"metrics/metrics_{algorithm}_{data_file_prefix}_{curr_time}.csv"
    # Continuing to use fpr, fnr, mer instead of fpp, fnp, mep (i.e. rate instead of probability)
    # in dataframes for backwards compatability.
    metrics_df = pd.DataFrame(
        columns=[
            "file",
            "n",
            "m",
            "fpr",
            "fnr",
            "mer",
            "time",
            "is_ptree",
            "ad_score",
            "dl_score",
            "fraction_diffs",
            "rf_score",
            "fpp_override",
            "fnp_override",
            "mep_override",
        ]
    )

    time_start = time.perf_counter()

    for i, (file_prefix, n, m, true_fpp, true_fnp, true_mep) in enumerate(
        file_prefixes
    ):
        if not quiet:
            print(f"Running file: {file_prefix}")
        true_df, noisy_df = load_file(file_prefix)

        if fpp_override is not None:
            if not quiet:
                print(f"Overriding fpp {true_fpp} with {fpp_override}")
            fpp = fpp_override
        else:
            fpp = true_fpp

        if fnp_override is not None:
            if not quiet:
                print(f"Overriding fnp {true_fnp} with {fnp_override}")
            fnp = fnp_override
        else:
            fnp = true_fnp

        if mep_override is not None:
            if not quiet:
                print(f"Overriding mep {true_mep} with {mep_override}")
            mep = mep_override
        else:
            mep = true_mep

        t0 = time.perf_counter()
        if algorithm == Algorithm.SEMPERVIRENS:
            # Call from command-line
            reconstruction = sempervirens_reconstruct(file_prefix, fpp, fnp, mep)
            # Call as library function
            # reconstruction = sempervirens.reconstruct(noisy_df.to_numpy(), fpp, fnp, mep)
        elif algorithm == Algorithm.SEMPERVIRENS_RS:
            reconstruction = sempervirens_rs_reconstruct(file_prefix, fpp, fnp, mep)
        elif algorithm == Algorithm.HUNTRESS:
            reconstruction = huntress_reconstruct(file_prefix, fpp, fnp, mep)
        elif algorithm == Algorithm.SCISTREEP:
            # Scphylo uses a parallelized scistree.
            reconstruction = scphylo.tl.scistree(
                noisy_df, alpha=fpp, beta=fnp, n_threads=mp.cpu_count(), experiment=True
            )[0].to_numpy()
        elif algorithm == Algorithm.SCISTREE:
            reconstruction = scistree_reconstruct(noisy_df, fpp, fnp, mep)
        elif algorithm == Algorithm.SPHYR:
            reconstruction = scphylo.tl.sphyr(
                noisy_df, fpp, fnp, n_threads=mp.cpu_count()
            ).to_numpy()
        elif algorithm == Algorithm.SCITE:
            reconstruction = scphylo.tl.scite(noisy_df, fpp, fnp, experiment=True)
            reconstruction = reconstruction[0].to_numpy()
        t1 = time.perf_counter()

        metrics = compute_metrics(true_df.to_numpy(), reconstruction)

        reconstruction_df = pd.DataFrame(reconstruction)
        reconstruction_df.columns = noisy_df.columns
        reconstruction_df.index = noisy_df.index
        reconstruction_df.index.name = "cellIDxmutID"
        try:
            # Computing RF score is slow
            # rf_score = scphylo.tl.rf(true_df, reconstruction_df)
            raise Exception()
        except:
            rf_score = float("nan")
        metrics = np.hstack((metrics, np.array(rf_score)))

        # print(metrics)

        new_df = pd.DataFrame(
            [
                [
                    file_prefix,
                    n,
                    m,
                    true_fpp,
                    true_fnp,
                    true_mep,
                    t1 - t0,
                    metrics[0],
                    metrics[1],
                    metrics[2],
                    metrics[3],
                    metrics[4],
                    fpp_override,
                    fnp_override,
                    mep_override,
                ]
            ],
            columns=metrics_df.columns,
        )
        metrics_df = pd.concat((metrics_df, new_df), ignore_index=True)

    time_end = time.perf_counter()
    print(f"Took {time_end - time_start} seconds total")

    print(f"{data_file_name}:")
    metrics_df.to_csv(data_file_name)

    print_metrics(metrics_df)


def sempervirens_reconstruct(file_prefix, fpr, fnr, mer):
    noisy_filename = "data/" + file_prefix + ".SC"
    output_filename = "/tmp/" + file_prefix + ".SC.CFMatrix"
    subprocess.run(
        [
            "python",
            "sempervirens/reconstructor.py",
            noisy_filename,
            str(fpr),
            str(fnr),
            str(mer),
            "-o",
            output_filename,
        ]
    )
    reconstruction = read_df(output_filename).to_numpy()
    return reconstruction


def sempervirens_rs_reconstruct(file_prefix, fpr, fnr, mer):
    noisy_data_filename = "data/" + file_prefix + ".SC"
    output_file = "/tmp/" + file_prefix + ".SC.CFMatrix"
    cmd = [
        "sempervirens-rs/target/release/sempervirens-rs",
        noisy_data_filename,
        f"{fpr}",
        f"{fnr}",
        f"{mer}",
        "-o",
        output_file,
        "--num-threads",
        f"{mp.cpu_count()}",
    ]
    subprocess.run(cmd)
    reconstruction = read_df(output_file)
    return reconstruction.to_numpy()


def huntress_reconstruct(file_prefix, fpr, fnr, mer):
    measured_data_filename = "data/" + file_prefix + ".SC"
    output_name = "/tmp/" + file_prefix + ".SC"
    fn_fpratio = 51
    fp_coeff = fpr
    fn_coeff = fnr
    fn_conorm = 0.1
    fp_conorm = fn_conorm * fp_coeff / fn_coeff
    huntress.Reconstruct(
        measured_data_filename,
        output_name,
        Algchoice="FPNA",
        n_proc=mp.cpu_count(),
        fnfp=fn_fpratio,
        post_fn=fn_conorm,
        post_fp=fp_conorm,
    )
    reconstruction = read_df(output_name + ".CFMatrix")
    return reconstruction.to_numpy()


def scistree_reconstruct(noisy_df, fpr, fnr, mer):
    cells = noisy_df.index
    snvs = noisy_df.columns
    df = noisy_df.transpose()

    df = df.replace(3, 0.5)
    df = df.replace(0, 1 - fnr)
    df = df.replace(1, fpr)

    df.index.name = f"HAPLOID {df.shape[0]} {df.shape[1]}"

    file = "/tmp/scistree.input"
    df.to_csv(file, sep=" ")
    with open(file) as ifile:
        data = ifile.read()
    with open(file, "w") as ofile:
        data = data.replace('"', "")
        ofile.write(data)

    cmd = [
        "tests/scistree",
        "-v",
        "-d",
        "0",
        "-e",
        # "-k",
        # f"{mp.cpu_count()}",
        "-o",
        "/tmp/scistree.gml",
        "/tmp/scistree.input",
    ]

    outfile = open("/tmp/scistree.output", "w")
    subprocess.run(cmd, stdout=outfile)

    data = []
    # detail = {"cost": "\n"}
    with open(f"/tmp/scistree.output") as infile:
        now_store = False
        for line in infile:
            line = line.strip()
            if "Imputed genotypes:" in line:
                now_store = True
            if line[:4] == "Site" and now_store:
                line = "".join(line.split(":")[1])
                line = line.replace("\t", "")
                data.append([int(x) for x in line.split(" ")])
            if "current cost: " in line:
                pass
                # cost = float(line.split("current cost: ")[1].split(", opt tree: ")[0])
                # detail["cost"] += f"    current best cost = {cost}\n"

    data = np.array(data)
    matrix_output = data.T

    df_output = pd.DataFrame(matrix_output)
    df_output.columns = snvs
    df_output.index = cells
    df_output.index.name = "cellIDxmutID"

    return df_output.to_numpy()


def run_front(data_file_prefix, file_prefixes):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=Algorithm.argparse,
        choices=list(Algorithm),
        default=Algorithm.SEMPERVIRENS,
    )
    parser.add_argument(
        "--fpp_override",
        type=float,
        default=None,
        help="False positive probability given to algorithm. Overrides true false positive probability.",
    )
    parser.add_argument(
        "--fnp_override",
        type=float,
        default=None,
        help="False negative probability given to algorithm. Overrides true false negative probability.",
    )
    parser.add_argument(
        "--mep_override",
        type=float,
        default=None,
        help="Missing entry probability given to algorithm. Overrides true missing entry probability.",
    )
    args = parser.parse_args()
    run(data_file_prefix, file_prefixes, **vars(args))

def run_300x300s_0_001fpp_0_05fnp():
    run_front("300x300s_0_001fpp_0_05fnp", file_prefixes_300x300s_0_001fpp_0_05fnp)
def run_300x300s_0_001fpp_0_2fnp():
    run_front("300x300s_0_001fpp_0_2fnp", file_prefixes_300x300s_0_001fpp_0_2fnp)
def run_300x300s_0_001fpp():
    run_front("300x300s_0_001fpp", file_prefixes_300x300s_0_001fpp)


def run_300x300s_0_001fpp_0_05fnp_0_1mep():
    run_front("300x300s_0_001fpp_0_05fnp_0_1mep", file_prefixes_300x300s_0_001fpp_0_05fnp_0_1mep)
def run_300x300s_0_001fpp_0_2fnp_0_1mep():
    run_front("300x300s_0_001fpp_0_2fnp_0_1mep", file_prefixes_300x300s_0_001fpp_0_2fnp_0_1mep)
def run_300x300s_0_001fpp_0_1mep():
    run_front("300x300s_0_001fpp_0_1mep", file_prefixes_300x300s_0_001fpp_0_1mep)


def run_300x300s_0_001fpp_0_05fnp_0_15mep():
    run_front("300x300s_0_001fpp_0_05fnp_0_15mep", file_prefixes_300x300s_0_001fpp_0_05fnp_0_15mep)
def run_300x300s_0_001fpp_0_2fnp_0_15mep():
    run_front("300x300s_0_001fpp_0_2fnp_0_15mep", file_prefixes_300x300s_0_001fpp_0_2fnp_0_15mep)
def run_300x300s_0_001fpp_0_15mep():
    run_front("300x300s_0_001fpp_0_15mep", file_prefixes_300x300s_0_001fpp_0_15mep)


def run_300x300s_0_003fpp():
    run_front("300x300s_0_003fpp", file_prefixes_300x300s_0_003fpp)

def run_300x300s_0_01fpp_0_05fnp():
    run_front("300x300s_0_01fpp_0_05fnp", file_prefixes_300x300s_0_01fpp_0_05fnp)
def run_300x300s_0_01fpp_0_2fnp():
    run_front("300x300s_0_01fpp_0_2fnp", file_prefixes_300x300s_0_01fpp_0_2fnp)
def run_300x300s_0_01fpp():
    run_front("300x300s_0_01fpp", file_prefixes_300x300s_0_01fpp)


def run_1000x300s_0_001fpp_0_05fnp():
    run_front("1000x300s_0_001fpp_0_05fnp", file_prefixes_1000x300s_0_001fpp_0_05fnp)


def run_1000x300s_0_001fpp_0_2fnp():
    run_front("1000x300s_0_001fpp_0_2fnp", file_prefixes_1000x300s_0_001fpp_0_2fnp)


def run_1000x300s_0_001fpp():
    run_front("1000x300s_0_001fpp", file_prefixes_1000x300s_0_001fpp)


def run_300x1000s_0_001fpp_0_05fnp():
    run_front("300x1000s_0_001fpp_0_05fnp", file_prefixes_300x1000s_0_001fpp_0_05fnp)


def run_300x1000s_0_001fpp_0_2fnp():
    run_front("300x1000s_0_001fpp_0_2fnp", file_prefixes_300x1000s_0_001fpp_0_2fnp)


def run_300x1000s_0_001fpp():
    run_front("300x1000s_0_001fpp", file_prefixes_300x1000s_0_001fpp)


def run_300x1000s_0_01fpp():
    run_front("300x1000s_0_01fpp", file_prefixes_300x1000s_0_01fpp)


def run_1000x1000s_0_001fpp_0_05fnp():
    run_front("1000x1000s_0_001fpp_0_05fnp", file_prefixes_1000x1000s_0_001fpp_0_05fnp)
def run_1000x1000s_0_001fpp_0_2fnp():
    run_front("1000x1000s_0_001fpp_0_2fnp", file_prefixes_1000x1000s_0_001fpp_0_2fnp)
def run_1000x1000s_0_001fpp():
    run_front("1000x1000s_0_001fpp", file_prefixes_1000x1000s_0_001fpp)

def run_1000x1000s_0_001fpp_0_05fnp_0_1mep():
    run_front("1000x1000s_0_001fpp_0_05fnp_0_1mep", file_prefixes_1000x1000s_0_001fpp_0_05fnp_0_1mep)
def run_1000x1000s_0_001fpp_0_2fnp_0_1mep():
    run_front("1000x1000s_0_001fpp_0_2fnp_0_1mep", file_prefixes_1000x1000s_0_001fpp_0_2fnp_0_1mep)
def run_1000x1000s_0_001fpp_0_1mep():
    run_front("1000x1000s_0_001fpp", file_prefixes_1000x1000s_0_001fpp_0_1mep)

def run_1000x1000s_0_001fpp_0_05fnp_0_15mep():
    run_front("1000x1000s_0_001fpp_0_05fnp_0_15mep", file_prefixes_1000x1000s_0_001fpp_0_05fnp_0_15mep)
def run_1000x1000s_0_001fpp_0_2fnp_0_15mep():
    run_front("1000x1000s_0_001fpp_0_2fnp_0_15mep", file_prefixes_1000x1000s_0_001fpp_0_2fnp_0_15mep)
def run_1000x1000s_0_001fpp_0_15mep():
    run_front("1000x1000s_0_001fpp_0_15mep", file_prefixes_1000x1000s_0_001fpp_0_15mep)


def run_1000x1000s_0_01fpp_0_05fnp():
    run_front("1000x1000s_0_01fpp_0_05fnp", file_prefixes_1000x1000s_0_01fpp_0_05fnp)
def run_1000x1000s_0_01fpp_0_2fnp():
    run_front("1000x1000s_0_01fpp_0_2fnp", file_prefixes_1000x1000s_0_01fpp_0_2fnp)
def run_1000x1000s_0_01fpp():
    run_front("1000x1000s_0_01fpp", file_prefixes_1000x1000s_0_01fpp)

def run_1000x1000s_0_01fpp_0_05fnp_0_1mep():
    run_front("1000x1000s_0_01fpp_0_05fnp_0_1mep", file_prefixes_1000x1000s_0_01fpp_0_05fnp_0_1mep)
def run_1000x1000s_0_01fpp_0_2fnp_0_1mep():
    run_front("1000x1000s_0_01fpp_0_2fnp_0_1mep", file_prefixes_1000x1000s_0_01fpp_0_2fnp_0_1mep)
def run_1000x1000s_0_01fpp_0_1mep():
    run_front("1000x1000s_0_01fpp_0_1mep", file_prefixes_1000x1000s_0_01fpp_0_1mep)

def run_1000x1000s_0_01fpp_0_05fnp_0_15mep():
    run_front("1000x1000s_0_01fpp_0_05fnp_0_15mep", file_prefixes_1000x1000s_0_01fpp_0_05fnp_0_15mep)
def run_1000x1000s_0_01fpp_0_2fnp_0_15mep():
    run_front("1000x1000s_0_01fpp_0_2fnp_0_15mep", file_prefixes_1000x1000s_0_01fpp_0_2fnp_0_15mep)
def run_1000x1000s_0_01fpp_0_15mep():
    run_front("1000x1000s_0_01fpp_0_15mep", file_prefixes_1000x1000s_0_01fpp_0_15mep)


def run_5000x500s_0_001fpp():
    run_front("5000x500s_0_001fpp", file_prefixes_5000x500s_0_001fpp)


def run_1000x10000s_0_001fpp():
    run_front("1000x10000s_0_001fpp", file_prefixes_1000x10000s_0_001fpp)


def run_2000x20000s_0_001fpp():
    run_front("2000x20000s_0_001fpp", file_prefixes_2000x20000s_0_001fpp)


def run_20000x2000s_0_001fpp():
    run_front("20000x2000s_0_001fpp", file_prefixes_20000x2000s_0_001fpp)


def test():
    pass

#     for file in reversed(files):
#         file = 'data/metrics/' + file
#         metrics_df = pd.read_csv(file)
#         print(file)
#         print_metrics(metrics_df)


def corrupt_data():
    # Generating some extra fpr data

    # Take the true matrices of these and corrupt them.
    prefixes = None

    rates = [prefix[-3:] for prefix in prefixes]
    for i in range(len(rates)):
        rates[i] = (rates[i][0], rates[i][1], 0.15)  # Set mep to 0.15

    rng = np.random.default_rng()

    for prefix, rate in zip(prefixes, rates):
        name = prefix[0]
        new_name = re.sub("-na_0.05-", "-na_0.15-", name)
        new_name = "me_0_15/" + new_name

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

        fp_coords = rng.choice(zero_coords, num_fp_needed, replace=False)
        assert fp_coords.shape[0] == num_fp_needed
        fn_coords = rng.choice(one_coords, num_fn_needed, replace=False)
        assert fn_coords.shape[0] == num_fn_needed

        remaining_zero_coords = set(map(tuple, zero_coords)) - set(
            map(tuple, fp_coords)
        )
        remaining_zero_coords = np.array(list(map(list, remaining_zero_coords)))
        remaining_one_coords = set(map(tuple, one_coords)) - set(map(tuple, fn_coords))
        remaining_one_coords = np.array(list(map(list, remaining_one_coords)))
        remaining_coords = np.vstack((remaining_zero_coords, remaining_one_coords))
        na_coords = rng.choice(remaining_coords, num_na_needed, replace=False)
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

        print(f'("{new_name}", 1000, 1000, {fpr}, {fnr}, {mer}),')


if __name__ == "__main__":
    test()

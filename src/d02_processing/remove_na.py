
import pandas as pd

def read_data(filename):     # reads a matrix from file and returns it in BOOL type
    ds = pd.read_csv(filename, sep='\t', index_col=0)
    M = ds.values
    return M 

def load_file(file_prefix: str):
    true_data_filename = "data/" + file_prefix + ".SC.before_FP_FN_NA"
    measured_data_filename = "data/" + file_prefix + ".SC"
    true_data = read_data(true_data_filename)
    measured_data = read_data(measured_data_filename)
    return true_data, measured_data

def write_file(file_prefix: str, matrix):
    true_data_filename = "data/" + file_prefix + ".SC.before_FP_FN_NA"
    df_in = pd.read_csv(true_data_filename, sep='\t', index_col=0)
    matrix = matrix.astype(int)
    df_out = pd.DataFrame(matrix)
    df_out.columns = df_in.columns
    df_out.index = df_in.index
    df_out.index_name = "cellIDxmutID"
    out_filename = "data/only_fpfn/" + file_prefix + ".SC"
    df_out.to_csv(out_filename, sep='\t')

file_prefixes = [
    # "simNo_10-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_5-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_10-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_5-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_10-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_5-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_10-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_5-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_10-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_5-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_10-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_5-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    "simNo_10-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_5-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_10-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_5-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_1-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_6-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_1-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_6-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_1-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_6-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_1-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_6-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_6-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_6-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    "simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_6-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_1-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    "simNo_6-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_2-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_7-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_2-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_7-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_2-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_7-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_2-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_7-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_2-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_7-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_2-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_7-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_2-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_7-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_2-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    "simNo_7-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_3-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_8-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_3-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_8-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_3-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_8-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_3-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_8-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_3-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_8-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_3-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_8-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    "simNo_3-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_8-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_3-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    "simNo_8-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_4-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_4-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_4-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_4-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_9-s_100-m_1000-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_4-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    # "simNo_4-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    # "simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_1000-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
    "simNo_4-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.05-na_0.05-d_0-l_1000000",
    "simNo_4-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000", 
    "simNo_9-s_100-m_300-h_1-minVAF_0.005-ISAV_0-n_300-fp_0.001-fn_0.2-na_0.05-d_0-l_1000000",
]

for file_prefix in file_prefixes:
    true_data, measured_data = load_file(file_prefix)
    print(f"size: {true_data.size}, file: {file_prefix}")
    # Set missing entries in measured data to true values (only considering false positives and negatives right now).
    missing_entries = (measured_data == 3)
    measured_data[missing_entries] = true_data[missing_entries]
    write_file(file_prefix, measured_data)

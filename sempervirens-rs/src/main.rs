extern crate blas_src;
extern crate openblas_src;

use std::ops::MulAssign;

use polars::prelude::*;
use ndarray::{prelude::*, ViewRepr};
use ndarray_stats::QuantileExt;
use ndarray::Ix;
use ndarray::azip;
use rayon::current_num_threads;
use rayon::prelude::*;
use clap::Parser;

mod file_prefixes;
use file_prefixes::*;

fn read_tsv(file_path: &str) -> DataFrame {
    let df = CsvReader::from_path(file_path).unwrap().with_delimiter(b'\t').finish().unwrap();
    let col_name = &df.get_column_names_owned()[0];
    df.drop(&col_name).unwrap()
}

fn is_conflict_free(mat: &Array2<f64>) -> bool {
    use itertools::Itertools;
    for pair in mat.columns().into_iter().combinations(2) {
        let a = pair[0];
        let b = pair[1];
        let intersection_size = (&a * &b).sum().round() as i64;
        let asize = a.sum().round() as i64;
        let bsize = b.sum().round() as i64;
        if intersection_size > 0 && intersection_size != asize && intersection_size != bsize {
            return false;
        }
    }
    true
}

fn ad_score(true_mat: &Array2<f64>, noisy_mat: &Array2<f64>) -> f64 {
    // assert_eq!(true_mat.ncols(), noisy_mat.ncols());
    // assert_eq!(true_mat.nrows(), noisy_mat.nrows());
    let mut num_true_ad_pairs: f64 = 0.0;
    let mut num_error_pairs: f64 = 0.0;
    for i in 0..true_mat.ncols() {
        for j in i..true_mat.ncols() {
            let intersection_size = (&true_mat.column(i) * &true_mat.column(j)).sum().round() as i64;
            if intersection_size == 0  {
                continue; // Disjoint
            }
            if intersection_size > 0 && true_mat.column(i).sum().round() as i64 == true_mat.column(j).sum().round() as i64 {
                continue; // Same edge
            }

            num_true_ad_pairs += 1.0;

            if (&noisy_mat.column(i) * &noisy_mat.column(j)).sum().round() as i64 == 0 {
                num_error_pairs += 1.0;
            } else {
                if true_mat.slice(s![.., j]).sum().round() as i64 > true_mat.column(i).sum().round() as i64 
                    && noisy_mat.column(j).sum().round() as i64 <= noisy_mat.column(i).sum().round() as i64 {
                    num_error_pairs += 1.0;
                } else if true_mat.column(i).sum().round() as i64 > true_mat.column(j).sum().round() as i64
                    && noisy_mat.column(i).sum().round() as i64 <= noisy_mat.column(j).sum().round() as i64 { 
                    num_error_pairs += 1.0;
                }
            }
        }
    }
    1.0 - num_error_pairs / num_true_ad_pairs
}

fn dl_score(true_mat: &Array2<f64>, reconstruction: &Array2<f64>) -> f64 {
    // assert_eq!(true_mat.ncols(), reconstruction.ncols());
    // assert_eq!(true_mat.nrows(), reconstruction.nrows());
    let mut diff_pairs: i64 = 0;
    let mut error_pairs: i64 = 0;
    for i in 0..true_mat.ncols() {
        for j in 1..true_mat.ncols() {
            if (&true_mat.column(i) * &true_mat.column(j)).sum().round() as i64 == 0  {
                diff_pairs += 1;
                if (&reconstruction.column(i) * &reconstruction.column(j)).sum().round() as i64 > 0 {
                    error_pairs += 1;
                }
            }
        }
    }
    if diff_pairs == 0 {
        1.0
    } else {
        1.0 - (error_pairs as f64) / (diff_pairs as f64)
    }
}

fn frac_correct(true_mat: &Array2<f64>, reconstruction: &Array2<f64>) -> f64 {
    let mut eq_count = 0;
    for i in 0..true_mat.nrows() {
        for j in 0..true_mat.ncols() {
            if true_mat[[i, j]].round() as i64 == reconstruction[[i, j]].round() as i64 {
                eq_count += 1;
            }
        }
    }
    (eq_count as f64) / (true_mat.len() as f64)
}

fn correct_pivot_subtree_columns(pivot_col: &Array1<f64>, mat: &Array2<f64>, _fpr: f64, _fnr: f64) -> Vec<Ix> {
    // assert_eq!(pivot_col.len(), mat.nrows());
    // let mat_cols = mat.select(Axis(1), cols);
    let k_11s = pivot_col.dot(mat);
    let k_01s = (1.0 - pivot_col).dot(mat);
    let mut cols_mask = Array1::zeros(mat.ncols());
    azip!((out in &mut cols_mask, &k_11 in &k_11s, &k_01 in &k_01s) *out = if k_11 >= k_01 { 1 } else { 0 });
    (0..mat.ncols()).filter(|i| cols_mask[*i] == 1).collect()
}

fn noisy_pivot_subtree_columns(pivot_col: &ArrayBase<ViewRepr<&f64>, Ix1>, cols: &[Ix], mat: &Array2<f64>, fpr: f64, fnr: f64) -> Vec<Ix> {
    let mat_cols = mat.select(Axis(1), cols);
    // assert_eq!(mat_cols.ncols(), cols.len());
    let k_11s = pivot_col.dot(&mat_cols);
    // assert_eq!(k_11s.len(), cols.len());
    let k_01s = (1.0 - pivot_col).dot(&mat_cols);
    let k_10s = pivot_col.dot(&(1.0 - mat_cols));
    let mut cols_mask = Array1::zeros(cols.len());
    if fnr >= fpr {
        let c0 = f64::ln((1.0-fnr)/fpr);
        let c1 = f64::ln((1.0-fpr)/fnr);
        let condf = |k_11: f64, k_01: f64, k_10: f64| k_11*c0 >= k_01*c1 || k_11*c0 >= k_10*c1;
        azip!((out in &mut cols_mask, &k_11 in &k_11s, &k_01 in &k_01s, &k_10 in &k_10s) *out = if condf(k_11, k_01, k_10) { 1 } else { 0 });
    } else {
        azip!((out in &mut cols_mask, &k_11 in &k_11s, &k_01 in &k_01s, &k_10 in &k_10s) *out = if k_11 >= k_01 || k_11 >= k_10 { 1 } else { 0 });
    }
    (0..cols.len()).filter(|i| cols_mask[*i] == 1).map(|i| cols[i]).collect()
}

fn reconstruct_root(cols_in_subtree: &[Ix], mat: &Array2<f64>, fnr: f64) -> Array1<f64> {
    let counts = mat.select(Axis(1), cols_in_subtree).sum_axis(Axis(0));
    // assert_eq!(counts.len(), cols_in_subtree.len());
    let max_count = *counts.max().unwrap() as f64;
    let mut root = Array1::zeros(mat.ncols());
    let mut cols_mask = Array1::zeros(cols_in_subtree.len());
    azip!((out in &mut cols_mask, &count in &counts) *out = if count as f64 >= max_count*(1.0-fnr) { 1 } else { 0 });
    for i in 0..cols_in_subtree.len() {
        if cols_mask[i] == 1 {
            root[cols_in_subtree[i]] = 1.0;
        }
    }
    root
}

fn reconstruct_pivot(cols_in_subtree: &[Ix], mat: &Array2<f64>) -> Array1<f64> {
    // for i in cols_in_subtree {
        // assert!(cols.contains(i));
    // }
    let subtree_in = mat.select(Axis(1), cols_in_subtree).sum_axis(Axis(1));
    // assert_eq!(subtree_in.len(), mat.nrows());
    
    let out_set: Vec<Ix> = (0..mat.ncols()).filter(|x| !cols_in_subtree.contains(x)).collect(); // TODO: this is n^2. Can do better by ensuring cols_in_subtree and cols are both sorted.
    let subtree_out = mat.select(Axis(1), &out_set).sum_axis(Axis(1));
    // assert_eq!(subtree_out.len(), mat.nrows());

    let mut reconstructed_pivot = Array1::zeros(mat.nrows());
    azip!((out in &mut reconstructed_pivot, &sin in &subtree_in, &sout in &subtree_out) *out = if sin >= sout && sin.round() as i64 > 0 { 1.0 } else { 0.0 });
    reconstructed_pivot
}

fn count_based_max_likelihood(pivot: &Array1<f64>, cols: &[Ix], mat: &Array2<f64>) -> Ix {
    // assert_eq!(pivot.len(), mat.nrows());
    let mat_cols = mat.select(Axis(1), cols);
    // assert_eq!(mat_cols.ncols(), cols.len());
    let counts = pivot.dot(&mat_cols);
    // assert_eq!(counts.len(), cols.len());
    cols[counts.argmax().unwrap()]
}

fn column_max_likelihood_refinement(reconstructed_mat: &ArrayBase<ViewRepr<&f64>, Ix2>, noisy_mat: &ArrayBase<ViewRepr<&f64>, Ix2>, fpr: f64, fnr: f64, mer: f64) -> Array2<f64> {
    // assert_eq!(reconstructed_mat.nrows(), noisy_mat.nrows());
    // assert_eq!(reconstructed_mat.ncols(), noisy_mat.ncols());
    let noisy0t = noisy_mat.mapv(|x| if x.round() as i64 == 0 { 1.0 } else { 0.0 }).reversed_axes(); 
    // assert!(noisy0t.nrows() == noisy_mat.ncols() && noisy0t.ncols() == noisy_mat.nrows());
    let noisy1t = noisy_mat.mapv(|x| if x.round() as i64 == 1 { 1.0 } else { 0.0 }).reversed_axes();
    let rmc = 1.0 - reconstructed_mat;
    let k_00s = noisy0t.dot(&rmc);
    // assert!(k_00s.nrows() == noisy_mat.ncols() && k_00s.ncols() == noisy_mat.ncols());
    let k_01s = noisy0t.dot(reconstructed_mat);
    let k_10s = noisy1t.dot(&rmc);
    let k_11s = noisy1t.dot(reconstructed_mat); 
    let log_probs = (1.0-fpr-mer).ln()*k_00s+ fnr.ln()*k_01s + fpr.ln()*k_10s + (1.0-fnr-mer).ln()*k_11s;
    // assert_eq!(log_probs.ncols(), noisy_mat.ncols());
    // assert_eq!(log_probs.nrows(), noisy_mat.ncols());
    let maximizers: Vec<Ix> = (0..noisy_mat.ncols()).map(|i| log_probs.row(i).argmax().unwrap()).collect();
    // assert_eq!(maximizers.len(), noisy_mat.ncols());
    let refinement = reconstructed_mat.select(Axis(1), &maximizers);
    refinement
}

fn split_part(mat: &mut Array2<f64>, fpr: f64, fnr: f64) -> (Vec<Vec<usize>>, Array1<f64>, usize) {
    
    let (pivot_col_i, cols_in_subtree) = {
        let mut partition = vec![];
        let mut temp_part: Vec<usize> = (0..mat.ncols()).collect();
        while temp_part.len() > 0 {
            let candidate_pivot_col_i = temp_part[0];
            let candidate_pivot = mat.column(candidate_pivot_col_i);
            let candidate_cols_in_subtree = noisy_pivot_subtree_columns(&candidate_pivot, &temp_part, mat, fpr, fnr);
            temp_part.retain(|x| !candidate_cols_in_subtree.contains(x));
            partition.push((candidate_pivot_col_i, candidate_cols_in_subtree));
        }
        let part_lengths: Vec<usize> = partition.iter().map(|&(_, ref part)| part.len()).collect();
        // assert_eq!(ndarray::arr1(&part_lengths).sum(), part.len());
        let part_i = ndarray::arr1(&part_lengths).argmax().unwrap();
        partition.remove(part_i)
    };

    let root = reconstruct_root(&cols_in_subtree, &mat, fnr);
    // assert_eq!(root.len(), mat.ncols());
    let halfrootsum = root.sum() / 2.0;
    let rows_in_subtree_mask = mat.dot(&root).mapv(|v| if v  > halfrootsum { 1 } else { 0 });
    // assert_eq!(rows_in_subtree_mask.len(), mat.nrows());
    for i in 0..mat.nrows() { 
        if rows_in_subtree_mask[i] == 1 {
            let row = &mut mat.row_mut(i);
            azip!((out in row, &rootval in &root) *out = if (*out).round() as i64 == 1 || rootval.round() as i64 == 1 { 1.0 } else { 0.0 });
        }
    }

    let reconstructed_pivot = reconstruct_pivot(&cols_in_subtree, &mat);

    let final_cols_in_subtree = correct_pivot_subtree_columns(&reconstructed_pivot, &mat, fpr, fnr);

    if final_cols_in_subtree.len() > 0 {
        let nonreconstructed_cols_out_subtree: Vec<Ix> = (0..mat.ncols()).filter(|x| !final_cols_in_subtree.contains(x)).collect();

        for i in &final_cols_in_subtree {
            let col = &mut mat.column_mut(*i);
            col.mul_assign(&reconstructed_pivot);
        }

        for i in &nonreconstructed_cols_out_subtree {
            let col = &mut mat.column_mut(*i);
            col.mul_assign(&(1.0 - &reconstructed_pivot));
        }

        let col_placement_i = count_based_max_likelihood(&reconstructed_pivot, &final_cols_in_subtree, &mat);

        let mut split_a = final_cols_in_subtree;
        split_a.retain(|&x| x != col_placement_i);

        let split_b = nonreconstructed_cols_out_subtree;

        // assert_eq!(split_a.len() + split_b.len() + 1, part.len());

        (vec![split_a, split_b], reconstructed_pivot, col_placement_i)
    } else {
        let col_placement_i = pivot_col_i;
        // let mut split_b = part.to_vec();
        // split_b.retain(|&x| x != col_placement_i);
        let split_b = (0..mat.ncols()).filter(|&x| x != col_placement_i).collect();
        (vec![split_b], reconstructed_pivot, col_placement_i)
    }
    
}

/// Reconstructs a phylogenetic tree from the noisy matrix.
/// 
/// Args:
///     noisy: N x M array. The noisy matrix to be corrected. 
///         Elements are of values 0, 1, or 3. 
///         Zero represents mutation absent, 1 represents mutation present,
///         and 3 represents missing entry.
///     fpr: False positive rate.
///     fnr: False negative rate.
///     mer: Missing entry rate.
/// 
/// Returns:
///     N x M matrix that represents the reconstructed phylogenetic tree.
///     In this matrix, for any two columns, either one is a subset of the other or they are disjoint.
fn reconstruct(noisy: &Array2<f64>, fpr: f64, fnr: f64, mer: f64) -> Array2<f64> {
    assert!(
        noisy.mapv(|v| v.round() as i64 == 0 || v.round() as i64 == 1 || v.round() as i64 == 3).fold(true, |a, &b| a && b),
        "Elements in the noisy matrix may only be 0 (absent), 1 (present), or 3 (missing)."
    );

    let mut mat = noisy.clone();
    mat.par_mapv_inplace(|v| if v.round() as i64 == 3 { 0.0 } else { v });

    let cols_sorted: Vec<usize> = {
        let col_sums = mat.sum_axis(Axis(0));
        // assert_eq!(col_sums.len(), noisy.ncols());
        let mut sorted = col_sums.iter().enumerate().collect::<Vec<_>>();
        // assert_eq!(sorted.len(), noisy.ncols());
        sorted.sort_unstable_by_key(|v| (-v.1).round() as i64);
        // sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap().reverse());// -v.1);
        sorted.iter().map(|&(i, _)| i).collect()
    };
    // assert_eq!(cols_sorted.len(), noisy.ncols());

    let mut partition = vec![cols_sorted.clone()];

    while partition.len() > 0 {
        let (subpartitions, parts_columns): (Vec<Vec<Vec<usize>>>, Vec<(&Vec<usize>, Array2<f64>)>) = partition.par_iter()
            .map(|part| (part, mat.select(Axis(1), &part)))
            .map(|(part, mut columns)| {
                let (mut subpartition, reconstructed_pivot, col_i) = split_part(&mut columns, fpr, fnr);
                columns.column_mut(col_i).assign(&reconstructed_pivot);
                for ref mut subpart in subpartition.iter_mut() {
                    for ref mut i in subpart.iter_mut() {
                        **i = part[**i];
                    }
                }
                // let subpartition = subpartition.into_iter().map(|subpart| subpart.into_iter().map(|i| part[i]).collect::<Vec<_>>()).collect::<Vec<_>>();
                (subpartition, (part, columns))
            })
            .unzip();

        for &(ref part, ref columns) in parts_columns.iter() {
            for (ref column, &i) in columns.columns().into_iter().zip(part.iter()) {
                mat.column_mut(i).assign(column);
            }
        }

        partition = subpartitions.into_iter().flatten().filter(|part| !part.is_empty()).collect();
    }

    mat = column_max_likelihood_refinement(&mat.t(), &noisy.t(), fpr, fnr, mer).reversed_axes();
    mat = column_max_likelihood_refinement(&mat.view(), &noisy.view(), fpr, fnr, mer);

    assert!(mat.mapv(|v| v.round() as i64 == 0 || v.round() as i64 == 1).fold(true, |a, &b| a && b));

    mat
}

fn print_metrics(metrics_df: DataFrame) {
    let ns = metrics_df.column("n").unwrap().unique().unwrap().rechunk();
    let ms = metrics_df.column("m").unwrap().unique().unwrap().rechunk();
    let fprs = metrics_df.column("fpr").unwrap().unique().unwrap().rechunk();
    let fnrs = metrics_df.column("fnr").unwrap().unique().unwrap().rechunk();
    let mers = metrics_df.column("mer").unwrap().unique().unwrap().rechunk();
    for n in ns.iter().map(|n| n.try_extract::<i64>().unwrap()) {
        for m in ms.iter().map(|m| m.try_extract::<i64>().unwrap()) {
            for fpr in fprs.iter().map(|fpr| fpr.try_extract::<f64>().unwrap()) {
                for fnr in fnrs.iter().map(|fnr| fnr.try_extract::<f64>().unwrap()) {
                    for mer in mers.iter().map(|mer| mer.try_extract::<f64>().unwrap()) {
                        // println!("{n}, {m}, {fpr}, {fnr}, {mer}");
                        let df = metrics_df.clone().lazy()
                            .filter(col("n").eq(lit(n)))
                            .filter(col("m").eq(lit(m)))
                            .filter(col("fpr").eq(lit(fpr)))
                            .filter(col("fnr").eq(lit(fnr)))
                            .filter(col("mer").eq(lit(mer)))
                            .collect()
                            .unwrap()
                            .select(["time", "is_ptree", "ad_score", "dl_score", "fraction_correct"])
                            .unwrap();

                        if df.height() > 0 {
                            let times = df.column("time").unwrap();
                            let is_conflict_frees = df.column("is_ptree").unwrap();
                            let ad_scores = df.column("ad_score").unwrap();
                            let dl_scores = df.column("dl_score").unwrap();
                            let fraction_corrects = df.column("fraction_correct").unwrap();
                            println!("* n: {n}    m: {m}    fpr: {fpr}    fnr: {fnr}    mer: {mer}");
                            println!("    * Average: [{}, {:.3}, {:.3}, {:.3}]",
                                is_conflict_frees.mean().unwrap(), ad_scores.mean().unwrap(), dl_scores.mean().unwrap(), fraction_corrects.mean().unwrap());
                            // println!("    * Std Dev: [{}, {:.3}, {:.3}, {:.3}]",
                                // is_conflict_frees.std_as_series()., ad_scores.mean().unwrap(), dl_scores.mean().unwrap(), fraction_corrects.mean().unwrap());
                            println!("    * Min      [{}, {:.3}, {:.3}, {:.3}]",
                                is_conflict_frees.min::<i8>().unwrap(), ad_scores.min::<f64>().unwrap(), dl_scores.min::<f64>().unwrap(), fraction_corrects.min::<f64>().unwrap());
                            println!("    * Max:     [{}, {:.3}, {:.3}, {:.3}]",
                                is_conflict_frees.max::<i8>().unwrap(), ad_scores.max::<f64>().unwrap(), dl_scores.max::<f64>().unwrap(), fraction_corrects.max::<f64>().unwrap());
                            println!("    * Time: {:.3} average, {:.3} min, {:.3} max", times.mean().unwrap(), times.min::<f64>().unwrap(), times.max::<f64>().unwrap());
                            println!("");
                            // println!("{}", df.describe(None).unwrap());
                        }
                    }
                }
            }
        }
    }
}

fn run(file_prefixes: &[(&'static str, i64, i64, f64, f64, f64)]) {
    let mut files = vec![];
    let mut ns = vec![];
    let mut ms = vec![];
    let mut fprs = vec![];
    let mut fnrs = vec![];
    let mut mers = vec![];
    let mut times = vec![];
    let mut is_conflict_frees: Vec<bool> = vec![];
    let mut ad_scores: Vec<f64> = vec![];
    let mut dl_scores: Vec<f64> = vec![];
    let mut frac_corrects: Vec<f64> = vec![];

    for &(file_prefix, n, m, fpr, fnr, mer) in file_prefixes {
        println!("Running: {file_prefix}");

        let true_filename = "../data/".to_string() + file_prefix + ".SC.before_FP_FN_NA";
        let true_df = read_tsv(&true_filename);
        let true_mat = true_df.to_ndarray::<Int32Type>(IndexOrder::C).unwrap().mapv(f64::from);

        let noisy_filename = "../data/".to_string() + file_prefix + ".SC";
        let noisy_df = read_tsv(&noisy_filename);
        let noisy_mat = noisy_df.to_ndarray::<Int32Type>(IndexOrder::C).unwrap().mapv(f64::from);

        let t0 = std::time::Instant::now();
        let reconstruction = reconstruct(&noisy_mat, fpr, fnr, mer);
        let tf = std::time::Instant::now();

        let is_cf = is_conflict_free(&reconstruction);
        let ad_score = ad_score(&true_mat, &reconstruction);
        let dl_score = dl_score(&true_mat, &reconstruction);
        let frac_correct = frac_correct(&true_mat, &reconstruction);
        let time = tf.duration_since(t0).as_secs_f64();

        files.push(file_prefix);
        ns.push(n);
        ms.push(m);
        fprs.push(fpr);
        fnrs.push(fnr);
        mers.push(mer);
        times.push(time);
        is_conflict_frees.push(is_cf);
        ad_scores.push(ad_score);
        dl_scores.push(dl_score);
        frac_corrects.push(frac_correct);
        
        println!("[{}, {:.3}, {:.3}, {:.3}]", is_cf, ad_score, dl_score, frac_correct);
    }

    // let times_series = Series::new("time", times);
    // println!("times: mean {:.3}, min {:.3}, max {:.3}", times_series.mean().unwrap(), times_series.min::<f64>().unwrap(), times_series.max::<f64>().unwrap());

    let metrics_df = df!(
        "file" => &files, "n" => &ns, "m" => &ms, "fpr" => &fprs, "fnr" => &fnrs, "mer" => &mers, "time" => &times,
        "is_ptree" => &is_conflict_frees, "ad_score" => &ad_scores, "dl_score" => &dl_scores, "fraction_correct" => &frac_corrects
    ).unwrap();

    print_metrics(metrics_df);

}

#[derive(Parser)]
#[command(author, version, about, long_about = "Sempervirens: Reconstruction of phylogenetic trees from noisy data.")]
struct Args {
    /// Input file to read noisy matrix from.
    in_file: String,

    /// False positive rate.
    fpr: f64,

    /// False negative rate.
    fnr: f64,

    /// Missing entry rate.
    mer: f64,

    /// Output file to write conflict-free matrix to. Defaults to IN_FILE.CFMatrix.
    #[arg(short, long)]
    out_file: Option<String>,

    /// Number of threads to use in threadpool. Defaults to number of cpu cores.
    #[arg(short, long)]
    num_threads: Option<usize>,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    let verbose = args.verbose;

    let in_file = args.in_file;

    let fpr = args.fpr;
    let fnr = args.fnr;
    let mer = args.mer;

    // Basic, but insufficient, checks.
    assert!(0.0 <= fpr && fpr <= 1.0, "FPR must be in [0, 1].");
    assert!(0.0 <= fnr && fnr <= 1.0, "FNR must be in [0, 1].");
    assert!(0.0 <= mer && mer <= 1.0, "MER must be in [0, 1].");
    assert!(fpr + mer <= 1.0, "FPR + MER must be in [0, 1].");
    assert!(fnr + mer <= 1.0, "FNR + MER must be in [0, 1].");

    let out_file = args.out_file.unwrap_or(in_file.clone() + ".CFMatrix");
    let mut write_file = std::fs::File::create(&out_file).expect("Could not open OUT_FILE to write to.");

    if let Some(num_threads) = args.num_threads {
        assert!(num_threads > 0, "Number of threads NUM_THREADS must be greater than 0.");
        rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().expect("Failed to build thread pool");
    }

    if verbose {
        println!("Number of threads: {}.", current_num_threads());
        println!("Reading from: {in_file}.");
        println!("False positive rate: {fpr}, false negative rate: {fnr}, missing entry rate: {mer}.");
        println!("Will output reconstruction to {out_file}.");
    }

    let noisy_df = CsvReader::from_path(in_file).unwrap().with_delimiter(b'\t').finish().unwrap();
    let col_name = &noisy_df.get_column_names_owned()[0];
    let noisy_mat = noisy_df.drop(&col_name).unwrap().to_ndarray::<Float64Type>(IndexOrder::C).unwrap(); //.mapv(f64::from);

    let reconstruction = reconstruct(&noisy_mat, fpr, fnr, mer);

    let seriess: Vec<Series> = std::iter::once(noisy_df.column(col_name).unwrap().to_owned()) // Add first column.
        .chain(reconstruction.columns().into_iter()
            .map(|c| c.mapv(|v| v.round() as i64).to_vec()) // Convert floats to ints.
            .zip(noisy_df.get_column_names().iter().skip(1)) // Get names of original columns, skipping first (which corresponds to names of rows).
            .map(|(column, name)| Series::new(name, column)) // Assign column names to data.
        )
        .collect();
    let mut write_df = DataFrame::new(seriess).unwrap();
    CsvWriter::new(&mut write_file).has_header(true).with_delimiter(b'\t').finish(&mut write_df).expect("Failed to write reconstructed matrix.");

    // let mut files = vec![];
    // files.extend_from_slice(&FILE_PREFIXES_300X300S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X300S_0_003FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X300S_0_01FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X300S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X1000S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X1000S_0_01FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X1000S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X1000S_0_01FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X10000S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_2000X20000S_0_001FPR);
    // run(&files);
}

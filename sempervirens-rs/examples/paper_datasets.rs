use polars::prelude::*;
use ndarray::prelude::*;

extern crate sempervirens_rs;
use sempervirens_rs::reconstruct;

mod file_prefixes;
use file_prefixes::*;

// Read tab separated value (TSV) file and ignore index column.
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

// Ancestor-descendant score.
fn ad_score(true_mat: &Array2<f64>, noisy_mat: &Array2<f64>) -> f64 {
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

// Different-lineage score.
fn dl_score(true_mat: &Array2<f64>, reconstruction: &Array2<f64>) -> f64 {
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

// Fraction of entries in reconstruction which match ground-truth.
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

    let metrics_df = df!(
        "file" => &files, "n" => &ns, "m" => &ms, "fpr" => &fprs, "fnr" => &fnrs, "mer" => &mers, "time" => &times,
        "is_ptree" => &is_conflict_frees, "ad_score" => &ad_scores, "dl_score" => &dl_scores, "fraction_correct" => &frac_corrects
    ).unwrap();

    print_metrics(metrics_df);

}

fn main() {
    let mut files = vec![];
    files.extend_from_slice(&FILE_PREFIXES_300X300S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X300S_0_003FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X300S_0_01FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X300S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X1000S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_300X1000S_0_01FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X1000S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X1000S_0_01FPR);
    // files.extend_from_slice(&FILE_PREFIXES_1000X10000S_0_001FPR);
    // files.extend_from_slice(&FILE_PREFIXES_2000X20000S_0_001FPR);
    run(&files);
}
use polars::prelude::*;
use rayon::current_num_threads;
use clap::Parser;

use sempervirens_rs::reconstruct;


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

    // Basic checks on fpr, fnr, and mer.
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

    // Read noisy matrix from file.
    let noisy_df = CsvReader::from_path(in_file).unwrap().with_delimiter(b'\t').finish().unwrap();
    let col_name = &noisy_df.get_column_names_owned()[0];
    let noisy_mat = noisy_df.drop(&col_name).unwrap().to_ndarray::<Float64Type>(IndexOrder::C).unwrap();

    // Reconstruct matrix.
    let reconstruction = reconstruct(&noisy_mat, fpr, fnr, mer);

    // Write reconstructed matrix to file.
    let seriess: Vec<Series> = std::iter::once(noisy_df.column(col_name).unwrap().to_owned()) // Add first column.
        .chain(reconstruction.columns().into_iter()
            .map(|c| c.mapv(|v| v.round() as i64).to_vec()) // Convert floats to ints.
            .zip(noisy_df.get_column_names().iter().skip(1)) // Get names of original columns, skipping first (which corresponds to names of rows).
            .map(|(column, name)| Series::new(name, column)) // Assign column names to data.
        )
        .collect();
    let mut write_df = DataFrame::new(seriess).unwrap();
    CsvWriter::new(&mut write_file).has_header(true).with_delimiter(b'\t').finish(&mut write_df).expect("Failed to write reconstructed matrix.");
}

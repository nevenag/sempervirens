extern crate blas_src;
extern crate openblas_src;

use std::ops::MulAssign;

use ndarray::{prelude::*, ViewRepr, Ix, azip};
use ndarray_stats::QuantileExt;
use rayon::prelude::*;

fn correct_pivot_subtree_columns(pivot_col: &Array1<f64>, mat: &Array2<f64>, _fpr: f64, _fnr: f64) -> Vec<Ix> {
    let k_11s = pivot_col.dot(mat);
    let k_01s = (1.0 - pivot_col).dot(mat);
    let mut cols_mask = Array1::zeros(mat.ncols());
    azip!((out in &mut cols_mask, &k_11 in &k_11s, &k_01 in &k_01s) *out = if k_11 >= k_01 { 1 } else { 0 });
    (0..mat.ncols()).filter(|i| cols_mask[*i] == 1).collect()
}

fn noisy_pivot_subtree_columns(pivot_col: &ArrayBase<ViewRepr<&f64>, Ix1>, cols: &[Ix], mat: &Array2<f64>, fpr: f64, fnr: f64) -> Vec<Ix> {
    let mat_cols = mat.select(Axis(1), cols);
    let k_11s = pivot_col.dot(&mat_cols);
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
    let subtree_in = mat.select(Axis(1), cols_in_subtree).sum_axis(Axis(1));
    
    let out_set: Vec<Ix> = (0..mat.ncols()).filter(|x| !cols_in_subtree.contains(x)).collect(); // TODO: this is n^2. Can do better by ensuring cols_in_subtree and cols are both sorted.
    let subtree_out = mat.select(Axis(1), &out_set).sum_axis(Axis(1));

    let mut reconstructed_pivot = Array1::zeros(mat.nrows());
    azip!((out in &mut reconstructed_pivot, &sin in &subtree_in, &sout in &subtree_out) *out = if sin >= sout && sin.round() as i64 > 0 { 1.0 } else { 0.0 });
    reconstructed_pivot
}

fn count_based_max_likelihood(pivot: &Array1<f64>, cols: &[Ix], mat: &Array2<f64>) -> Ix {
    let mat_cols = mat.select(Axis(1), cols);
    let counts = pivot.dot(&mat_cols);
    cols[counts.argmax().unwrap()]
}

fn column_max_likelihood_refinement(reconstructed_mat: &ArrayBase<ViewRepr<&f64>, Ix2>, noisy_mat: &ArrayBase<ViewRepr<&f64>, Ix2>, fpr: f64, fnr: f64, mer: f64) -> Array2<f64> {
    let noisy0t = noisy_mat.mapv(|x| if x.round() as i64 == 0 { 1.0 } else { 0.0 }).reversed_axes();
    let noisy1t = noisy_mat.mapv(|x| if x.round() as i64 == 1 { 1.0 } else { 0.0 }).reversed_axes();
    let rmc = 1.0 - reconstructed_mat;
    let k_00s = noisy0t.dot(&rmc);
    let k_01s = noisy0t.dot(reconstructed_mat);
    let k_10s = noisy1t.dot(&rmc);
    let k_11s = noisy1t.dot(reconstructed_mat); 
    let log_probs = (1.0-fpr-mer).ln()*k_00s+ fnr.ln()*k_01s + fpr.ln()*k_10s + (1.0-fnr-mer).ln()*k_11s;
    let maximizers: Vec<Ix> = (0..noisy_mat.ncols()).map(|i| log_probs.row(i).argmax().unwrap()).collect();
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
        let part_i = ndarray::arr1(&part_lengths).argmax().unwrap();
        partition.remove(part_i)
    };

    let root = reconstruct_root(&cols_in_subtree, &mat, fnr);
    let halfrootsum = root.sum() / 2.0;
    let rows_in_subtree_mask = mat.dot(&root).mapv(|v| if v  > halfrootsum { 1 } else { 0 });
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

        (vec![split_a, split_b], reconstructed_pivot, col_placement_i)
    } else {
        let col_placement_i = pivot_col_i;
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
pub fn reconstruct(noisy: &Array2<f64>, fpr: f64, fnr: f64, mer: f64) -> Array2<f64> {
    assert!(
        noisy.mapv(|v| v.round() as i64 == 0 || v.round() as i64 == 1 || v.round() as i64 == 3).fold(true, |a, &b| a && b),
        "Elements in the noisy matrix may only be 0 (absent), 1 (present), or 3 (missing)."
    );

    let mut mat = noisy.clone();
    mat.par_mapv_inplace(|v| if v.round() as i64 == 3 { 0.0 } else { v });

    let cols_sorted: Vec<usize> = {
        let col_sums = mat.sum_axis(Axis(0));
        let mut sorted = col_sums.iter().enumerate().collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|v| (-v.1).round() as i64);
        sorted.iter().map(|&(i, _)| i).collect()
    };

    let mut partition = vec![cols_sorted.clone()];

    while partition.len() > 0 {
        // TODO: this parallelizes over the current partition and then groups the result, before starting the next set.
        // Could be better parallelized by starting the next task as each split_part finishes.
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

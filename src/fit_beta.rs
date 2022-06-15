use ndarray::{arr1, Array1, ArrayView1, ArrayView2};
use ordered_float::OrderedFloat;
use pyo3::number::int;
use rand::Rng;

pub fn fit_beta(
    y: &ArrayView2<'_, f64>,
    x: &ArrayView2<'_, f64>,
    normalization_factors: &ArrayView2<'_, f64>,
    alpha_hat: &ArrayView1<'_, f64>,
    contrast: &ArrayView1<'_, f64>,
    beta_matrix: &ArrayView2<'_, f64>,
    lambda: &ArrayView1<'_, f64>,
    weights: &ArrayView2<'_, f64>,
    use_weights: bool,
    tolerance: f64,
    max_iterations: usize,
    use_qr_decomposition: bool,
) -> Array1<f64> {
    let y_n = y.nrows();
    let y_m = y.ncols();
    let x_p = x.ncols();

    let k = contrast.len();
    let mut beta_matrix = beta_matrix.to_owned();
}

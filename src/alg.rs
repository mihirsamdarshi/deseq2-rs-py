use nalgebra::{Const, DMatrix, Dynamic, Matrix, SliceStorage, VecStorage};
use ndarray::{ArrayView1, ArrayView2};
use nshare::ToNalgebra;

struct FitBetaResult {}

/// Fit beta coefficients for negative binomial GLM
pub fn fit_beta(
    y: ArrayView2<'_, f64>,
    x: ArrayView2<'_, f64>,
    normalization_factors: ArrayView2<'_, f64>,
    alpha_hat: ArrayView1<'_, f64>,
    contrast: ArrayView1<'_, f64>,
    beta_matrix: ArrayView2<'_, f64>,
    lambda: ArrayView1<'_, f64>,
    weights: ArrayView2<'_, f64>,
    use_weights: bool,
    tolerance: f64,
    max_iterations: u16,
    use_qr_decomposition: bool,
    min_mu: f64,
) {
    let y = y.into_nalgebra();
    let x = x.into_nalgebra();
    let normalization_factors = normalization_factors.into_nalgebra();
    let alpha_hat = alpha_hat.into_nalgebra();
    let mut beta_matrix = beta_matrix.into_nalgebra();
    let lambda = lambda.into_nalgebra();
    let weights = weights.into_nalgebra();

    let y_n = y.nrows();
    let y_m = y.ncols();
    let x_p = x.ncols();

    let k = contrast.len();

    let beta_var_mat = DMatrix::<f64>::zeros(beta_matrix.nrows(), beta_matrix.ncols());
    let contrast_num = DMatrix::<f64>::zeros(beta_matrix.nrows(), 1);
    let contrast_denom = DMatrix::<f64>::zeros(beta_matrix.nrows(), 1);

    let mut iter = vec![0; y_n];
    let deviance = vec![0.0; y_n];

    for i in 0..y_n {
        let nfrow = normalization_factors.row(i).transpose();
        let yrow = y.row(i).transpose();
        let beta_hat = beta_matrix.row(i).transpose();
        let mut mu_hat = nfrow * (x.dot(&beta_hat)).exp();

        for j in 0..y_m {
            mu_hat[j] = mu_hat.get(j).unwrap().max(min_mu);
        }

        let ridge = Matrix::from_diagonal(&lambda);
        let dev = 0.0;
        let dev_old = 0.0;

        for t in 0..max_iterations {
            iter[i] += 1;
            let (w_vec, w_sqrt_vec) = get_weights(use_weights, weights, i, &mu_hat, alpha_hat);
        }
    }
}

type F64Matrix<R, C, S> = Matrix<f64, R, C, S>;
type StdMatrix<C, S> = F64Matrix<Dynamic, C, S>;
type SingleRowStorage = VecStorage<f64, Dynamic, Const<1>>;
type SingleRowStdMatrix<S> = StdMatrix<Const<1>, S>;
type SingleRowStdStorageMatrix = SingleRowStdMatrix<SingleRowStorage>;

fn get_weights(
    use_weights: bool,
    weights: StdMatrix<Dynamic, SliceStorage<f64, Dynamic, Dynamic, Dynamic, Dynamic>>,
    i: usize,
    mu_hat: &SingleRowStdStorageMatrix,
    alpha_hat: SingleRowStdMatrix<SliceStorage<f64, Dynamic, Const<1>, Const<1>, Dynamic>>,
) -> (SingleRowStdStorageMatrix, SingleRowStdStorageMatrix) {
    return if use_weights {
        let w_vec = weights.row(i).transpose().component_mul(
            &mu_hat.component_div(
                &(alpha_hat
                    .get(i)
                    .expect("Unable to get alpha_hat value")
                    .to_owned()
                    * mu_hat)
                    .add_scalar(1.0),
            ),
        );
        let w_sqrt_vec = w_vec.map(|x| x.sqrt());
        (w_vec, w_sqrt_vec)
    } else {
        let w_vec = mu_hat.component_div(
            &((alpha_hat
                .get(i)
                .expect("Unable to get alpha_hat value")
                .to_owned()
                * mu_hat)
                .add_scalar(1.0)),
        );
        let w_sqrt_vec = w_vec.map(|x| x.sqrt());
        (w_vec, w_sqrt_vec)
    };
}

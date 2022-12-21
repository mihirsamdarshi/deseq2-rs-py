use std::fmt::Debug;

use nalgebra::{DMatrix, DMatrixSlice, DVector, Dim, Dynamic, Matrix, OMatrix, Scalar, Storage};
use nalgebra_lapack::{LU, QR};
use num_traits::Zero;

use crate::math::rf_dnbinom_mu;

/// Fit beta coefficients for negative binomial GLM
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn fit_beta(
    y: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    x: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    normalization_factors: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    alpha_hat: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    contrast: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    beta_matrix: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    lambda: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    weights: DMatrixSlice<'_, f64, Dynamic, Dynamic>,
    use_weights: bool,
    tolerance: f64,
    max_iterations: u16,
    use_qr_decomposition: bool,
    min_mu: f64,
) -> FitBetaResult {
    let y_n = y.nrows();
    let y_m = y.ncols();
    let x_p = x.ncols();

    let mut beta_matrix = beta_matrix.clone_owned();

    let contrast = contrast.fixed_columns::<1>(0);
    let lambda = lambda.fixed_columns::<1>(0);

    let mut beta_var_matrix = DMatrix::<f64>::zeros(beta_matrix.nrows(), beta_matrix.ncols());
    let mut contrast_num = DMatrix::<f64>::zeros(beta_matrix.nrows(), 1);
    let mut contrast_denom = DMatrix::<f64>::zeros(beta_matrix.nrows(), 1);
    let mut hat_diagonals = DMatrix::<f64>::zeros(y.nrows(), y.ncols());

    let mut iter = vec![0; y_n];
    let mut deviance = vec![0.0; y_n];

    let large = 30.0;

    for i in 0..y_n {
        let nfrow = normalization_factors.row(i).transpose();
        let yrow = y.row(i).transpose();
        let beta_hat = beta_matrix.row(i).transpose();

        let mut mu_hat = nfrow.component_mul(&((x * &beta_hat).map(f64::exp)));
        mu_hat.rows_range_mut(0..y_m).apply(|x| {
            if *x < min_mu {
                *x = min_mu;
            }
        });

        let reset_mu = mu_hat.clone_owned();

        let ridge = Matrix::from_diagonal(&lambda);
        let mut dev = 0.0;
        let mut dev_old = 0.0;
        let mut num_iterations = 0;

        for t in 0..max_iterations {
            let (w_vec, w_sqrt_vec) = get_weights(use_weights, weights, i, &mu_hat, alpha_hat);
            let x_wvec = multiply_each_column(&x, &w_sqrt_vec);
            let ridge_sqrt = ridge.map(f64::sqrt);
            let weighted_x_ridge = join_matrix_cols(&x_wvec, &ridge_sqrt);
            let (q, r) = QR::new(weighted_x_ridge).unpack();

            let mut big_w_diag = w_vec.clone();
            big_w_diag.resize_vertically_mut(y_m + x_p, 1.0);

            let z_sqrt_w = multiply_each_column(
                &(mu_hat.component_div(&nfrow) + (&yrow - &mu_hat)).component_div(&mu_hat),
                &w_vec,
            );
            let mut big_z_sqrt_w = z_sqrt_w.clone();
            big_z_sqrt_w.resize_vertically_mut(y_m + x_p, 0.0);

            let gamma_hat = q * big_z_sqrt_w;

            let beta_hat = if let Some(x) = LU::new(r).solve(&gamma_hat) {
                x
            } else {
                eprintln!("Unable to solve for beta_hat");
                beta_hat.clone()
            };

            if beta_hat
                .abs()
                .fold(0.0, |acc, x| if x > large { acc + x } else { acc })
                > 0.0
            {
                eprintln!("beta_hat is zero");
                num_iterations = max_iterations;
                break;
            }

            let mu_hat = reset_mu.clone();

            dev = 0.0;
            for j in 0..y_m {
                // note the order for Rf_dnbinom_mu: x, sz, mu, lg
                if use_weights {
                    dev += -2.0
                        * weights[(i, j)]
                        * rf_dnbinom_mu(yrow[j], 1.0 / alpha_hat[i], mu_hat[j], true);
                } else {
                    dev += -2.0 * rf_dnbinom_mu(yrow[j], 1.0 / alpha_hat[i], mu_hat[j], true);
                }
            }

            let conv_test = (dev - dev_old).abs() / (dev.abs() + 0.1);

            if conv_test.is_nan() {
                eprintln!("deviance is NAN");
                num_iterations = max_iterations;
                break;
            }

            if t > 0 && conv_test < tolerance {
                break;
            }
            dev_old = dev;
        }

        iter[i] = num_iterations;
        deviance[i] = dev;
        beta_matrix.set_row(i, &beta_hat.transpose());
        // recalculate w so that this is identical if we start with beta_hat
        let (w_vec, w_sqrt_vec) = get_weights(use_weights, weights, i, &mu_hat, alpha_hat);

        // arma::mat xw = x.each_col() % w_sqrt_vec;
        let xw = multiply_each_column(&x, &w_sqrt_vec);
        let xsw = multiply_each_column(&x, &w_vec);
        // arma::mat xtwxr_inv = (x.t() * (x.each_col() % w_vec) + ridge).i();
        let xtwxr_inv = ((x.transpose() * &xsw) + &ridge)
            .try_inverse()
            .expect("Unable to invert matrix");

        let hat_matrix_diag = (&xw * xtwxr_inv * xw.transpose()).diagonal();

        hat_diagonals.set_row(i, &hat_matrix_diag.transpose());

        let cov_m = (x.transpose() * (&xsw) + &ridge)
            .try_inverse()
            .expect("Unable to invert matrix");

        let sigma = &cov_m * x.transpose() * (&xsw) * &cov_m;

        contrast_num.set_row(i, &(contrast.transpose() * beta_hat));
        contrast_denom.set_row(
            i,
            &(contrast.transpose() * &sigma * contrast).map(f64::sqrt),
        );
        beta_var_matrix.set_row(i, &sigma.diagonal().transpose());
    }

    FitBetaResult {
        beta_matrix,
        beta_var_matrix,
        iterations: iter,
        hat_diagonals,
        contrast_num,
        contrast_denom,
        deviance,
    }
}

fn multiply_each_column<C: Dim, S>(
    a: &Matrix<f64, Dynamic, C, S>,
    b: &DVector<f64>,
) -> OMatrix<f64, Dynamic, C>
where
    S: Storage<f64, Dynamic, C>,
{
    let mut c = a.clone_owned();
    for mut col in c.column_iter_mut() {
        col.component_mul_assign(b);
    }
    c
}

#[allow(dead_code)]
pub struct FitBetaResult {
    beta_matrix: DMatrix<f64>,
    beta_var_matrix: DMatrix<f64>,
    iterations: Vec<u16>,
    hat_diagonals: DMatrix<f64>,
    contrast_num: DMatrix<f64>,
    contrast_denom: DMatrix<f64>,
    deviance: Vec<f64>,
}

/// Concatenates two matrices by columns
fn join_matrix_cols<T: Copy + Debug + Scalar + Zero>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T> {
    let mut c = DMatrix::zeros(a.nrows(), a.ncols() + b.ncols());
    c.fixed_columns_mut::<1>(0).copy_from(a);
    c.fixed_columns_mut::<1>(a.ncols()).copy_from(b);
    c
}

fn get_weights(
    use_weights: bool,
    weights: DMatrixSlice<f64, Dynamic>,
    i: usize,
    mu_hat: &DVector<f64>,
    alpha_hat: DMatrixSlice<f64, Dynamic>,
) -> (DVector<f64>, DVector<f64>) {
    return if use_weights {
        let w_vec = weights
            .row(i)
            .transpose()
            .component_mul(&mu_hat.component_div(&(alpha_hat[i] * mu_hat).add_scalar(1.0)));
        let w_sqrt_vec = w_vec.map(f64::sqrt);
        (w_vec, w_sqrt_vec)
    } else {
        let w_vec = mu_hat.component_div(&((alpha_hat[i] * mu_hat).add_scalar(1.0)));
        let w_sqrt_vec = w_vec.map(f64::sqrt);
        (w_vec, w_sqrt_vec)
    };
}

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

mod alg;
mod math;

// NOTE
// * numpy defaults to np.float64, if you use other type than f64 in Rust you
//   will have to change type in Python before calling the Rust function.

// The name of the module must be the same as the rust package name
#[pymodule]
fn deseq2_rs_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// Fit beta coefficients for negative binomial GLM
    ///
    /// This function estimates the coefficients (betas) for negative binomial
    /// generalized linear models.
    ///
    /// - `y` - ***n*** by ***m*** matrix of counts
    /// - `x` - ***m*** by ***k*** design matrix
    /// - `normalization_factors` - ***n*** by ***m*** matrix of normalization
    ///   factors
    /// - `alpha_hat` - ***n*** length vector of the disperion estimates
    /// - `contrast` - a ***k*** length vector for a possible contrast
    /// - `beta_matrix` - ***n*** by ***k*** matrix of the initial estimates for
    ///   the betas
    /// - `lambda` - ***k*** length vector of the ridge values
    /// - `weights` - ***n*** by ***m*** matrix of weights
    /// - `use_weights` - whether to use weights
    /// - `tolerance` - tolerance for convergence in estimates
    /// - `max_iterations` - maximum number of iterations
    /// - `use_qr_decomposition` - whether to use QR decomposition
    #[pyfn(m)]
    fn fit_beta<'py>(
        py: Python<'py>,
        y: PyReadonlyArray2<f64>,
        x: PyReadonlyArray2<f64>,
        normalization_factors: PyReadonlyArray2<f64>,
        alpha_hat: PyReadonlyArray1<f64>,
        contrast: PyReadonlyArray1<f64>,
        beta_matrix: PyReadonlyArray2<f64>,
        lambda: PyReadonlyArray1<f64>,
        weights: PyReadonlyArray2<f64>,
        use_weights: bool,
        tolerance: f64,
        max_iterations: u16,
        use_qr_decomposition: bool,
        min_mu: f64,
    ) {
        // Simple demonstration of creating an ndarray inside Rust and return
        let array = alg::fit_beta(
            y.as_matrix(),
            x.as_matrix(),
            normalization_factors.as_matrix(),
            alpha_hat.as_matrix(),
            contrast.as_matrix(),
            beta_matrix.as_matrix(),
            lambda.as_matrix(),
            weights.as_matrix(),
            use_weights,
            tolerance,
            max_iterations,
            use_qr_decomposition,
            min_mu,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

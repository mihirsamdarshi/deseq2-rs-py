mod fit_beta;

use ndarray;
use ndarray::array;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// NOTE
// * numpy defaults to np.float64, if you use other type than f64 in Rust
//   you will have to change type in Python before calling the Rust function.

// The name of the module must be the same as the rust package name
#[pymodule]
fn deseq2_rs_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // This is a pure function (no mutations of incoming data).
    // You can see this as the python array in the function arguments is readonly.
    // The object we return will need ot have the same lifetime as the Python.
    // Python will handle the objects deallocation.
    // We are having the Python as input with a lifetime parameter.
    // Basically, none of the data that comes from Python can survive
    // longer than Python itself. Therefore, if Python is dropped, so must our Rust Python-dependent variables.
    #[pyfn(m)]
    fn max_min<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<f64>) -> &'py PyArray1<f64> {
        // Here we have a numpy array of dynamic size. But we could restrict the
        // function to only take arrays of certain size
        // e.g. We could say PyReadonlyArray3 and only take 3 dim arrays.
        // These functions will also do type checking so a
        // numpy array of type np.float32 will not be accepted and will
        // yield an Exception in Python as expected
        let array = x.as_array();
        let result_array = fit_beta::max_min(&array);
        result_array.into_pyarray(py)
    }
    #[pyfn(m)]
    fn double_and_random_perturbation(
        _py: Python<'_>,
        x: &PyArrayDyn<f64>,
        perturbation_scaling: f64,
    ) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let mut array = unsafe { x.as_array_mut() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        fit_beta::double_and_random_perturbation(&mut array, perturbation_scaling);
    }

    /// Fit beta coefficients for negative binomial GLM
    ///
    /// This function estimates the coefficients (betas) for negative binomial generalized linear models.
    ///
    /// - `y` - ***n*** by ***m*** matrix of counts
    /// - `x` - ***m*** by ***k*** design matrix
    /// - `normalization_factors` - ***n*** by ***m*** matrix of normalization factors
    /// - `alpha_hat` - ***n*** length vector of the disperion estimates
    /// - `contrast` - a ***k*** length vector for a possible contrast
    /// - `beta_matrix` - ***n*** by ***k*** matrix of the initial estimates for the betas
    /// - `lambda` - ***k*** length vector of the ridge values
    /// - `weights` - ***n*** by ***m*** matrix of weights
    /// - `use_weights` - whether to use weights
    /// - `tolerance` - tolerance for convergence in estimates
    /// - `max_iterations` - maximum number of iterations
    /// - `use_qr_decomposition` - whether to use QR decomposition
    #[pyfn(m)]
    fn fit_beta(
        py: Python,
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
        max_iterations: usize,
        use_qr_decomposition: bool,
    ) -> &PyArray1<f64> {
        // Simple demonstration of creating an ndarray inside Rust and return
        array = fit_beta::fit_beta(
            &y.as_array(),
            &x.as_array(),
            &normalization_factors.as_array(),
            &alpha_hat.as_array(),
            &contrast.as_array(),
            &beta_matrix.as_array(),
            &lambda.as_array(),
            &weights.as_array(),
            use_weights,
            tolerance,
            max_iterations,
            use_qr_decomposition,
        );
        array.into_pyarray(py)
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

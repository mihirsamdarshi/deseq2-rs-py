import numpy as np

from . import fit_beta_impl

# def fit_disp_wrapper(
#     y,
#     x,
#     mu_hat,
#     log_alpha,
#     log_alpha_prior_mean,
#     log_alpha_prior_sigmasq,
#     min_log_alpha,
#     kappa_0,
#     tol,
#     maxit,
#     use_prior,
#     weights,
#     use_weights,
#     weight_threshold,
#     use_cr,
# ):
#     """
#     Fit dispersions for negative binomial GLM

#     This function estimates the dispersion parameter (alpha) for negative binomial
#     generalized linear models. The fitting is performed on the log scale.

#     :param y: n by m matrix of counts
#     :param x: m by k design matrix
#     :param mu_hat: n by m matrix, the expected mean values, given beta-hat
#     :param log_alpha: n length vector of initial guesses for log(alpha)
#     :param log_alpha_prior_mean: n length vector of the fitted values for log(alpha)
#     :param log_alpha_prior_sigmasq: a single numeric value for the variance of the prior
#     :param min_log_alpha: the minimum value of log alpha
#     :param kappa_0: a parameter used in calculating the initial proposal for the
#     backtracking search
#       initial proposal = log(alpha) + kappa_0 * deriv. of log lik. w.r.t. log(alpha)
#     :param tol: tolerance for convergence in estimates
#     :param maxit: maximum number of iterations
#     :param use_prior: boolean variable, whether to use a prior or just calculate the MLE
#     :param weights: n by m matrix of weights
#     :param use_weights: whether to use weights
#     :param weight_threshold: the threshold for subsetting the design matrix and GLM weights
#       for calculating the Cox-Reid correction
#     :param use_cr: whether to use the Cox-Reid correction

#     :return: a list with elements: log_alpha, iter, iter_accept, last_change, initial_lp,
#     intial_dlp, last_lp, last_dlp, last_d2lp
#     """

#     fitDisp(
#         y=y,
#         x=x,
#         mu_hat=mu_hat,
#         log_alpha=log_alpha,
#         log_alpha_prior_mean=log_alpha_prior_mean,
#         log_alpha_prior_sigmasq=log_alpha_prior_sigmasq,
#         min_log_alpha=min_log_alpha,
#         kappa_0=kappa_0,
#         tol=tol,
#         maxit=maxit,
#         usePrior=use_prior,
#         weights=weights,
#         useWeights=use_weights,
#         weightThreshold=weight_threshold,
#         useCR=use_cr,
#     )


# # return the estimate of dispersion (not log scale)
# def fit_disp_grid_wrapper(
#     y,
#     x,
#     mu,
#     log_alpha_prior_mean,
#     log_alpha_prior_sigma_sq,
#     use_prior,
#     weights,
#     use_weights,
#     weight_threshold,
#     use_cr,
# ):
#     """
#     Fit dispersions by evaluating over grid

#     This function estimates the dispersion parameter (alpha) for negative binomial
#     generalized linear models. The fitting is performed on the log scale.

#     :param y: n by m matrix of counts
#     :param x: m by k design matrix
#     :param mu: n by m matrix, the expected mean values, given beta-hat
#     :param log_alpha_prior_mean: n length vector of the fitted values for log(alpha)
#     :param log_alpha_prior_sigma_sq: a single numeric value for the variance of the prior
#     :param use_prior: boolean variable, whether to use a prior or just calculate the MLE
#     :param weights: n by m matrix of weights
#     :param use_weights: whether to use weights
#     :param weight_threshold: the threshold for sub-setting the design matrix and GLM weights, for
#       calculating the Cox-Reid correction
#     :param use_cr: whether to use the Cox-Reid correction
#     :return:
#     """
#     # test for any NAs in arguments

#     min_log_alpha = np.log(1e-8)
#     max_log_alpha = np.log(max(10, y.ncol))
#     disp_grid = range(min_log_alpha, max_log_alpha, 20)
#     logAlpha = fitDispGrid(
#         y=y,
#         x=x,
#         mu_hat=mu,
#         disp_grid=disp_grid,
#         log_alpha_prior_mean=log_alpha_prior_mean,
#         log_alpha_prior_sigmasq=log_alpha_prior_sigma_sq,
#         usePrior=use_prior,
#         weights=weights,
#         useWeights=use_weights,
#         weightThreshold=weight_threshold,
#         useCR=use_cr,
#     )
#     exp(logAlpha)


def fit_beta_wrapper(
    y,
    x,
    nf,
    alpha_hat,
    contrast,
    beta_matrix,
    p_lambda,
    weights,
    use_weights,
    tol,
    max_it,
    use_qr,
    min_mu,
):
    """
    Fit beta for negative binomial GLM

    This function fits the beta parameters for negative binomial generalized linear models

    :param y: n by m matrix of counts
    :param x: m by k design matrix
    :param nf: number of factors
    :param alpha_hat: n length vector of fitted values for log(alpha)
    :param contrast: a k by k contrast matrix
    :param beta_matrix: a k by k matrix of initial values for beta
    :param p_lambda: a k by k matrix of initial values for lambda
    :param weights: n by m matrix of weights
    :param use_weights: whether to use weights
    :param tol: tolerance for convergence in estimates
    :param max_it: maximum number of iterations
    :param use_qr: whether to use QR decomposition
    :param min_mu: a small positive value to add to the mean to avoid numerical problems

    :return: a list with elements: beta, lambda, iter, last_change, initial_lp, last_lp
    """
    if contrast is None:
        contrast = [1, np.repeat(0, x.shape[1] - 1)]

    fit_beta_impl(
        y=y,
        x=x,
        nf=nf,
        alpha_hat=alpha_hat,
        contrast=contrast,
        beta_mat=beta_matrix,
        p_lambda=p_lambda,
        weights=weights,
        use_weights=use_weights,
        tol=tol,
        maxit=max_it,
        use_qr=use_qr,
        min_mu=min_mu,
    )

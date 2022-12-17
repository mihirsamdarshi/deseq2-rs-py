from typing import Literal

import pandas as pd

from .dataset import Dataset


def inp():
    """
    DESeq <- function(object, test=c("Wald","LRT"),
                      fitType=c("parametric","local","mean", "glmGamPoi"),
                      sfType=c("ratio","poscounts","iterate"),
                      betaPrior,
                      full=design(object), reduced, quiet=FALSE,
                      minReplicatesForReplace=7, modelMatrixType,
                      useT=FALSE, minmu=if (fitType=="glmGamPoi") 1e-6 else 0.5,
                      parallel=FALSE, BPPARAM=bpparam()) {
      # check arguments
      stopifnot(is(object, "DESeqDataSet"))
      test <- match.arg(test, choices=c("Wald","LRT"))
      fitType <- match.arg(fitType, choices=c("parametric","local","mean","glmGamPoi"))
      dispersionEstimator <- if (fitType == "glmGamPoi") {
        "glmGamPoi"
      } else {
        "DESeq2"
      }
      # turn off outlier replacement for glmGamPoi
      if (fitType == "glmGamPoi") {
        minReplicatesForReplace <- Inf
        if (parallel) {
          warning("parallelization of DESeq() is not implemented for fitType='glmGamPoi'")
        }
      }
      sfType <- match.arg(sfType, choices=c("ratio","poscounts","iterate"))
      # more check arguments
      stopifnot(is.logical(quiet))
      stopifnot(is.numeric(minReplicatesForReplace))
      stopifnot(is.logical(parallel))
      modelAsFormula <- !is.matrix(full) & is(design(object), "formula")

      if (missing(betaPrior)) {
        betaPrior <- FALSE
      } else {
        stopifnot(is.logical(betaPrior))
      }
      # get rid of any NA in the mcols(mcols(object))
      object <- sanitizeRowRanges(object)

      if (test == "LRT") {
        if (missing(reduced)) {
          stop("likelihood ratio test requires a 'reduced' design, see ?DESeq")
        }
        if (betaPrior) {
          stop("test='LRT' does not support use of LFC shrinkage, use betaPrior=FALSE")
        }
        if (!missing(modelMatrixType) && modelMatrixType=="expanded") {
          stop("test='LRT' does not support use of expanded model matrix")
        }
        if (is.matrix(full) | is.matrix(reduced)) {
          if (!(is.matrix(full) & is.matrix(reduced))) {
            stop("if one of 'full' and 'reduced' is a matrix, the other must be also a matrix")
          }
        }
        if (modelAsFormula) {
          checkLRT(full, reduced)
        } else {
          checkFullRank(full)
          checkFullRank(reduced)
          if (ncol(full) <= ncol(reduced)) {
            stop("the number of columns of 'full' should be more than the number of columns of 'reduced'")
          }
        }
      }
      if (test == "Wald" & !missing(reduced)) {
        stop("'reduced' ignored when test='Wald'")
      }
      if (dispersionEstimator == "glmGamPoi" && test == "Wald") {
        warning("glmGamPoi dispersion estimator should be used in combination with a LRT and not a Wald test.",
                call. = FALSE)
      }

      if (modelAsFormula) {
        # run some tests common to DESeq, nbinomWaldTest, nbinomLRT
        designAndArgChecker(object, betaPrior)

        if (design(object) == formula(~1)) {
          warning("the design is ~ 1 (just an intercept). is this intended?")
        }

        if (full != design(object)) {
          stop("'full' specified as formula should equal design(object)")
        }
        modelMatrix <- NULL
      } else {
        # model not as formula, so DESeq() is using supplied model matrix
        if (!quiet) message("using supplied model matrix")
        if (betaPrior == TRUE) {
          stop("betaPrior=TRUE is not supported for user-provided model matrices")
        }
        checkFullRank(full)
        # this will be used for dispersion estimation and testing
        modelMatrix <- full
      }

      attr(object, "betaPrior") <- betaPrior
      stopifnot(length(parallel) == 1 & is.logical(parallel))

      if (!is.null(sizeFactors(object)) || !is.null(normalizationFactors(object))) {
        if (!quiet) {
          if (!is.null(normalizationFactors(object))) {
            message("using pre-existing normalization factors")
          } else {
            message("using pre-existing size factors")
          }
        }
      } else {
        if (!quiet) message("estimating size factors")
        object <- estimateSizeFactors(object, type=sfType, quiet=quiet)
      }

      if (!parallel) {
        if (!quiet) message("estimating dispersions")
        object <- estimateDispersions(object, fitType=fitType, quiet=quiet, modelMatrix=modelMatrix, minmu=minmu)
        if (!quiet) message("fitting model and testing")
        if (test == "Wald") {
          object <- nbinomWaldTest(object, betaPrior=betaPrior, quiet=quiet,
                                   modelMatrix=modelMatrix,
                                   modelMatrixType=modelMatrixType,
                                   useT=useT,
                                   minmu=minmu)
        } else if (test == "LRT") {
          object <- nbinomLRT(object, full=full,
                              reduced=reduced, quiet=quiet,
                              minmu=minmu,
                              type = dispersionEstimator)
        }
      } else if (parallel) {
        if (!missing(modelMatrixType)) {
          if (betaPrior) stopifnot(modelMatrixType=="expanded")
        }
        object <- DESeqParallel(object, test=test, fitType=fitType,
                                betaPrior=betaPrior, full=full, reduced=reduced,
                                quiet=quiet, modelMatrix=modelMatrix,
                                useT=useT, minmu=minmu,
                                BPPARAM=BPPARAM)
      }

      # if there are sufficient replicates, then pass through to refitting function
      sufficientReps <- any(nOrMoreInCell(attr(object,"modelMatrix"),minReplicatesForReplace))
      if (sufficientReps) {
        object <- refitWithoutOutliers(object, test=test, betaPrior=betaPrior,
                                       full=full, reduced=reduced, quiet=quiet,
                                       minReplicatesForReplace=minReplicatesForReplace,
                                       modelMatrix=modelMatrix,
                                       modelMatrixType=modelMatrixType)
      }

      # stash the package version (again, also in construction)
      metadata(object)[["version"]] <- packageVersion("DESeq2")

      object
    }"""
    pass


FitType = Literal["parametric", "local", "mean", "glmGamPoi"]
SFType = Literal["ratio", "poscounts", "iterate"]
TestType = Literal["Wald", "LRT"]


def analyze(
    data: Dataset,
    test: TestType,
    fit_type: FitType,
    sf_type: SFType,
    beta_prior,
    model_matrix_type,
    full,
    reduced,
    min_replicates_for_replace: int = 7,
    use_t: bool = False,
    min_mu=None,
    verbose: bool = True,
):
    """
    DESeq2 analysis.

    :param data: a DESeqDataSet object, see the class `DESeqDataSetFromMatrix`
    :param test: either "Wald" or "LRT", which will then use either Wald significance
        tests (defined by `nbinomWaldTest`), or the likelihood ratio test on the
        difference in deviance between a full and reduced model formula (defined
        by `nbinomLRT`)
    :param fit_type: either "parametric", "local", "mean", or "glmGamPoi" for the type of
        fitting of dispersions to the mean intensity. See `estimateDispersions`
        for description.
    :param sf_type: either "ratio", "poscounts", or "iterate" for the type of size factor
        estimation. See `estimateSizeFactors` for description.
    :param beta_prior: whether to put a zero-mean normal prior on the non-intercept
        coefficients.
        See `nbinomWaldTest` for description of the calculation of the beta prior.
        In versions `>=1.16`, the default is set to `False`, and shrunken LFCs are
        obtained afterwards using `lfcShrink`.
    :param full: for `test="LRT"`, the full model formula, which is restricted to the
        formula in `design(object)`.
        Alternatively, it can be a model matrix constructed by the user.
        advanced use: specifying a model matrix for full and `test="Wald"`
        is possible if `betaPrior=FALSE`
    :param reduced: for `test="LRT"`, a reduced formula to compare against,
        i.e., the full formula with the term(s) of interest removed.
        alternatively, it can be a model matrix constructed by the user
    :param verbose: whether to print messages at each step
    :param min_replicates_for_replace: the minimum number of replicates required
        in order to use `replaceOutliers` on a sample. If there are samples with so
        many replicates, the model will be refit after these replacing outliers, flagged
        by Cook's distance.
        Set to `Inf` in order to never replace outliers.
        It is set to `Inf` for `fitType="glmGamPoi"`.
    :param model_matrix_type: either "standard" or "expanded", which describe
        how the model matrix, X of the GLM formula is formed.
        "standard" is as created by `model.matrix` using the
        design formula. "expanded" includes an indicator variable for each
        level of factors in addition to an intercept. for more information
        see the Description of `nbinomWaldTest`.
        betaPrior must be set to TRUE in order for expanded model matrices
        to be fit.
    :param use_t: logical, passed to `nbinomWaldTest`, default is False,
        where Wald statistics are assumed to follow a standard Normal
    :param min_mu: lower bound on the estimated count for fitting gene-wise dispersion
        and for use with `nbinomWaldTest` and `nbinomLRT`.
        If `fitType="glmGamPoi"`, then 1e-6 will be used (as this fitType is optimized
        for single cell data, where a lower minmu is recommended), otherwise the
        default value as evaluated on bulk datasets is 0.5
    """
    # Check data
    if not isinstance(data, Dataset):
        raise TypeError("data must be a Dataset")

    # Check fitType
    if fit_type not in ["parametric", "local", "mean", "glmGamPoi"]:
        raise ValueError(
            "fitType must be one of 'parametric', 'local', 'mean', 'glmGamPoi'"
        )

    # Check sfType
    if sf_type not in ["ratio", "poscounts", "iterate"]:
        raise ValueError("sfType must be one of 'ratio', 'poscounts', 'iterate'")

    min_mu = 1e-6 if fit_type == "glmGamPoi" else 0.5

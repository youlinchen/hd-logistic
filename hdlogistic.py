"""
High-dimensional Logistic Regression
"""

# Author: You-Lin Chen <youlinchen@galton.uchicago.edu>

import numpy as np
import pyipopt
from sklearn.linear_model.base import BaseEstimator, LinearClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import optimize
from ipopt import minimize_ipopt
import time


# Define some useful function for chebyshev_greedy_algorithm_path and _cga_hdic_trim
def logistic(X, beta):
    lterm = X@beta
    lterm[lterm < 50.0] = np.exp(lterm[lterm < 50.0]) / (1+np.exp(lterm[lterm < 50.0]))
    lterm[lterm >= 50.0] = 1.0
    return lterm


def _logistic_loss(X, y):
    """Given input data X and labels y, create the log-likelihood function.

    Parameters
    ----------
    X : nd-array, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.
    y : nd-array, shape (n_samples,)
        Target vector relative to X.

    Returns
    -------
    loss : callable
        Logistic loss given X and y.
    """
    n = X.shape[0]

    def loss(beta):
            pterm = X @ beta
            pterm[pterm < 50.0] = np.log(1 + np.exp(pterm[pterm < 50.0]))
            return np.sum(pterm/n) - (y @ X @ beta)/n

    return loss


def _logistic_grad_hess(X, y):
    """Given input data X and labels y, create the gradient and the Hessian of the log-likelihood function.

    Parameters
    ----------
    X : nd-array, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.
    y : nd-array, shape (n_samples,)
        Target vector relative to X.

    Returns
    -------
    grad : callable
        the gradient of the logistic loss given X and y.
    hess : callable
        the Hessian of the logistic loss given X and y.
    """
    n = X.shape[0]

    def grad(beta): return (logistic(X, beta) - y) @ X / n

    def hess(beta): return X.T * (logistic(X, beta) * (1-logistic(X, beta))) @ X / n

    return grad, hess


def _hd_information_criterion(ic, loss, k, wn, n, p):
    """compute the information criterion with high dimensional penalty.
    Parameters
    ----------
    ic : str, {'HQIC', 'AIC', 'BIC'}
        The information criterion for model selection.
    loss : float
        the negative maximized value of the likelihood function of the model.
    k : int
        the number of free parameters to be estimated.
    wn : float
        the tuning parameter for the penalty term.
    n : int
        sample size.
    p : int
        the number of free parameters to be estimated.

    Returns
    -------
    val : float
        the value of the chosen information criterion given the loss and parameters.
    """
    val = 0.0
    if ic is 'HQIC':
        val = 2.0 * n * loss + 2.0 * k * wn * np.log(np.log(n)) * np.log(p)
    elif ic is 'AIC':
        val = 2.0 * n * loss + 2.0 * k * wn * np.log(p)
    elif ic is 'BIC':
        val = 2.0 * n * loss + k * wn * np.log(n) * np.log(p)
    return val


def _information_criterion(ic, loss, k, n):
    """compute the information criterion.
    Parameters
    ----------
    ic : str, {'HQIC', 'AIC', 'BIC'}
        The information criterion for model selection.
    loss : float
        the negative maximized value of the likelihood function of the model.
    k : int
        the number of free parameters to be estimated.
    n : int
        sample size.

    Returns
    -------
    val : float
        the value of the chosen information criterion given the loss and parameters.
    """
    val = 0
    if ic is 'HQIC':
        val = 2.0 * n * loss + k * np.log(np.log(n))
    elif ic is 'AIC':
        val = 2.0 * n * loss + 2.0 * k
    elif ic is 'BIC':
        val = 2.0 * n * loss + k * np.log(n)
    return val


def _minimize(loss, x0, method, jac, hess, tol, options):
    if method is 'pyipopt':
        pyipopt.set_loglevel(0)
        res = pyipopt.fmin_unconstrained(loss, x0, fprime=jac, fhess=hess, tol= tol)
        return res[0], res[4]
    elif method is 'ipopt':
        res = minimize_ipopt(loss, x0, jac=jac)
        return res.x, res.fun
    else:
        res = optimize.minimize(loss, x0, method=method, jac=jac, hess=hess, tol=tol, options=options)
        return res.x, res.fun


def chebyshev_greedy_algorithm_path(X, y, ic='HQIC', wn=1.0, fit_intercept=True, kn=3.0, method='dogleg',
                                    tol=1e-8, options=None):
    """compute the path of Chebyshev greedy algorithm.
    Parameters
    ----------
    X : nd-array, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.
    y : nd-array, shape (n_samples,)
        Target vector relative to X.
    ic : str, {'HQIC', 'AIC', 'BIC'}, default: 'HQIC'
        The information criterion for model selection.
    wn : float, default: 1.0
        the tuning parameter for the penalty term.
    fit_intercept : bool, default: True
        Whether to fit an intercept for the model. In this case the length of the returned array is n_features + 1.
        Also the parameter index will be added by one.
    kn : float, default: 3.0
        the tuning parameter for the number of iteration.
    method : str, {'dogleg', 'trust-ncg', 'Newton-CG'}, default: 'dogleg'
        Type of solver.
    tol : float, default: 1e-8
        Tolerance for termination.
    options : dict, optional
        A dictionary of solver options. All methods accept the following generic options:
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see the document for scipy.optimize.minimize.

    Returns
    -------
    beta_cga : nd-array, shape (n_features + fit_intercept, iter_cga)
        the estimated coefficient in each iteration where iter_cga is the total number of iterations.
    path_cga : nd-array, shape (iter_cga,)
        The sequentially chosen regressors in each iteration. Note that if fit_intercept is True,
        the first term must be the intercept.
    hdic_cga : nd-array, shape (iter_cga,)
        The computed value of chosen information criterion in each iteration.
    iter_cga : int
        The total number of iterations of CGA.
    loss_path_cga : nd-array, shape (iter_cga,)
        The computed value of loss(the negative maximized value of the likelihood function) in each iteration.
    """
    # initialize the variables
    n = X.shape[0]
    if fit_intercept:
        X = np.c_[np.ones([n, 1]), X]
    p = X.shape[1]
    iter_cga = int(np.ceil(kn * np.sqrt(n / np.log(p))) + int(fit_intercept))
    beta_cga = np.zeros([p, iter_cga])
    path_cga = np.zeros(iter_cga, dtype=np.int)
    hdic_cga = np.zeros(iter_cga)
    loss_path_cga = np.zeros(iter_cga)
    loss_grad = _logistic_grad_hess(X, y)[0]

    # the first step of CGA or estimation of intercept if fit_intercept is True.
    # choose the regressor that has maximal derivative.
    path_cga[0] = 0 if fit_intercept else np.argmax(np.abs(loss_grad(beta_cga[:, 0])))
    # create the loss function and its gradient and Hessian with respect y and chosen regressors.
    loss_cga = _logistic_loss(X[:, path_cga[0]].reshape(-1,1), y)
    (loss_grad_cga, loss_hess_cga) = _logistic_grad_hess(X[:, path_cga[0]].reshape(-1,1), y)
    # set the initial value
    x0 = beta_cga[0, 0:1]
    # use scipy.optimize.minimize to minimize the loss function given its gradient and Hessian.
    (res_x, res_fun) = _minimize(loss_cga, x0, method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
    # extract the result of the optimization.
    beta_cga[path_cga[0], 0] = res_x[0]
    loss_path_cga[0] = res_fun
    # calculate the value of information criterion
    hdic_cga[0] = _hd_information_criterion(ic, res_fun, 1, wn, n, p)

    # The (iter_cga-1) steps of CGA
    for k in range(1, iter_cga):
        # choose the regressor that has maximal derivative but is not in the path_cga.
        loss_grad_abs = np.abs(loss_grad(beta_cga[:, k - 1]))
        loss_grad_abs[path_cga[0:k]] = -1
        path_cga[k] = np.argmax(loss_grad_abs)
        # create the loss function and its gradient and Hessian with respect y and chosen regressors.
        loss_cga = _logistic_loss(X[:, path_cga[0:k+1]], y)
        (loss_grad_cga, loss_hess_cga) = _logistic_grad_hess(X[:, path_cga[0:k+1]], y)
        # solve the optimization problem
        x0 = beta_cga[path_cga[0:k+1], k]
        (res_x, res_fun) = _minimize(loss_cga, x0, method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
        # get the information
        beta_cga[path_cga[0:k+1], k] = res_x
        loss_path_cga[k] = res_fun
        hdic_cga[k] = _hd_information_criterion(ic, res_fun, k+1, wn, n, p)

    return beta_cga, path_cga, hdic_cga, iter_cga, loss_path_cga


def _cga_hdic_trim(X, y, ic, wn, fit_intercept, kn, method, tol, options, trimming):
    """use the three stage method(CGA+HDIC+Trim) to compute the model.
    Parameters
    ----------
    X : nd-array, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.
    y : nd-array, shape (n_samples,)
        Target vector relative to X.
    ic : str, {'HQIC', 'AIC', 'BIC'}, default: 'HQIC'
        The information criterion for model selection.
    wn : float, default: 1.0
        the tuning parameter for the penalty term.
    fit_intercept : bool, default: True
        Whether to fit an intercept for the model. In this case the length of the returned array is n_features + 1.
        Also the parameter index will be added by one.
    kn : float, default: 3.0
        the tuning parameter for the number of iteration.
    method : str, {'dogleg', 'trust-ncg', 'Newton-CG'}, default: 'dogleg'
        Type of solver.
    tol : float, default: 1e-8
        Tolerance for termination.
    options : dict, optional
        A dictionary of solver options. All methods accept the following generic options:
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see the document for scipy.optimize.minimize.
    trimming : bool
        Whether to trim the model.

    Returns
    -------
    ### IMPORTANT: the return result will separate intercept and coefficients, and will correct the index of
        returns, regressors and final model if fit_itercept is True ###
    intercept : nd-array, shape (1,),
        0 if fit_intercept is True.
    coef : nd-array, shape (iter_cga,)
        The coefficients of all regressors.
    model_trim : nd-array, shape (???,)
        The final model.
    loss : float
        The negative maximized value of the likelihood function of final model.
    beta_cga : nd-array, shape (n_features + fit_intercept, iter_cga)
        the estimated coefficient in each iteration where iter_cga is the total number of iterations.
    path_cga : nd-array, shape (iter_cga,)
        The sequentially chosen regressors in each iteration.
    hdic_cga : nd-array, shape (iter_cga,)
        The computed value of chosen information criterion in each iteration.
    iter_cga : int
        The total number of iterations of CGA.
    """
    # Check the shape of X and y, and initialize the variables
    X = check_array(X)
    (n, p) = X.shape
    beta_hat = np.zeros(p+1) if fit_intercept else np.zeros(p)

    # CGA
    (beta_cga, path_cga, hdic_cga, iter_cga, loss_path_cga) = chebyshev_greedy_algorithm_path(
        X, y, ic, wn, fit_intercept, kn, method, tol, options)
    if fit_intercept:
        X = np.c_[np.ones([n, 1]), X]

    # HDIC
    # truncate the path at the variable that has minimal high-dimensional information criterion
    k_hdic = np.argmin(hdic_cga)
    model = path_cga[0:k_hdic+1]

    # Trim
    model_trim_list = list()
    model_size = model.shape[0]
    # exclude the variable if the value high-dimensional information criterion of exclusion model
    # is lower than original model.
    if model_size > 1 and trimming:
        for k in range(int(fit_intercept), k_hdic+1):
            # exclude the variable.
            model_trim_k = np.delete(model, k)
            # create the loss function and its gradient and Hessian with respect the exclusion model.
            loss_cga = _logistic_loss(X[:, model_trim_k], y)
            (loss_grad_cga, loss_hess_cga) = _logistic_grad_hess(X[:, model_trim_k], y)
            # set the initial value.
            x0 = beta_cga[model_trim_k, k_hdic]
            # use scipy.optimize.minimize to minimize the loss function given its gradient and Hessian.
            (res_x, res_fun) = _minimize(loss_cga, x0, method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
            # calculate the value high-dimensional information criterion of exclusion model.
            hdic_trim = _hd_information_criterion(ic, res_fun, k_hdic, wn, n, p)
            # compare with the original model.
            if hdic_trim < hdic_cga[k_hdic]:
                model_size -= 1
                model_trim_list += [k]
    # get the trimming model
    model_trim = np.delete(model, model_trim_list)

    # if there is variable that is excluded, reestimate the trimming model.
    if model_size < model.shape[0]:
        loss_cga = _logistic_loss(X[:, model_trim], y)
        (loss_grad_cga, loss_hess_cga) = _logistic_grad_hess(X[:, model_trim], y)
        x0 = beta_cga[model_trim, k_hdic]
        (res_x, res_fun) = _minimize(loss_cga, x0, method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
        beta_hat[model_trim] = res_x
        loss = res_fun
    else:
        beta_hat[model] = beta_cga[model, k_hdic]
        loss = loss_path_cga[k_hdic]

    # correct the index if fit_intercept is True.
    if fit_intercept:
        intercept = beta_hat[0]
        intercept_cga = beta_cga[0, :]
        coef = beta_hat[1:].reshape(1, -1)
        coef_cga = beta_cga[1:, :]
        hdic_cga = hdic_cga[1:]
        path_cga = path_cga[1:] - 1
        model_trim = model_trim[1:] - 1
        iter_cga -= 1
    else:
        intercept = np.zeros(1)
        intercept_cga = np.zeros(iter_cga)
        coef_cga = beta_cga
        coef = beta_hat.reshape(1,-1)

    return intercept, coef, model_trim, loss, path_cga, intercept_cga, coef_cga, hdic_cga, iter_cga


class HighDimensionalLogisticRegression(BaseEstimator, LinearClassifierMixin):
    """The High-dimensional Logistic Regression.
    Parameters
    ----------
    ic : str, {'HQIC', 'AIC', 'BIC'}, default: 'HQIC'
        The information criterion for model selection.
    wn : float, default: 1.0
        the tuning parameter for the penalty term.
    fit_intercept : bool, default: True
        Whether to fit an intercept for the model. In this case the length of the returned array is n_features + 1.
        Also the parameter index will be added by one.
    kn : float, default: 3.0
        the tuning parameter for the number of iteration.
    method : str, {'dogleg', 'trust-ncg', 'Newton-CG'}, default: 'dogleg'
        Type of solver.
    tol : float, default: 1e-8
        Tolerance for termination.
    options : dict, optional
        A dictionary of solver options. All methods accept the following generic options:
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see the document for scipy.optimize.minimize.

    Attributes
    -------
    ### IMPORTANT: the return result will separate intercept and coefficients, and will correct the index of
        returns, regressors and final model if fit_itercept is True ###
    intercept_ : nd-array, shape (1,),
        0 if fit_intercept is True.
    coef_ : nd-array, shape (iter_cga,)
        The coefficients of all regressors.
    model_ : nd-array, shape (???,)
        The final model.
    loss_ : float
        The negative maximized value of the likelihood function of final model.
    intercept_cga_ : nd-array, shape (n_features + fit_intercept, iter_cga)
        the estimated coefficient in each iteration where iter_cga is the total number of iterations.
    coef_cga_ : nd-array, shape (n_features + fit_intercept, iter_cga)
        the estimated coefficient in each iteration where iter_cga is the total number of iterations.
    path_cga_ : nd-array, shape (iter_cga,)
        The sequentially chosen regressors in each iteration.
    hdic_cga_ : nd-array, shape (iter_cga,)
        The computed value of chosen information criterion in each iteration.
    iter_cga_ : int
        The total number of iterations of CGA.
    """

    def __init__(self, ic='HQIC', wn=1.0, fit_intercept=True, kn=1.0, method='dogleg', tol=1e-8,
                 options=None, trimming=True):
        self.ic = ic
        self.wn = wn
        self.fit_intercept = fit_intercept
        self.kn = kn
        self.method = method
        self.tol = tol
        self.trimming = trimming
        self.options = options

    def fit(self, X, y):
        """Fit the model according to the given training data.
            Parameters
            ----------
            X : nd-array, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y : nd-array, shape (n_samples,)
                Target vector relative to X.

            Returns
            -------
            self : object
                Returns self.
        """
        (X, y) = check_X_y(X, y)
        self.classes_ = np.unique(y)
        (self.intercept_, self.coef_, self.model_, self.loss_, self.path_cga_, self.intercept_cga_, self.coef_cga_,
         self.hdic_cga_, self.iter_cga_) = _cga_hdic_trim(X, y, self.ic, self.wn, self.fit_intercept, self.kn,
                                                          self.method, self.tol, self.options, self.trimming)

        return self

    def predict_proba(self, X):
        """Probability estimates.
            ----------
            X : nd-array, shape (n_samples, n_features)

            Returns
            -------
            P : nd-array, shape (n_samples, n_features)
                Returns the probability of the sample in the model.
        """
        check_is_fitted(self, ['model_'])
        X = check_array(X)
        n = X.shape[0]
        if self.fit_intercept:
            X = np.c_[np.ones([n, 1]), X]
            beta_hat = np.c_[self.intercept_, self.coef_]
            P = logistic(X, beta_hat.T)
        else:
            P = logistic(X, self.coef_.T)
        return P

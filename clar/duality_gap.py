import numpy as np

from numba import njit
from numpy.linalg import norm
from numpy.linalg import slogdet

from clar.utils import l_2_inf
from clar.utils import l_2_1


def get_p_obj_mrce(X, Y, Y2, Sigma, Sigma_inv, alpha, alpha_Sigma_inv, B, sigma_min):
    n, q = Y.shape
    XB = X @ B
    XBYT = XB @ Y.T
    emp_cov = (Y2 + XB @ XB.T - XBYT - XBYT.T)
    emp_cov /= q
    p_obj = Sigma_inv @ emp_cov
    p_obj = np.trace(p_obj)
    p_obj /= n
    p_obj += alpha_Sigma_inv * np.abs(Sigma_inv).sum()
    p_obj += alpha * l_2_1(B)
    logdet_Sigma = slogdet(Sigma)[1]
    # logdet_Sigma, _ = update_sigma_glasso(Sigma, sigma_min)
    p_obj += logdet_Sigma # to improve with a log det
    return p_obj


@njit
def get_p_obj_mtl(R, B, alpha):
    n_sensors, n_times = R.shape
    p_obj = (R ** 2).sum() / (2 * n_times * n_sensors) \
        + alpha * l_2_1(B)
    return p_obj


@njit
def get_d_obj_mtl(Y, Theta, alpha):
    n_sensors, n_times = Y.shape
    d_obj = alpha * (Theta * Y).sum() - \
        alpha ** 2 * n_times * n_sensors * (Theta ** 2).sum() / 2
    return d_obj


@njit
def get_feasible_theta_mtl(R, X, alpha):
    n_sensors, n_times = R.shape
    scaling_factor = l_2_inf(X.T @ R)
    scaling_factor = max(scaling_factor, alpha * n_sensors * n_times)
    return R / scaling_factor


@njit
def get_p_obj_me(R_all_epochs, B, S_inv_R, S_trace, alpha):
    n_epochs, n_channels, n_times = R_all_epochs.shape
    p_obj = (R_all_epochs * S_inv_R).sum()
    p_obj /= (2 * n_channels * n_times * n_epochs)
    p_obj += S_trace / (2. * n_channels)
    p_obj += alpha * l_2_1(B)
    return p_obj


@njit
def get_d_obj_me(all_epochs, Theta, sigma_min, alpha):
    n_epochs, n_channels, n_times = all_epochs.shape
    d_obj = alpha * (all_epochs * Theta).sum() / n_epochs
    d_obj += sigma_min / 2 * (
        1 - n_channels * n_times * alpha ** 2. * (Theta ** 2).sum()
        / n_epochs
    )
    return d_obj


@njit
def get_d_obj(Y, Theta, sigma_min, alpha):
    n_channels, n_times = Y.shape
    d_obj = alpha * (Y * Theta).sum()
    d_obj += sigma_min / 2. * (
        1. - n_channels * n_times * alpha ** 2. * (Theta ** 2).sum())
    return d_obj


@njit
def get_p_obj(R, B, S_trace, alpha, S_inv_R):
    n_channels, n_times = R.shape
    p_obj = (R * S_inv_R).sum() / (2. * n_channels * n_times)
    p_obj += S_trace / (2. * n_channels)
    p_obj += alpha * l_2_1(B)
    return p_obj


@njit
def get_feasible_theta(X, alpha, S_inv_R):
    n_channels, n_times = S_inv_R.shape
    scaling_factor = max(
        l_2_inf(X.T @ S_inv_R),
        alpha * n_channels * np.sqrt(n_times) * norm(S_inv_R, 2))
    scaling_factor = max(scaling_factor, n_channels * n_times * alpha)
    return S_inv_R / scaling_factor


@njit
def get_feasible_theta_me(X, alpha, S_inv_R):
    n_epochs, n_channels, n_times = S_inv_R.shape
    S_inv_R_mean = np.zeros((n_channels, n_times), dtype=np.float64)
    for l in range(n_epochs):
        S_inv_R_mean += S_inv_R[l, :, :]
    S_inv_R_mean /= n_epochs

    matrix_2 = S_inv_R[0, :, :]
    for l in range(n_epochs - 1):
        matrix_2 = np.concatenate(
            (matrix_2, S_inv_R[l + 1, :, :]),
            axis=1)
    scaling_factor = max(
        l_2_inf(X.T @ S_inv_R_mean),
        alpha * n_channels * np.sqrt(n_times / n_epochs) *
        norm(matrix_2, 2))
    scaling_factor = max(scaling_factor, n_channels * n_times * alpha)
    return S_inv_R / scaling_factor


@njit
def get_duality_gap_mtl(X, Y, B, alpha):
    n_sensors, n_times = Y.shape
    R = Y - X @ B
    p_obj = get_p_obj_mtl(R, B, alpha)
    Theta = get_feasible_theta_mtl(R, X, alpha)
    d_obj = get_d_obj_mtl(Y, Theta, alpha)
    return p_obj, d_obj


@njit
def get_duality_gap(
        R, X, Y, B, S_trace, S_inv_R, sigma_min, alpha):
    p_obj = get_p_obj(R, B, S_trace, alpha, S_inv_R)
    Theta = get_feasible_theta(X, alpha, S_inv_R)
    d_obj = get_d_obj(Y, Theta, sigma_min, alpha)
    return p_obj, d_obj


@njit
def get_duality_gap_me(
        X, all_epochs, B, S_trace, S_inv,
        sigma_min, alpha):
    R_all_epochs = all_epochs - X @ B
    n_epochs, _, _ = R_all_epochs.shape
    S_inv_R = np.empty(R_all_epochs.shape)
    for l in range(n_epochs):
        S_inv_R[l, :, :] = S_inv @ R_all_epochs[l, :, :]
    p_obj = get_p_obj_me(R_all_epochs, B, S_inv_R, S_trace, alpha)
    Theta = get_feasible_theta_me(X, alpha, S_inv_R)
    d_obj = get_d_obj_me(all_epochs, Theta, sigma_min, alpha)
    return p_obj, d_obj

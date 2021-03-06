import numpy as np
from clar.solvers import solver, update_sigma_glasso
from clar.utils import (
    get_alpha_max, get_sigma_min)
from clar.data.artificial import get_data_me


def test_update_sigma_glasso():
    rho_noise = 0.8
    SNR = 1
    n_epochs, n_channels, n_sources, n_times = 5, 20, 10, 30

    all_epochs = get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_multivariate",
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=3, rho_noise=rho_noise,
        SNR=SNR)[1]

    emp_cov = np.zeros((n_channels, n_channels))
    for i in range(n_epochs):
        emp_cov += all_epochs[i, :, :] @ all_epochs[i, :, :].T
    emp_cov /= (n_times * n_epochs)

    alpha_prec = 0.001

    update_sigma_glasso(
        emp_cov, alpha_prec, enet_tol=1e-4, max_iter=100)


def test_mrce():
    rho_noise = 0.6
    SNR = 1
    n_epochs, n_channels, n_sources, n_times = 5, 20, 10, 30
    pb_name = "mrce"
    tol = 1e-4
    X, all_epochs = get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_multivariate",
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=3, rho_noise=rho_noise,
        SNR=SNR)[:2]

    alpha_Sigma_inv = 0.01

    Y = np.mean(all_epochs, axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(
        X, all_epochs, sigma_min, pb_name=pb_name,
        alpha_Sigma_inv=alpha_Sigma_inv)

    alpha = alpha_max * 0.9

    solver(
        X, all_epochs, alpha, sigma_min, B0=None,
        tol=tol, pb_name=pb_name, n_iter=1000, alpha_Sigma_inv=alpha_Sigma_inv)


if __name__ == '__main__':
    test_mrce()

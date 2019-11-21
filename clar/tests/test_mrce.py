import numpy as np
from clar.solvers import solver
from clar.utils import (
    get_alpha_max, get_sigma_min)
from clar.data.artificial import get_data_me


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

    Es = solver(
        X, all_epochs, alpha, sigma_min, B0=None,
        tol=tol, pb_name=pb_name, n_iter=1000,
        alpha_Sigma_inv=alpha_Sigma_inv)[-2]

    assert (Es[-1] - Es[-2]) <= 1e-10


if __name__ == '__main__':
    test_mrce()

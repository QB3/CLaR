import numpy as np
from clar.solvers import solver
from clar.utils import (
    get_alpha_max, get_sigma_min)
from clar.data.artificial import get_data_me


def test_mtl():
    rho_noise = 0.3
    SNR = 1
    n_epochs, n_channels, n_sources, n_times = 5, 20, 10, 30
    pb_name = "MTL"
    tol = 1e-7

    X, all_epochs, _, _ = get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_iid",
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=3, rho_noise=rho_noise,
        SNR=SNR)

    Y = np.mean(all_epochs, axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(X, Y, sigma_min, pb_name=pb_name)

    alpha_div = 5
    alpha = alpha_max / alpha_div

    B_mtl, _, E, gaps = solver(
        X, Y, alpha, sigma_min, B0=None,
        tol=tol, pb_name=pb_name, n_iter=10000)
    gap = gaps[-1]
    np.testing.assert_array_less(gap, tol)

    _, _, E, gaps = solver(
        X, Y, alpha, sigma_min, B0=B_mtl,
        tol=tol, pb_name=pb_name, n_iter=10000)
    np.testing.assert_equal(len(E), 2)
    gap = gaps[-1]
    np.testing.assert_array_less(gap, tol * E[0])


def test_mtl_me():
    rho_noise = 0.3
    SNR = 1
    n_epochs, n_channels, n_sources, n_times = 5, 20, 10, 30
    pb_name = "MTLME"
    tol = 1e-7

    X, all_epochs, _, _ = get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_iid",
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=3, rho_noise=rho_noise,
        SNR=SNR)

    Y = np.mean(all_epochs, axis=0)
    sigma_min = get_sigma_min(Y)

    alpha_max = get_alpha_max(X, all_epochs, sigma_min, pb_name=pb_name)

    alpha_div = 1.1
    alpha = alpha_max / alpha_div

    _, _, _, gaps = solver(
        X, all_epochs, alpha, sigma_min, B0=None,
        tol=tol, pb_name=pb_name, n_iter=10000)
    gap = gaps[-1]
    np.testing.assert_array_less(gap, tol)


if __name__ == '__main__':
    test_mtl()
    test_mtl_me()

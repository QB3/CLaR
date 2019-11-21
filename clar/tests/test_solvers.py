import numpy as np
import pytest

from clar.solvers import solver, get_path
from clar.utils import get_alpha_max, get_sigma_min
from clar.data.artificial import get_data_me


def test_mtl():
    rho_noise = 0.3
    SNR = 1
    n_epochs, n_channels, n_sources, n_times = 5, 20, 10, 30
    pb_name = "MTL"
    tol = 1e-10

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

    gap = solver(
        X, all_epochs, alpha, sigma_min, B0=None,
        tol=tol, pb_name=pb_name, n_iter=10000)[-1]
    np.testing.assert_array_less(gap, tol)


@pytest.mark.parametrize("n_sources", [10, 15, 20])
def tests_sgcl(n_sources):
    rho_noise = 0.3
    SNR = 0.5
    n_channels = 20
    n_times = 30
    n_active = 3
    n_iter = 10**6
    tol = 10**-4

    X, all_epochs, _, _ = get_data_me(
        n_channels=n_channels, n_times=n_times, n_sources=n_sources,
        n_active=n_active, rho_noise=rho_noise, SNR=SNR,
        n_epochs=50)

    Y = all_epochs.mean(axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(X, Y, sigma_min, "SGCL")
    alpha = alpha_max * 0.9
    print("alpha = %.2e" % alpha)
    print("sigma_min = %.2e" % sigma_min)

    all_epochs = np.zeros((1, *Y.shape))
    all_epochs[0] = Y

    _, _, E, (gaps, gaps_accel) = solver(
        X, Y, alpha, sigma_min, B0=None, n_iter=n_iter,
        pb_name="SGCL", use_accel=True, tol=tol,
        verbose=True)

    log_gap = np.log10(gaps[-1])
    log_gap_accel = np.log10(gaps_accel[-1])
    np.testing.assert_array_less(
        np.minimum(log_gap, log_gap_accel), np.log10(tol) * E[0])


@pytest.mark.parametrize("n_sources", [10, 15, 20])
def test_clar(n_sources):
    rho_noise = 0.3
    SNR = 0.5
    n_channels = 20
    n_times = 30
    n_epochs = 50
    n_active = 3
    n_iter = 10**4
    p_alpha = 0.2
    tol = 1e-7
    X, all_epochs, _, _ = get_data_me(
        dictionary_type="Toeplitz",
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=n_active, rho_noise=rho_noise,
        SNR=SNR
    )
    Y = np.mean(all_epochs, axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(X, all_epochs, sigma_min, pb_name="CLAR")
    alpha = alpha_max * p_alpha

    gaps_me = solver(
        X, all_epochs, alpha, sigma_min, B0=None,
        n_iter=n_iter, tol=tol, pb_name="CLAR")[-1]
    gap_me = gaps_me[-1]
    np.testing.assert_array_less(gap_me, tol)

    p_alphas = np.geomspace(1, 0.1, 10)
    dict_masks = get_path(
        X, all_epochs, p_alphas, alpha_max, sigma_min,
        n_iter=10**4, tol=10**-4)[0]

    old_size_supp = 0
    for supp in dict_masks.values():
        size_supp = supp.sum()
        np.testing.assert_array_less(old_size_supp - size_supp, 1)
        old_size_supp = size_supp

    # import ipdb; ipdb.set_trace()


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

import numpy as np
import pytest

from clar.solvers import solver
from clar.utils import get_alpha_max, get_sigma_min
from clar.data.artificial import get_data_me


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

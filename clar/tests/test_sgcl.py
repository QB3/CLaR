import numpy as np

from clar.solvers import solver
from clar.utils import get_alpha_max, get_sigma_min
from clar.data.artificial import get_data_me


def test_sgcl1():
    tests_sgcl(
        n_channels=30, n_times=30,
        n_sources=10, n_active=1, n_iter=10**7, tol=10**-4)


def tests_sgcl(
        noise_type="Gaussian_iid", rho_noise=0.3,
        SNR=0.5, n_channels=20, n_times=30, n_sources=10, n_active=3,
        gap_freq=100, active_set_freq=1, S_freq=10, n_iter=10**6,
        p_alpha_max=0.9, tol=10**-4):

    X, all_epochs, _, _ = get_data_me(
        dictionary_type="Gaussian", noise_type=noise_type,
        n_channels=n_channels, n_times=n_times, n_sources=n_sources,
        n_active=n_active, rho_noise=rho_noise, SNR=SNR,
        n_epochs=50)

    Y = all_epochs.mean(axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(X, Y, sigma_min, "SGCL")
    alpha = alpha_max * p_alpha_max
    print("alpha = %.2e" % alpha)
    print("sigma_min = %.2e" % sigma_min)

    all_epochs = np.zeros((1, *Y.shape))
    all_epochs[0] = Y

    _, _, E, (gaps, gaps_accel) = solver(
        X, Y, alpha, sigma_min, B0=None, n_iter=n_iter,
        gap_freq=gap_freq, active_set_freq=active_set_freq,
        S_freq=S_freq, pb_name="SGCL", use_accel=True, tol=tol,
        verbose=True)

    log_gap = np.log10(gaps[-1])
    log_gap_accel = np.log10(gaps_accel[-1])
    assert log_gap_accel < np.log10(tol) * E[0] or \
        log_gap < np.log10(tol) * E[0]


if __name__ == '__main__':
    test_sgcl1()

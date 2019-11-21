import numpy as np

from clar.solvers import solver
from clar.data.artificial import get_data_me
from clar.utils import get_sigma_min, get_alpha_max


def test_clar1():
    test_clar(
        n_channels=20, n_times=30, n_epochs=50,
        n_sources=50, n_active=3, n_iter=10**4, tol=1e-7)


def test_clar(
        noise_type="Gaussian_iid", rho_noise=0.3,
        SNR=0.5, n_channels=20, n_times=30, n_sources=10, n_epochs=50,
        n_active=3, gap_freq=50, active_set_freq=1, S_freq=10,
        n_iter=10**4, alpha_under_alpha_max=0.2, tol=1e-7):
    X, all_epochs, _, _ = get_data_me(
        dictionary_type="Gaussian", noise_type=noise_type,
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=n_active, rho_noise=rho_noise,
        SNR=SNR
    )
    Y = np.mean(all_epochs, axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(X, all_epochs, sigma_min, pb_name="CLAR")
    alpha = alpha_max * \
        alpha_under_alpha_max

    print("alpha = %.2e" % alpha)
    print("alpha = %.2e" % alpha)
    print("sigma_min = %.2e" % sigma_min)
    gaps_me = solver(
        X, all_epochs, alpha, sigma_min, B0=None,
        n_iter=n_iter,
        gap_freq=gap_freq, active_set_freq=active_set_freq,
        S_freq=S_freq, tol=tol, pb_name="CLAR")[-1]
    gap_me = gaps_me[-1]
    assert gap_me < tol


if __name__ == '__main__':
    test_clar1()

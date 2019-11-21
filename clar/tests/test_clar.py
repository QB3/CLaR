import numpy as np
import pytest

from clar.solvers import solver
from clar.data.artificial import get_data_me
from clar.utils import get_sigma_min, get_alpha_max


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
    assert gap_me < tol

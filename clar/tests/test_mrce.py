import numpy as np
from clar.solvers import solver, update_Sigma_glasso
from clar.utils import (
    get_alpha_max, get_sigma_min)
from clar.data.artificial import get_data_me
# import matplotlib.pyplot as plt


def test_update_Sigma_glasso():
    rho_noise = 0.8
    SNR = 1
    n_epochs, n_channels, n_sources, n_times = 5, 20, 10, 30
    pb_name = "MTL"
    tol = 1e-7

    X, all_epochs, B_star, (_, S_star) = get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_multivariate",
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=3, rho_noise=rho_noise,
        SNR=SNR)

    Y = np.mean(all_epochs, axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(X, Y, sigma_min, pb_name=pb_name)

    emp_cov = np.zeros((n_channels, n_channels))
    for i in range(n_epochs):
        emp_cov += all_epochs[i, :, :] @ all_epochs[i, :, :].T
    emp_cov /= (n_times * n_epochs)

    alpha_prec = 0.001

    covariance, precision = update_Sigma_glasso(
        emp_cov, alpha_prec, cov_init=None, mode='cd', tol=1e-4,
        enet_tol=1e-4, max_iter=100, verbose=False,
        return_costs=False, eps=np.finfo(np.float64).eps,
        return_n_iter=False)
    # plt.figure()
    # plt.imshow(covariance)
    # plt.show(block=False)
    # plt.figure()
    # plt.imshow(S_star @ S_star.T)
    # plt.show(block=True)
    # plt.figure()
    # plt.imshow(precision)
    # plt.show(block=True)
    # import ipdb; ipdb.set_trace()


def test_MRCE():
    rho_noise = 0.6
    SNR = 1
    n_epochs, n_channels, n_sources, n_times = 5, 20, 10, 30
    pb_name = "MRCE"
    tol = 1e-4

    X, all_epochs, B_star, (_, S_star) = get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_multivariate",
        n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
        n_sources=n_sources, n_active=3, rho_noise=rho_noise,
        SNR=SNR)

    alpha_Sigma_inv = 0.01

    Y = np.mean(all_epochs, axis=0)
    sigma_min = get_sigma_min(Y)
    alpha_max = get_alpha_max(X, all_epochs, sigma_min, pb_name=pb_name, alpha_Sigma_inv=alpha_Sigma_inv)

    alpha = alpha_max * 0.9


    B_MRCE, (Sigma, Sigma_inv), E, gaps = solver(
        X, all_epochs, alpha, alpha_max, sigma_min, B0=None,
        tol=tol, pb_name=pb_name, n_iter=1000, alpha_Sigma_inv=alpha_Sigma_inv)
    gap = gaps[-1]
    # np.testing.assert_array_less(gap, tol)
    # np.testing.assert_array_less(gap, tol * E[0])
    # plt.figure()
    # plt.imshow(Sigma)
    # plt.show(block=False)
    # plt.figure()
    # plt.imshow(S_star @ S_star.T)
    # plt.show(block=True)
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    # test_update_Sigma_glasso()
    test_MRCE()

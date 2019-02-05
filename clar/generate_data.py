import copy
import numpy as np

from numpy.linalg import norm
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state

from clar.utils import sqrtm


def get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_iid", n_channels=20,
        n_sources=20, n_times=30, n_epochs=50, n_active=3, rho=0.3,
        rho_noise=0.6,
        SNR=1, seed=0):

    rng = check_random_state(seed)

    X = get_dictionary(
        dictionary_type, n_channels=n_channels,
        n_sources=n_sources, rho=rho, seed=seed)
    Sigma_star = get_Sigma_star(
        noise_type=noise_type, n_channels=n_channels, rho_noise=rho_noise, seed=seed)

    rng = check_random_state(seed)
    # creates the signal XB
    B_star = np.zeros([n_sources, n_times])
    supp = rng.choice(n_sources, n_active, replace=False)
    B_star[supp, :] = rng.randn(n_active, n_times)

    return get_data_from_X_S_and_B_star(
        X, B_star, Sigma_star, n_epochs=n_epochs,
        n_active=n_active, SNR=SNR, seed=seed)


def get_Sigma_star(noise_type="Gaussian_iid", n_channels=20, rho_noise=0.7, seed=0):
    if noise_type == "Gaussian_iid":
        Sigma_star = np.eye(n_channels)
    elif noise_type == "Gaussian_multivariate":
        vect = rho_noise ** np.arange(n_channels)
        Sigma_star = toeplitz(vect, vect)
    return Sigma_star


def get_data_from_X_S_and_B_star(X, B_star, S_star, n_epochs=50, n_active=3, SNR=0.5, seed=0):
    rng = check_random_state(seed)
    XB = X @ B_star
    n_channels, n_sources = X.shape
    _, n_times = B_star.shape
    # creates the noise
    noise_all_epochs = np.empty((n_epochs, n_channels, n_times))
    for l in range(n_epochs):
        noise = S_star @ rng.randn(n_channels, n_times)
        noise_all_epochs[l, :, :] = noise
    denom = np.sqrt((noise_all_epochs ** 2).sum(axis=(1, 2))).mean()
    multiplicativ_factor = norm(XB, ord='fro') / denom
    multiplicativ_factor /= SNR
    noise_all_epochs *= multiplicativ_factor

    # add noise to signal
    all_epochs = noise_all_epochs + XB
    return X, all_epochs, B_star, (multiplicativ_factor, S_star)


def get_dictionary(
        dictionary_type, n_channels=20, n_sources=30,
        rho=0.3, seed=0):
    rng = check_random_state(seed)

    if dictionary_type == 'Toeplitz':
        X = get_toeplitz_dictionary(
            n_channels=n_channels, n_sources=n_sources, rho=rho, seed=seed)
    elif dictionary_type == 'Gaussian':
        X = rng.randn(n_channels, n_sources)
    else:
        raise NotImplementedError("No dictionary '{}' in maxsparse"
                                  .format(dictionary_type))
    normalize(X)
    return X


def normalize(X):
    for i in range(X.shape[1]):
        X[:, i] /= norm(X[:, i])
    return X


def get_toeplitz_dictionary(
        n_channels=20, n_sources=30, rho=0.3, seed=0):
    rng = check_random_state(seed)
    vect = rho ** np.arange(n_sources)
    covar = toeplitz(vect, vect)
    X = rng.multivariate_normal(np.zeros(n_sources), covar, n_channels)
    return X

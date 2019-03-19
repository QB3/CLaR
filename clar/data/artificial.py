import numpy as np

from numpy.linalg import norm
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state


def get_data_me(
        dictionary_type="Gaussian", noise_type="Gaussian_iid", n_channels=20,
        n_sources=20, n_times=30, n_epochs=50, n_active=3, rho=0.3,
        rho_noise=0.6,
        SNR=1, seed=0,
        meg=None, eeg=None):
    """Simulate artificial data.

    Parameters:
    ----------
    dictionary_type: string
        "Gaussian", "Toeplitz", real_me
    noise_type: string
         "Gaussian_iid", or "Gaussian_multivariate"
    n_channels: int
        number of channels
    n_sources: int
        number of potential sources.
    n_times: int
        number of time points
    n_epochs: int
        number of epochs/repetitions
    n_active: int
        number of active sources
    rho: float
        coefficient of correlation for the Toeplitz-corralted dictionary
    rho_noise: float
        coefficient of correlation for the Toeplitz covariance of the noise
    SNR: float
        Signal to Noise Ratio
    seed: int
    meg: bool or string
        True, "mag" or "eeg".
        If True, keeps magnetometers and gradiometers in cov.
        If "grad", keeps only gradiometers in cov.
        If "mag", keeps only the magnetometers in cov.
    eeg: bool
        If True keep electro-ancephalogramme in cov.
        If False remove electro-ancephalogramme in cov.

    Returns
    -------
    dictionary: np.array, shape (n_sensors, n_sources)
        dictionary/gain matrix
    all_epochs: np.array, shape (n_epochs, n_sensors, n_times)
        data observed
    B_star: np.array, shape (n_sources, n_times))
        real regression coefficients
    multiplicativ_factor
    S_star: np.array, shape (n_sensors, n_sensors)
        covariane matrix
    """
    rng = check_random_state(seed)

    X = get_dictionary(
        dictionary_type, n_channels=n_channels,
        n_sources=n_sources, rho=rho, meg=meg, eeg=eeg, seed=seed)
    S_star = get_S_star(
        noise_type=noise_type, n_channels=n_channels,
        rho_noise=rho_noise, meg=meg, eeg=eeg, seed=seed)

    rng = check_random_state(seed)
    # creates the signal XB
    B_star = np.zeros([n_sources, n_times])
    supp = rng.choice(n_sources, n_active, replace=False)
    B_star[supp, :] = rng.randn(n_active, n_times)

    X, all_epochs, B_star, (multiplicativ_factor, S_star) =\
        get_data_from_X_S_and_B_star(
        X, B_star, S_star, n_epochs=n_epochs,
        n_active=n_active, SNR=SNR, seed=seed)
    return X, all_epochs, B_star, (multiplicativ_factor, S_star)


def get_S_star(
        noise_type="Gaussian_iid", n_channels=20, rho_noise=0.7, seed=0,
        meg=True, eeg=True):
        """Simulate co-standard devation matrix.

        Parameters:
        ----------
        noise_type: string
            "Gaussian_iid", or "Gaussian_multivariate"
        n_channels: int
            number of channels
        rho_noise: float
            coefficient of correlation for the Toeplitz covariance of the noise
        seed: int
        meg: bool or string
            True, "mag" or "eeg".
            If True, keeps magnetometers and gradiometers in cov.
            If "grad", keeps only gradiometers in cov.
            If "mag", keeps only the magnetometers in cov.
        eeg: bool
            If True keep electro-ancephalogramme in cov.
            If False remove electro-ancephalogramme in cov.

        Returns
        -------
        S_star: np.array, shape (n_sensors, n_sensors)
            co-satndard deviation matrix
        """
        if noise_type == "Gaussian_iid":
            S_star = np.eye(n_channels)
        elif noise_type == "Gaussian_multivariate":
            vect = rho_noise ** np.arange(n_channels)
            S_star = toeplitz(vect, vect)
        else:
            raise ValueError("Unknown noise type %s" % noise_type)
        return S_star


def get_data_from_X_S_and_B_star(
        X, B_star, S_star, n_epochs=50,
        n_active=3, SNR=0.5, seed=0):
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
        rho=0.3, meg=None, eeg=None, seed=0):
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
    """This function returns a toeplitz dictionnary phi.

    Maths formula:
    S = toepltiz(\rho ** [|0, n_sources-1|], \rho ** [|0, n_sources-1|])
    X[:, i] \sim \mathcal{N}(0, S).

    Parameters
    ----------
    n_channels: int
        number of channels/measurments in your problem
    n_labels: int
        number of labels/atoms in your problem
    rho: float
        correlation matrix

    Results
    -------
    X : array, shape (n_channels, n_labels)
        The dictionary.
    """
    rng = check_random_state(seed)
    vect = rho ** np.arange(n_sources)
    covar = toeplitz(vect, vect)
    X = rng.multivariate_normal(np.zeros(n_sources), covar, n_channels)
    return X


def decimate(M, n_channels, axis, seed):
    if n_channels == M.shape[0] or n_channels == -1:
        return M

    n_channels_max = M.shape[0]
    rng = check_random_state(seed)

    to_choose = rng.choice(np.arange(n_channels_max), n_channels)
    to_choose.sort()
    if axis.__contains__(1):
        M = M[:, to_choose]
    if axis.__contains__(0):
        M = M[to_choose, :]
    return M

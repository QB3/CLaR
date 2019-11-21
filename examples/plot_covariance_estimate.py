"""
=======================================================
Plot an estimate of the covariance matrix with CLaR
=======================================================

The example runs CLaR on simulated data.
"""

import numpy as np
import matplotlib

import matplotlib.pyplot as plt


from clar.solvers import solver
from clar.utils import get_alpha_max_me, get_sigma_min
from clar.data.artificial import get_data_me


rho_noise = 0.3
SNR = 0.5
n_channels = 20
n_times = 30
n_sources = 10
n_epochs = 50
n_active = 3
gap_freq = 50
update_S_freq = 10
n_iter = 10**4
tol = 1e-7


X, all_epochs, B_star, S_star = get_data_me(
    dictionary_type="Gaussian", noise_type="Gaussian_multivariate",
    n_epochs=n_epochs, n_channels=n_channels, n_times=n_times,
    n_sources=n_sources, n_active=n_active, rho_noise=rho_noise,
    SNR=SNR)
S_star = S_star[-1]

Y = np.mean(all_epochs, axis=0)
sigma_min = get_sigma_min(Y)


alpha_max = get_alpha_max_me(X, all_epochs, sigma_min)
alpha = alpha_max / 5

print("alpha = %.2e" % alpha)
print("sigma_min = %.2e" % sigma_min)
B_clar, S_inv, E, gaps_me = solver(
    X, all_epochs, alpha, alpha_max, sigma_min, B0=None,
    n_iter=n_iter, gap_freq=50,
    S_freq=update_S_freq, tol=tol, pb_name="CLAR")
gap_me = gaps_me[-1]
assert gap_me < tol

try:
    matplotlib.rcParams["text.usetex"] = True
except:
    print("Could not use tex for matplotlib rendering")

fig, axarr = plt.subplots(1, 2)
S = np.linalg.inv(S_inv)
labels = [r"$S^*$", r"$\hat S^{\mathrm{CLaR}}$"]
for ax, S, label in zip(axarr, [S_star, S], labels):
    ax.imshow(S)
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label)
plt.show(block=False)


supp = np.where(B_clar.any(axis=1))[0]
true_supp = np.where(B_star.any(axis=1))[0]


TP = len(np.intersect1d(supp, true_supp))
FP = len(supp) - TP

print("TP: %d" % TP)
print("FP: %d" % TP)

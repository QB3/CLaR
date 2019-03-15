import numpy as np

from numba import njit
from numpy.linalg import norm

from .utils import condition_better, BST
from .duality_gap import (
    get_duality_gap, get_duality_gap_me, get_duality_gap_mtl)


def get_path(
        X, measurement, list_pourcentage_alpha, alpha_max,
        sigma_min, B0=None,
        n_iter=10**4, tol=10**-4, gap_freq=10, active_set_freq=5,
        update_S_freq=10, solver_name="CLAR", use_accel=False,
        verbose=True, use_heuristic_stopping_criterion=False):
    dict_masks = {}
    dict_dense_Bs = {}
    B_hat = None

    for n_alpha, pourcentage_alpha in enumerate(list_pourcentage_alpha):
        print("--------------------------------------------------------")
        print("%i-th alpha over %i" % (n_alpha, len(list_pourcentage_alpha)))
        # unique params to store results
        alpha = pourcentage_alpha * alpha_max
        # run solver of solver_name
        B_hat, _, _, _ = solver(
            X, measurement, alpha, alpha_max, sigma_min, B0=B_hat,
            n_iter=n_iter, gap_freq=gap_freq, active_set_freq=active_set_freq,
            update_S_freq=update_S_freq, solver_name=solver_name, tol=tol,
            use_accel=use_accel,
            use_heuristic_stopping_criterion=use_heuristic_stopping_criterion)
        # save the results
        mask = np.abs(B_hat).sum(axis=1) != 0
        str_pourcentage_alpha = '%0.10f' % pourcentage_alpha
        if solver_name == "MTLME":
            n_sources = X.shape[1]
            n_epochs, _, n_times = measurement.shape
            B_reshaped = B_hat.reshape((n_sources, n_epochs, n_times))
            B_reshaped = B_reshaped.mean(axis=1)
            dict_masks[str_pourcentage_alpha] = mask
            dict_dense_Bs[str_pourcentage_alpha] = B_reshaped[mask, :]
        else:
            dict_masks[str_pourcentage_alpha] = mask
            dict_dense_Bs[str_pourcentage_alpha] = B_hat[mask, :]
    assert len(dict_dense_Bs.keys()) == len(list_pourcentage_alpha)
    return dict_masks, dict_dense_Bs


def solver(
        X, all_epochs, alpha, alpha_max, sigma_min, B0=None,
        n_iter=10**4, tol=10**-4, gap_freq=10, active_set_freq=5,
        update_S_freq=10, solver_name="CLAR", use_accel=False,
        verbose=True, use_heuristic_stopping_criterion=False):
    """
    Parameters
    --------------
    X: np.array, shape (n_sensors, n_sources)
        gain matrix
    all_epochs: np.array, shape (n_epochs, n_sensors, n_times)
        observations
    B0: np.array, shape (n_sources, n_time)
        initial value of B
    n_iter: int
        nuber of iterations of the algorithm
    S0: np.array, shape (n_sensors, n_sensors)
        initial value of the covariance
    """
    if use_accel and solver_name != "SGCL":
        raise NotImplementedError()

    # if not np.isfortran(X):
    X = np.asfortranarray(X, dtype='float64')

    n_sources = X.shape[1]
    n_times = all_epochs.shape[-1]
    if verbose:
        print("--------- %s -----------------" % solver_name)

    if B0 is None:
        if solver_name != "MTLME":
            B = np.zeros((n_sources, n_times), dtype=float)
        else:
            n_epochs, _, n_times = all_epochs.shape
            B = np.zeros((n_sources, n_times * n_epochs), dtype=float)
    else:
        B = B0.copy().astype(np.float64)

    if solver_name in ("SGCL", "MTL"):
        if all_epochs.ndim != 2:
            raise ValueError("Wrong number of dimensions, expected 2, "
                             "got %d " % all_epochs.ndim)
        observations = all_epochs[None, :, :]

    elif solver_name == "CLAR":
        observations = all_epochs
    elif solver_name == "MTLME":
        assert all_epochs.ndim == 3
        observations = all_epochs.transpose((1, 0, 2))
        observations = observations.reshape(observations.shape[0], -1)
        observations = observations.reshape((1, *observations.shape))
        n_epochs, n_channels, n_times = all_epochs.shape

    else:
        raise ValueError("Unknown solver %s" % solver_name)

    results = solver_(
        observations, X, alpha, alpha_max, sigma_min, B, n_iter, gap_freq,
        use_accel, active_set_freq, update_S_freq, tol=tol,
        solver_name=solver_name, verbose=verbose,
        use_heuristic_stopping_criterion=use_heuristic_stopping_criterion)
    return results


def solver_(
        all_epochs, X, alpha, alpha_max, sigma_min, B, n_iter, gap_freq, use_accel,
        active_set_freq=5, update_S_freq=10, tol=10**-4,
        solver_name="CLAR", verbose=True, use_heuristic_stopping_criterion=False):
    gaps = []
    gaps_acc = []
    E = []  # E for energy, ie value of primal objective

    p_obj = np.infty
    d_obj_acc = - np.infty
    n_epochs, n_sensors, n_times = all_epochs.shape

    if solver_name == "CLAR":
        # compute Y2, costly quantity to compute once
        Y2 = np.zeros((n_sensors, n_sensors))
        Y = np.zeros((n_sensors, n_times))
        for l in range(n_epochs):
            Y2 += all_epochs[l, :, :] @ all_epochs[l, :, :].T
            Y += all_epochs[l, :, :]
        Y2 /= n_epochs
        Y /= n_epochs
    elif solver_name == "MTL" or "SGCL":
        Y = all_epochs[0]
        Y2 = None
    elif solver_name == "MTLME":
        Y = all_epochs

    if use_accel:
        K = 6
        last_K_res = np.zeros((K, n_sensors * n_times))
        onesKm1 = np.ones(K - 1)
        U = np.zeros((K - 1, n_sensors * n_times))
    R = np.asfortranarray(Y - X @ B, dtype='float64')
    # compute the value of the first primal
    B_first = np.zeros(B.shape)
    S_trace_first, S_inv_first = update_S(
        Y, X, B_first, Y, Y2, sigma_min, solver_name)
    if solver_name == "CLAR":
        primal_first, _ = get_duality_gap_me(
            X, all_epochs, B_first, S_trace_first, S_inv_first,
            sigma_min, alpha)
    elif solver_name == "SGCL":
        S_inv_R = S_inv_first @ Y
        primal_first, _ = get_duality_gap(
            Y, X, Y, B_first, S_trace_first,
            S_inv_R, sigma_min, alpha)
    elif solver_name == "MTL" or solver_name == "MTLME":
        primal_first, _ = get_duality_gap_mtl(
            X, Y, B_first, alpha)
    E.append(primal_first)
    ##########################################################
    for t in range(n_iter):
        #####################################################
        # update S
        if t % update_S_freq == 0:
            if solver_name == "CLAR":
                XB = X @ B
                YXB = Y @ XB.T
                ZZT = (Y2 - YXB - YXB.T + XB @ XB.T) / n_times
                S_trace, S_inv = condition_better(ZZT, sigma_min)
                S_inv_R = np.asfortranarray(S_inv @ R)
                S_inv_X = S_inv @ X
            elif solver_name == "SGCL":
                Z = Y - X @ B
                ZZT = Z @ Z.T / n_times
                S_trace, S_inv = condition_better(ZZT, sigma_min)
                S_inv_R = np.asfortranarray(S_inv @ R)
                S_inv_X = S_inv @ X
            elif solver_name == "MTL" or solver_name == "MTLME":
                # this else case is for MTL
                # dummy variables for njit to work:
                S_trace = n_sensors
                S_inv = np.eye(1)
                S_inv_R = R
                S_inv_X = X
        ###################################################
        # update B
        update_B(
            X, Y, B, R, S_inv_R, S_inv_X,
            alpha, solver_name,
            active_set_freq)
        # compute duality gap
        if t % gap_freq == 0:
            if solver_name == "CLAR":
                p_obj, d_obj = get_duality_gap_me(
                    X, all_epochs, B, S_trace, S_inv,
                    sigma_min, alpha)
            elif solver_name == "SGCL":
                p_obj, d_obj = get_duality_gap(
                    R, X, Y, B, S_trace,
                    S_inv_R, sigma_min, alpha)

            elif solver_name == "MTL" or solver_name == "MTLME":
                p_obj, d_obj = get_duality_gap_mtl(
                    X, Y, B, alpha)

            gap = p_obj - d_obj
            E.append(p_obj)
            gaps.append(gap)
            if verbose:
                print("p_obj: %.6e" % (p_obj))
                print("d_obj: %.6e" % (d_obj))
                print("iteration: %d, gap: %.4e" % (t, gap))
        if t // gap_freq >= 1 and use_heuristic_stopping_criterion:
            heuristic_stopping_criterion = (E[-2] - E[-1]) < tol * E[0] / 10
        else:
            heuristic_stopping_criterion = False
        if gap < tol * E[0] or \
            (use_accel and (p_obj - d_obj_acc < tol * E[0])) \
                or heuristic_stopping_criterion:
            break

    results = (B, S_inv, np.asarray(E), np.asarray(gaps))
    if use_accel:
        results = (B, S_inv, np.asarray(
            E), (np.asarray(gaps), np.asarray(gaps_acc)))
    return results


def update_S(Y, X, B, R, Y2, sigma_min, solver_name):
    n_times = B.shape[1]
    n_sensors = Y.shape[-1]
    if solver_name == "CLAR":
        XB = X @ B
        YXB = Y @ XB.T
        ZZT = (Y2 - YXB - YXB.T + XB @ XB.T) / n_times
        S_trace, S_inv = condition_better(ZZT, sigma_min)
        S_inv_R = np.asfortranarray(S_inv @ R)
        S_inv_X = S_inv @ X
    elif solver_name == "SGCL":
        Z = Y - X @ B
        ZZT = Z @ Z.T / n_times
        S_trace, S_inv = condition_better(ZZT, sigma_min)
        S_inv_R = np.asfortranarray(S_inv @ R)
        S_inv_X = S_inv @ X
    elif solver_name == "MTL" or solver_name == "MTLME":
        # this else case is for MTL
        # dummy variables for njit to work:
        S_trace = n_sensors
        S_inv = np.eye(1)
        S_inv_R = R
        S_inv_X = X
    return S_trace, S_inv


@njit
def update_B(
        X, Y, B, R,  S_inv_R, S_inv_X,
        alpha, solver_name,
        active_set_passes=5):
    n_sensors, n_times = Y.shape
    n_sources = X.shape[1]

    is_not_MTL = (solver_name != "MTL") and (solver_name != "MTLME")

    active_set = np.ones(n_sources)

    L = np.empty(n_sources)
    for j in range(n_sources):
        L[j] = X[:, j] @ S_inv_X[:, j]

    # practical hack to speed up convergence, does not affect convergence
    for t in range(active_set_passes):
        if t == 0:
            sources_to_update = np.arange(n_sources)
        else:
            sources_to_update = np.where(active_set != 0)[0]

        for j in sources_to_update:
            # update line j of B
            if active_set[j]:
                R += X[:, j:j+1] @ B[j:j+1, :]
                if is_not_MTL:
                    S_inv_R += S_inv_X[:, j:j+1] @ B[j:j+1, :]

            B[j:j+1, :], line_is_zero = BST(
                X[:, j:j+1].T @ S_inv_R / L[j],
                alpha * n_sensors * n_times / L[j])

            active_set[j] = not line_is_zero
            if not line_is_zero:
                R -= X[:, j:j+1] @ B[j:j+1, :]
                if is_not_MTL:
                    S_inv_R -= S_inv_X[:, j:j+1] @ B[j:j+1, :]

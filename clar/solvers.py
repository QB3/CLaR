import numpy as np

from numba import njit
from scipy import linalg

from sklearn.linear_model import cd_fast
from sklearn.utils import check_random_state
from clar.utils import (
    clp_sqrt, BST,
    condition_better_glasso)
from .duality_gap import (
    get_duality_gap, get_duality_gap_me, get_duality_gap_mtl,
    get_p_obj_mrce)


def get_path(
        X, measurement, list_pourcentage_alpha, alpha_max,
        sigma_min, B0=None, n_iter=10**4, tol=10**-4, gap_freq=10,
        active_set_freq=5, S_freq=10, pb_name="CLAR", use_accel=False,
        verbose=True, heur_stop=False):
    dict_masks = {}
    dict_dense_Bs = {}
    B_hat = B0

    for n_alpha, pourcentage_alpha in enumerate(list_pourcentage_alpha):
        print("--------------------------------------------------------")
        print("%i-th alpha over %i" % (n_alpha, len(list_pourcentage_alpha)))
        # unique params to store results
        alpha = pourcentage_alpha * alpha_max
        # run solver of pb_name
        B_hat, _, _, _ = solver(
            X, measurement, alpha, sigma_min, B0=B_hat,
            n_iter=n_iter, gap_freq=gap_freq, active_set_freq=active_set_freq,
            S_freq=S_freq, pb_name=pb_name, tol=tol,
            use_accel=use_accel,
            heur_stop=heur_stop, verbose=verbose)
        # save the results
        mask = np.abs(B_hat).sum(axis=1) != 0
        str_pourcentage_alpha = '%0.10f' % pourcentage_alpha
        if pb_name == "MTLME":
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
        X, all_epochs, alpha, sigma_min, B0=None,
        n_iter=10**4, tol=10**-4, gap_freq=10, active_set_freq=5,
        S_freq=10, pb_name="CLAR", use_accel=False,
        verbose=True, heur_stop=False,
        alpha_Sigma_inv=0.0001):
    """
    Parameters
    --------------
    X: np.array, shape (n_sensors, n_sources)
        gain matrix
    all_epochs: np.array, shape (n_epochs, n_sensors, n_times)
        observations
    alpha: float
        positive number, coefficient multiplying the penalization
    alpha_max: float
        positive number, if alpha is bigger than alpha max, B=0
    sigma_min: float
        positive number, value to which to eigenvalue smaller than sigma_min
        are put to when computing the inverse of ZZT
    B0: np.array, shape (n_sources, n_time)
        initial value of B
    n_iter: int
        number of iterations of the algorithm
    tol : float
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``
    gap_freq: int
        Compute the duality gap every gap_freq iterations.
    active_set_freq: int
        When updating B, while B_{j, :} != 0,  B_{j, :} keeps to
        be updated, at most active_set_freq times.
    S_freq: int
        S is updated every S times.
    pb_name: str
        choose the problem you want to solve between
        "MTL", "MTLME", "SGCL", "CLAR" and "mrce"
    use_accel: bool
        States if you want to use accelratio while computing the dual.
    heur_stop: bool
        States if you want to use an heuristic stopping criterion ot stop
        the algo.
        Here the heuristic stopping criterion is
        primal[i] - primal[i+1] < primal[0] * tol / 10.
    """

    if use_accel and pb_name != "SGCL":
        raise NotImplementedError()

    X = np.asfortranarray(X, dtype='float64')

    n_sources = X.shape[1]
    n_times = all_epochs.shape[-1]
    if verbose:
        print("--------- %s -----------------" % pb_name)

    if B0 is None:
        if pb_name != "MTLME":
            B = np.zeros((n_sources, n_times), dtype=float)
        else:
            n_epochs, _, n_times = all_epochs.shape
            B = np.zeros((n_sources, n_times * n_epochs), dtype=float)
    else:
        B = B0.copy().astype(np.float64)

    if pb_name in ("SGCL", "MTL"):
        if all_epochs.ndim != 2:
            raise ValueError("Wrong number of dimensions, expected 2, "
                             "got %d " % all_epochs.ndim)
        observations = all_epochs[None, :, :]
    elif pb_name in ("CLAR", "mrce"):
        observations = all_epochs
    elif pb_name == "MTLME":
        if all_epochs.ndim != 3:
            raise ValueError(
                "Wrong number of dimensions, expected 2, "
                "got %d " % all_epochs.ndim)
        observations = all_epochs.transpose((1, 0, 2))
        observations = observations.reshape(observations.shape[0], -1)
        observations = observations.reshape((1, *observations.shape))
        n_epochs, _, n_times = all_epochs.shape
    else:
        raise ValueError("Unknown solver %s" % pb_name)

    results = solver_(
        observations, X, alpha, sigma_min, B, n_iter, gap_freq,
        use_accel, active_set_freq, S_freq, tol=tol,
        pb_name=pb_name, verbose=verbose,
        heur_stop=heur_stop, alpha_Sigma_inv=alpha_Sigma_inv)
    return results


def solver_(
        all_epochs, X, alpha, sigma_min, B, n_iter, gap_freq,
        use_accel, active_set_freq=5, S_freq=10, tol=10**-4,
        pb_name="CLAR", verbose=True, heur_stop=False, alpha_Sigma_inv=0.01):
    gaps = []
    gaps_acc = []
    E = []  # E for energy, ie value of primal objective
    p_obj = np.infty
    d_obj = - np.infty
    d_obj_acc = - np.infty
    n_epochs, n_sensors, n_times = all_epochs.shape

    if pb_name in ("CLAR", "mrce"):
        # compute Y2, costly quantity to compute once
        Y2 = np.zeros((n_sensors, n_sensors))
        Y = np.zeros((n_sensors, n_times))
        for l in range(n_epochs):
            Y2 += all_epochs[l, :, :] @ all_epochs[l, :, :].T
            Y += all_epochs[l, :, :]
        Y2 /= n_epochs
        Y /= n_epochs
    elif pb_name in ("MTL", "SGCL", "MTLME"):
        Y = all_epochs[0]
        Y2 = None
    elif pb_name == "MTLME":
        Y = all_epochs

    if use_accel:
        K = 6
        last_K_res = np.zeros((K, n_sensors * n_times))
        onesKm1 = np.ones(K - 1)
        U = np.zeros((K - 1, n_sensors * n_times))
    R = np.asfortranarray(Y - X @ B, dtype='float64')
    # compute the value of the first primal
    B_first = np.zeros(B.shape)
    if pb_name != "mrce":
        S_trace_first, S_inv_first = update_S(
            Y, X, B_first, Y2, sigma_min, pb_name)
    if pb_name == "CLAR":
        primal_first, _ = get_duality_gap_me(
            X, all_epochs, B_first, S_trace_first, S_inv_first,
            sigma_min, alpha)
    elif pb_name == "SGCL":
        S_inv_R = S_inv_first @ Y
        primal_first, _ = get_duality_gap(
            Y, X, Y, B_first, S_trace_first,
            S_inv_R, sigma_min, alpha)
    elif pb_name in("MTL", "MTLME"):
        primal_first, _ = get_duality_gap_mtl(
            X, Y, B_first, alpha)
    elif pb_name == "mrce":
        Sigma = Y2 / n_times
        Sigma_inv = linalg.pinvh(Sigma)
        primal_first = get_p_obj_mrce(
            X, Y, Y2, Sigma, Sigma_inv, alpha,
            alpha_Sigma_inv, B_first)
    E.append(primal_first)
    print("------------------------")
    print("First primal: %0.2e" % primal_first)
    ##########################################################
    for t in range(n_iter):
        #####################################################
        # update S
        if t % S_freq == 0:
            if pb_name == "CLAR":
                XB = X @ B
                YXB = Y @ XB.T
                ZZT = (Y2 - YXB - YXB.T + XB @ XB.T) / n_times
                S_trace, S_inv = clp_sqrt(ZZT, sigma_min)
                S_inv_R = np.asfortranarray(S_inv @ R)
                S_inv_X = S_inv @ X
            elif pb_name == "mrce":
                XB = X @ B
                YXB = Y @ XB.T
                emp_cov = (Y2 - YXB - YXB.T + XB @ XB.T) / n_times
                Sigma, Sigma_inv = update_sigma_glasso(
                    emp_cov, alpha_Sigma_inv)
                _, Sigma_inv = condition_better_glasso(Sigma_inv, sigma_min)
                S_inv_R = Sigma_inv @ R  # be careful this is not real S_inv_R
                S_inv_X = Sigma_inv @ X
            elif pb_name == "SGCL":
                Z = Y - X @ B
                ZZT = Z @ Z.T / n_times
                S_trace, S_inv = clp_sqrt(ZZT, sigma_min)
                S_inv_R = np.asfortranarray(S_inv @ R)
                S_inv_X = S_inv @ X
            elif pb_name in ("MTL", "MTLME"):
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
            alpha, pb_name,
            active_set_freq)
        # compute duality gap
        if t % gap_freq == 0:
            if pb_name == "CLAR":
                p_obj, d_obj = get_duality_gap_me(
                    X, all_epochs, B, S_trace, S_inv,
                    sigma_min, alpha)
            elif pb_name == "SGCL":
                p_obj, d_obj = get_duality_gap(
                    R, X, Y, B, S_trace,
                    S_inv_R, sigma_min, alpha)

                if use_accel:
                    if t // gap_freq < K:
                        last_K_res[t // gap_freq, :] = R.ravel(order="F")
                        R_acc = R
                    else:
                        for k in range(K - 1):
                            last_K_res[k] = last_K_res[k + 1]
                        last_K_res[K - 1] = R.ravel(order='F')
                        R_acc = R

                        for k in range(K - 1):
                            U[k] = last_K_res[k + 1] - last_K_res[k]
                        C = np.dot(U, U.T)

                        try:
                            z = np.linalg.solve(C, onesKm1)
                            c = z / z.sum()
                            R_acc = (np.sum(
                                last_K_res[:-1] * np.expand_dims(c, 1),
                                axis=0)).reshape(
                                    n_sensors, n_times, order='F')
                        except np.linalg.LinAlgError:
                            print("##### linalg failed")
                            R_acc = R

                    S_trace_acc, S_inv_acc = clp_sqrt(
                        R_acc @ R_acc.T / n_times, sigma_min)
                    _, d_obj_acc = get_duality_gap(
                        R, X, Y, B, S_trace_acc, S_inv_acc @ R_acc,
                        sigma_min, alpha)

                    if verbose:
                        print("gap_acc: %.2e" % (p_obj - d_obj_acc))
                    gaps_acc.append(p_obj - d_obj_acc)
            elif pb_name in ("MTL", "MTLME"):
                p_obj, d_obj = get_duality_gap_mtl(
                    X, Y, B, alpha)
            elif pb_name == "mrce":
                p_obj = get_p_obj_mrce(
                    X, Y, Y2, Sigma, Sigma_inv, alpha,
                    alpha_Sigma_inv, B)
            gap = p_obj - d_obj
            E.append(p_obj)
            gaps.append(gap)
            if verbose:
                print("p_obj: %.6e" % (p_obj))
                print("d_obj: %.6e" % (d_obj))
                print("iteration: %d, gap: %.4e" % (t, gap))
        if t // gap_freq >= 1 and heur_stop:
            heuristic_stopping_criterion = (E[-2] - E[-1]) < tol * E[0] / 10
        else:
            heuristic_stopping_criterion = False
        if gap < tol * E[0] or \
            (use_accel and (p_obj - d_obj_acc < tol * E[0])) \
                or heuristic_stopping_criterion:
            break
    if pb_name != "mrce":
        results = (B, S_inv, np.asarray(E), np.asarray(gaps))
    else:
        results = (B, (Sigma, Sigma_inv), np.asarray(E), np.asarray(gaps))
    if use_accel:
        results = (B, S_inv, np.asarray(E),
                   (np.asarray(gaps), np.asarray(gaps_acc)))
    return results


def update_S(Y, X, B, Y2, sigma_min, pb_name):
    n_times = B.shape[1]
    n_sensors = Y.shape[-1]
    if pb_name == "CLAR":
        XB = X @ B
        YXB = Y @ XB.T
        ZZT = (Y2 - YXB - YXB.T + XB @ XB.T) / n_times
        S_trace, S_inv = clp_sqrt(ZZT, sigma_min)
    elif pb_name == "SGCL":
        Z = Y - X @ B
        ZZT = Z @ Z.T / n_times
        S_trace, S_inv = clp_sqrt(ZZT, sigma_min)
    elif pb_name in ("MTL", "MTLME"):
        # this else case is for MTL
        # dummy variables for njit to work:
        S_trace = n_sensors
        S_inv = np.eye(1)
    return S_trace, S_inv


@njit
def update_B(
        X, Y, B, R, S_inv_R, S_inv_X,
        alpha, pb_name,
        active_set_passes=5):
    n_sensors, n_times = Y.shape
    n_sources = X.shape[1]

    is_not_MTL = pb_name not in ("MTL", "MTLME")

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


def update_sigma_glasso(
        emp_cov, alpha_Sigma_inv, cov_init=None,
        enet_tol=1e-4, max_iter=1e4, eps=np.finfo(np.float64).eps):
    _, n_features = emp_cov.shape
    if cov_init is None:
        covariance_ = emp_cov.copy()
        # covariance_ = clp_sqrt(covariance_, sigmamin ** 2)
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed,
    #  as in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends
    # to make the convergence go faster.
    covariance_ *= 0.95
    diagonal = emp_cov.flat[::n_features + 1]
    covariance_.flat[::n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    indices = np.arange(n_features)
    errors = dict(over='raise', invalid='ignore')
    sub_covariance = np.copy(covariance_[1:, 1:], order='C')

    for idx in range(n_features):
        # To keep the contiguous matrix `sub_covariance` equal to
        # covariance_[indices != idx].T[indices != idx]
        # we only need to update 1 column and 1 line when idx changes
        if idx > 0:
            di = idx - 1
            sub_covariance[di] = covariance_[di][indices != idx]
            sub_covariance[:, di] = covariance_[:, di][indices != idx]
        else:
            sub_covariance[:] = covariance_[1:, 1:]
        row = emp_cov[idx, indices != idx]
        with np.errstate(**errors):
            # Use coordinate descent
            coefs = -(precision_[indices != idx, idx] /
                      (precision_[idx, idx] + 1000 * eps))
            coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                coefs, alpha_Sigma_inv, 0, sub_covariance,
                row, row, max_iter, enet_tol,
                check_random_state(None), False)
        # Update the precision matrix
        precision_[idx, idx] = (
            1. / (covariance_[idx, idx] -
                  np.dot(covariance_[indices != idx, idx], coefs)))
        precision_[indices != idx, idx] = (- precision_[idx, idx] *
                                           coefs)
        precision_[idx, indices != idx] = (- precision_[idx, idx] *
                                           coefs)
        coefs = np.dot(sub_covariance, coefs)
        covariance_[idx, indices != idx] = coefs
        covariance_[indices != idx, idx] = coefs
    if not np.isfinite(precision_.sum()):
        raise FloatingPointError('The system is too ill-conditioned '
                                 'for this solver')
    return covariance_, precision_

import numpy as np

from copy import deepcopy


def trad_rhat(chains, split=True):
    n, m = chains.shape
    if split:
        chains = np.reshape(chains, (n // 2, 2 * m), order="F")
        n, m = chains.shape
    between_var = np.var(np.mean(chains, axis=0))
    within_var = np.mean(np.var(chains, axis=0))
    return np.sqrt((n - 1) / n + between_var / within_var)


def univariate_local_rhat(x, chains):
    return trad_rhat(chains <= x)


def univariate_grid_for_R(chains, max_nb_points=500):
    n, m = chains.shape
    grid = np.reshape(chains, (n * m))
    if max_nb_points != "ALL" and n * m > max_nb_points:
        grid = grid[np.linspace(1, n * m, max_nb_points, dtype=int) - 1]
    return grid


def rhat_infinity(chains, direction=None, max_nb_points=500):
    is_multivariate = len(chains.shape) == 3

    if is_multivariate:
        grid = multivariate_grid_for_R(chains, max_nb_points)
    else:
        grid = univariate_grid_for_R(chains, max_nb_points)

    max_R = 0
    if direction is None:
        for idx in range(grid.shape[0]):
            if is_multivariate:
                max_R = np.nanmax([max_R, multivariate_local_rhat(grid[idx], chains)])
            else:
                max_R = np.nanmax([max_R, univariate_local_rhat(grid[idx], chains)])
    else:
        for idx in range(grid.shape[0]):
            max_R = np.nanmax([max_R, multivariate_directed_local_rhat(grid[idx], chains, direction)])
    return max_R


def get_threshold(m, alpha=0.05):
    with open("data/quantile_r_inf.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                idx = row.index(str(alpha))
            if row[1] == str(m):
                threshold = row[idx]
            line_count += 1
    return float(threshold)


def get_multivariate_threshold(m, d, alpha=0.05):
    with open("data/quantile_max_r_inf.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                idx = row.index(str(alpha))
            if row[2] == str(m):
                if row[1] == str(d):
                    threshold = row[idx]
            line_count += 1
    return float(threshold)


######################################################################################################################


def multivariate_local_rhat(x, chains):
    return trad_rhat((chains.transpose(0, 2, 1) <= x).all(axis=2) * 1)


def multivariate_directed_local_rhat(x, chains, direction):
    n, d, m = chains.shape
    t_chains = chains.transpose(0, 2, 1)
    bool_chains = deepcopy(t_chains)
    for i in range(d):
        if direction[i] == 1:
            bool_chains[:, :, i] = t_chains[:, :, i] <= x[i]
        else:
            bool_chains[:, :, i] = t_chains[:, :, i] >= x[i]
    return trad_rhat(bool_chains.all(axis=2) * 1)


def multivariate_grid_for_R(chains, max_nb_points=500):
    n, d, m = chains.shape
    grid = np.reshape(chains.transpose(0, 2, 1), (n * m, d))
    if max_nb_points != "ALL" and n * m > max_nb_points:
        grid = grid[np.linspace(1, n * m, max_nb_points, dtype=int) - 1,]
    return grid


def multivariate_all_local_rhat(chains, max_nb_points=500):
    grid = multivariate_grid_for_R(chains, max_nb_points)
    r_values = []
    for idx in range(grid.shape[0]):
        r_values.append(multivariate_local_rhat(grid[idx], chains))
    return r_values


def rhat_infinity_max_directions(chains, max_nb_points=500):
    d = chains.shape[1]
    r_max = 0
    for i in range(2 ** (d - 1) - 1):
        direction = [int(x) for x in bin(i)[2:]]
        direction = [0] * (d - len(direction)) + direction
        r_max = np.nanmax([r_max, rhat_infinity(chains, direction, max_nb_points)])
    return r_max

######################################################################################################################


def localrhat_summary(data, var_names):
    m = len(data.posterior)
    d = len(var_names)
    res = az.summary(data, var_names)
    res["r_threshold"] = [round(get_threshold(m), 3)] * d
    res["r_hat_inf"] = [round(rhat_infinity(np.asarray(data.posterior[v]).transpose(1, 0)), 3) for v in var_names]

    if d < 6:
        mul_chains = np.asarray([data.posterior[v] for v in var_names]).transpose(2, 0, 1)
        print("Multivariate R-hat-infinity on all dependence directions: {:.3f} (threshold: {:.3f})"
              .format(rhat_infinity_max_directions(mul_chains), get_multivariate_threshold(m, d)))
    return res


def r_x_values(chains, max_nb_points=500):
    grid = np.sort(univariate_grid_for_R(chains, max_nb_points))
    r_val = [univariate_local_rhat(grid[idx], chains) for idx in range(grid.shape[0])]
    return grid, r_val




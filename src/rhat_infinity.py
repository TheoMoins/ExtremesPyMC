import numpy as np
import arviz as az

from copy import deepcopy
import csv


def trad_rhat(chains, split=True):
    """
    Compute the traditional Gelman-Rubin diagnostic.


    Parameters
        chains: an array of size n x m where n is the length of the chains and m is the number of chains.
        split: a boolean that indicate if the chains are splitted in half before computation.

    Return
        A real value, R-hat computed on the m chains.
    """
    n, m = chains.shape
    if split:
        chains = np.reshape(chains, (n // 2, 2 * m), order="F")
        n, m = chains.shape
    between_var = np.var(np.mean(chains, axis=0))
    within_var = np.mean(np.var(chains, axis=0))
    return np.sqrt((n - 1) / n + between_var / within_var)


def univariate_local_rhat(x, chains, split=True):
    """
    Compute R-hat(x), a version of univariate R-hat computed on indicator variables for a given quantile x.


    Parameters
        x: a float number corresponding to the quantile used for the computation of R-hat(x)
        chains: an array of size n x m where n is the length of the chains and m is the number of chains.
        split: a boolean that indicate if the chains are splitted in half before computation.

    Return
        A real value, R-hat(x) computed on the m chains.
    """
    return trad_rhat(chains <= x, split)


def univariate_grid_for_R(chains, max_nb_points=500):
    '''
    Return the set of points that will be used to estimate the supremum of R-hat(x) over x. 
    The number of different values taken by R-hat(x) can not exceed the number of samples, 
    so the number of points used for the computation is the minimum between nm and a threshold
    value that can be specified in the argument max_nb_points    
    
    Parameters
        max_nb_points: the maximal length of the grid in the case where the total number of samples is larger.
        chains: an array of size n x m where n is the length of the chains and m is the number of chains.
        
    Return 
        A list that contains the different x values.
    '''
    n, m = chains.shape
    grid = np.reshape(chains, (n * m))
    if max_nb_points != "ALL" and n * m > max_nb_points:
        grid = grid[np.linspace(1, n * m, max_nb_points, dtype=int) - 1]
    return grid


def rhat_infinity(chains, direction=None, max_nb_points=500, split=True):
    """
    Compute R-hat-infinity, a scalar summary of the function R-hat(x) corresponding to the supremum over the quantiles x.

    Parameters
        chains: an array of size n x m where n is the length of the chains and m is the number of chains.
                In the multivariate case, the size is n x d x m
        direction: a vector specifying which indicator to use for the multivariate case.
                   See the function multivariate_directed_local_rhat for more details.
        max_nb_points: the maximal length of the grid in the case where the total number of samples is larger.
        split: a boolean that indicate if the chains are splitted in half before computation.

    Return
        A real value, R-hat-infinity computed on the m chains.
    """
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
    '''
    Compute the threshold associated for R-hat-infinity to a given number of chains and confidence level.   
    
    Parameters
        m: the number of chain
        alpha: the confidence level
        
    Return 
        A real value, the suggested threshold for R-hat-infinity.
    '''
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
    """
    Compute the threshold associated for R-hat-infinity in the multivariate case.

    Parameters
        m: the number of chain
        d: the dimension of the chains
        alpha: the confidence level

    Return
        A real value, the suggested threshold for R-hat-infinity in the multivariate case.
    """
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


def multivariate_local_rhat(x, chains, split=True):
    '''
    Multivariate version of R-hat(x).   
    
    Parameters
        x: a vector of size d corresponding to the quantile used for the computation of R-hat(x).
        chains: an array of size n x d x m, where n is the length of the chains, 
                m is the number of chains and d is the dimension.
        split: a boolean that indicate if the chains are splitted in half before computation.
        
    Return 
        A real value, the multivartiate version of R-hat(x).
    '''
    return trad_rhat((chains.transpose(0, 2, 1) <= x).all(axis=2) * 1, split)


def multivariate_directed_local_rhat(x, chains, direction, split=True):
    """
    Compute R-hat(x) in the multivariate case, for a given quantile x and a given sense for the d different signs
    in the indicator variable.

    Parameters
        x: a vector of size d corresponding to the quantile used for the computation of R-hat(x).
        chains: an array of size n x d x m, where n is the length of the chains,
                m is the number of chains and d is the dimension.
        direction: a binary vector of size d indicating the signs in the indicator variable.
        split: a boolean that indicate if the chains are splitted in half before computation.

    Return
        A real value, the corresponding multivariate R-hat(x) computed on the given dependence direction.
    """
    n, d, m = chains.shape
    t_chains = chains.transpose(0, 2, 1)
    bool_chains = deepcopy(t_chains)
    for i in range(d):
        if direction[i] == 1:
            bool_chains[:, :, i] = t_chains[:, :, i] <= x[i]
        else:
            bool_chains[:, :, i] = t_chains[:, :, i] >= x[i]
    return trad_rhat(bool_chains.all(axis=2) * 1, split)


def multivariate_grid_for_R(chains, max_nb_points=500):
    """
    Return the set of points that will be used to estimate the supremum of R-hat(x)
    over x. The function works in the same way as univariate_grid_for_R.

    Parameters
        chains: an array of size n x d x m, where n is the length of the chains,
                m is the number of chains and d is the dimension.
        max_nb_points: the maximal length of the grid in the case where the total number of samples is larger.

    Return
        A list that contains the different x used for the computation of R-hat(x).
    """
    n, d, m = chains.shape
    grid = np.reshape(chains.transpose(0, 2, 1), (n * m, d))
    if max_nb_points != "ALL" and n * m > max_nb_points:
        grid = grid[np.linspace(1, n * m, max_nb_points, dtype=int) - 1,]
    return grid


def rhat_infinity_max_directions(chains, max_nb_points=500, split=True):
    """
    Compute the multivariate version of R-hat-infinity in all possible dependence direction
    and return the maximum value.

    Parameters
        chains: an array of size n x d x m, where n is the length of the chains,
                m is the number of chains and d is the dimension.
        max_nb_points: the maximal length of the grid in the case where the total number of samples is larger.

    Return
        A real value, the maximum value of all R-hat-infinity computed.
    """
    d = chains.shape[1]
    r_max = 0
    for i in range(2 ** (d - 1) - 1):
        direction = [int(x) for x in bin(i)[2:]]
        direction = [0] * (d - len(direction)) + direction
        r_max = np.nanmax([r_max, rhat_infinity(chains, direction, max_nb_points, split)])
    return r_max

######################################################################################################################


def localrhat_summary(data, var_names, split=True):
    """
    A summary of different convergence diagnostic, and in particular the local version of R-hat.
    Can only be used with PyMC3 and ArviZ!
    """
    m = len(data.posterior.chain)
    if split:
        m *= 2
    d = len(var_names)
    res = az.summary(data, var_names)
    res["r_threshold"] = [round(get_threshold(m), 3)] * d
    res["r_hat_infty"] = [round(rhat_infinity(np.asarray(data.posterior[v]).transpose(1, 0)), 3) for v in var_names]

    if d < 6:
        mul_chains = np.asarray([data.posterior[v] for v in var_names]).transpose(2, 0, 1)
        print("Multivariate R-hat-infinity on all dependence directions: {:.3f} (threshold: {:.3f})"
              .format(rhat_infinity_max_directions(mul_chains), get_multivariate_threshold(m, d)))
    return res


def r_x_values(chains, max_nb_points=500):
    grid = np.sort(univariate_grid_for_R(chains, max_nb_points))
    r_val = [univariate_local_rhat(grid[idx], chains) for idx in range(grid.shape[0])]
    return grid, r_val




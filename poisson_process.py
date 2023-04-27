import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

COLOR_LIST = ["xkcd:blue violet", "xkcd:shamrock", "xkcd:coral"]


class NHPoissonProcess:
    def __init__(self, mu, sig, xi, u, m):
        self.u = u
        self.m = m
        self.mu = mu
        self.sig = sig
        self.xi = xi

        # assert self.u >= self.mu
        if self.xi < 0:
            assert self.u < self.mu - self.sig / self.xi

    def get_parameters(self):
        """
        Returns the hyperparameters m and u and the parameters (mu, sig, xi).
        :return: u, m, mu, sig, xi
        """
        return self.u, self.m, self.mu, self.sig, self.xi

    def update_parameters(self, mu=None, sig=None, xi=None, u=None, m=None):
        """
        Update the values of (u, m, mu, sig, xi) with new ones given as inputs.
        If some parameters are not not specified, the actual one are maintained.
        :return: Nothing
        """
        if mu is not None:
            self.mu = mu
        if sig is not None:
            self.sig = sig
        if xi is not None:
            self.xi = xi
        if u is not None:
            self.u = u
        if m is not None:
            self.m = m

    def get_measure(self):
        """
        Compute the mean parameters of the random variable that gives the
        number of observations, which follows a Poisson distribution
        :return: the parameter lambda of the Poisson rv
        """
        if self.xi == 0:
            return self.m * np.exp(-(self.u - self.mu) / self.sig)
        else:
            return self.m * (1 + self.xi * (self.u - self.mu) / self.sig) ** (-1 / self.xi)

    def get_orthogonal_reparam(self):
        """
        Return the value of the orthogonal parameterization of the Poisson process
        :return: (r, nu, xi), with r the expected number of observations and nu the
                 orthogonal version of sig
        """
        r = self.get_measure()
        nu = (1 + self.xi) * (self.sig + self.xi * (self.u - self.mu))
        return r, nu, self.xi

    def gen_number_points(self, n_samples=1):
        """
        Generate n_samples samples of number of points observations.
        :param n_samples: an integer, 1 by default
        :return: The generated samples of numbers of observations
        """
        r = self.get_measure()
        return np.random.poisson(lam=r, size=n_samples)

    def gen_positions(self, n_obs):
        """
        Given a number of observed points n_obs, generates n_obs positions
        of the poisson process.
        :param n_obs: Number of observations
        :return: n_obs positions between u and the endpoint
        """
        scaled_sig = self.sig + self.xi * (self.u - self.mu)
        return st.genpareto.rvs(c=self.xi, loc=self.u, scale=scaled_sig, size=n_obs)

    def gen_time_events(self, n_obs):
        """
        Given a number of observed points n_obs, generates n_obs time of events
        of the poisson process.
        As the time dependance of the Poisson intensity is uniform (t2-t1)*Lambda(Iu),
        the time events are distributed uniformly
        :param n_obs: Number of observations
        :return: n_obs time events between 0 and m
        """
        return np.sort(np.random.rand(n_obs) * self.m)

    def plot_simulation(self, times, positions):
        """
        Plot the simulation of the NHPP and the QQ-plot
        :param times: The time of events, typically obtained with gen_time_events
        :param positions: The of events, typically obtained with gen_positions
        :return: Nothing, just plot
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.vlines(times, [0], [self.u], linestyles = "dashed", colors=COLOR_LIST[0], alpha=0.5)
        ax.vlines(times, [self.u], positions, colors=COLOR_LIST[0])
        
        ax.plot(times, positions, ".", alpha = 0.7, linewidth=1.5, ms = 10, color=COLOR_LIST[0])
        
        ax.hlines(self.u, 0.0, self.m, colors="r")
        ax.vlines(0, self.u / max(positions), 1, transform=ax.get_xaxis_transform(), colors="r")
        ax.vlines(self.m, self.u / max(positions), 1, transform=ax.get_xaxis_transform(), colors="r")
        
        ax.set_ylabel("Position", fontsize = 15)
        ax.set_xlabel("Time", fontsize = 15)

        ax.set_title("Number of generated points: " + str(len(positions)), fontsize=14)

    def get_param_m_update(self, m):
        """
        Return the value of mu and sigma corresponding to a scaling factor equal to m.
        :param m: the scaling factor on which we want to express the value of mu and sigma
        :return: mu_m and sigma_m
        """
        mu_m = self.mu - (self.sig/self.xi)(1 - (m/self.m)**(-self.xi))
        sigma_m = self.sig*(m/self.m)**(-self.xi)
        return mu_m, sigma_m

    def get_original_m_param(self, mu, sig, m):
        """
        Return the value of mu and sigma corresponding to a scaling factor equal to self.m,
        from given values of mu and sig and a given m
        :param mu: mu_m parameter
        :param sig: sigma_m parameter
        :param m: the scaling factor on which mu and sigma are linked to
        :return: mu_(self.m) and sigma_(self.m)
        """
        mu_2 = mu - (sig/self.xi)(1 - (self.m/m)**(-self.xi))
        sigma_2 = sig*(self.m/m)**(-self.xi)
        return mu_2, sigma_2

    def get_param_from_orthogonal(self, r, nu):
        """
        Return the value of mu sigma corresponding to given (r, nu) from
        the orthogonal parameterisation
        :param r: first parameter
        :param nu: second parameter
        :return: parameters mu and sigma
        """
        mu = self.u - (nu/(self.xi*(1+self.xi)))*(1 - (r/self.m) ** self.xi)
        sigma = (nu/(self.xi*(1+self.xi))) * (r/self.m) ** self.xi
        return mu, sigma


def sharkey_optimal_m(xi, n_obs):
    """
    Select a suitable value of M according to Sharkey and J. A. Tawn 2017.
    :return: a value of m
    """
    l = (xi + 1) * np.log((2 * xi + 3) / (2 * xi + 1))
    m1 = (1 + 2 * xi + l) / (3 + 2 * xi - l)
    m2 = (2 * xi ** 2 + 13 * xi + 8) / (2 * xi ** 2 + 9 * xi + 8)

    opt_M = round(n_obs * (m1 + m2) / 2)
    return opt_M

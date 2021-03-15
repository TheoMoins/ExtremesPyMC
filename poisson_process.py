import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# class genpareto_trunc(st.rv_continuous):
#     """
#     Distribution used for the density of the positions of the points given the
#     number of observations.
#     """
#     def _pdf(self, x, mu, sig, xi, u):
#         if xi == 0:
#             r = np.exp(-(u-mu)/sig)
#             pdf = (1/sig*r)*np.exp(-(x-mu)/sig)
#         else:
#             r = (1 + xi * (u - mu)/sig)**(-1/xi)
#             pdf = (1/sig*r)*(1 + xi * (x - mu)/sig)**(-1-1/xi)
#         return pdf
#
#     def _argcheck(self, mu, sig, xi, u):
#         return np.isfinite(xi)
#
#     def _get_support(self, mu, sig, xi, u):
#         a = u
#         b = np.inf
#         if xi < 0:
#             b = mu - sig/xi
#         return a, b
#
#
# pareto_trunc = genpareto_trunc(name='pareto_trunc')


class NHPoissonProcess:
    def __init__(self, mu, sig, xi, u, m):
        self.u = u
        self.m = m
        self.mu = mu
        self.sig = sig
        self.xi = xi

        # assert self.u >= self.mu
        if self.xi < 0:
            assert self.u < self.mu - self.sig/self.xi

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
            return self.m * (1 + self.xi * (self.u - self.mu)/self.sig)**(-1/self.xi)

    def get_orthogonal_reparam(self):
        """
        Return the value of the orthogonal parameterization of the Poisson process
        :return: (r, nu, xi), with r the expected number of observations and nu the
                 orthogonal version of sig
        """
        r = self.get_measure()
        nu = (1 + self.xi)*(self.sig + self.xi*(self.u - self.mu))
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
        scaled_sig = self.sig+self.xi*(self.u - self.mu)
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
        return np.sort(np.random.rand(n_obs)*self.m)

    def plot_simulation(self, times, positions):
        """
        Plot the simulation of the NHPP and the QQ-plot
        :param times: The time of events, typically obtained with gen_time_events
        :param positions: The of events, typically obtained with gen_positions
        :return: Nothing, just plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.vlines(times, [0], positions)
        ax.hlines(
            self.u,
            0.0,
            self.m,
            colors="r")
        ax.vlines(
            0,
            self.u / max(positions),
            1,
            transform=ax.get_xaxis_transform(),
            colors="r")

        ax.vlines(
            self.m,
            self.u / max(positions),
            1,
            transform=ax.get_xaxis_transform(),
            colors="r")

        ax.set(
            xlabel="Time",
            ylabel="Position")

        fig.suptitle("Simulation of the NHPP in $[0;m]$ x $[u ; +\infty[$ \n Number of generated points: "
                     + str(len(positions)), fontsize=14)

        plt.figure()
        QQplot = st.probplot(positions, dist=st.genpareto(c=self.xi,
                                                          loc=self.mu,
                                                          scale=self.sig + self.xi * (self.u - self.mu)),
                             fit=False, plot=plt)
        plt.show()


import pymc3 as pm
import theano.tensor as tt
import scipy as ss

from pymc3.distributions import draw_values, generate_samples
from src.priors import get_prior, need_potential, get_potential
from src.densities import gpd_logp, gpd_quantile
from src.poisson_process import sharkey_optimal_m

from src.densities import EPS

import logging

logger = logging.getLogger("pymc3")
# The different level of output to dismiss are:
# INFO, WARNING, ERROR, OR CRITICAL
logger.setLevel(logging.CRITICAL)


class PoissonMCMC:
    def __init__(self, priors, step_method, niter, obs, u, m, quantiles, orthogonal_param=True):

        self.priors = priors
        self.step_method = step_method
        self.niter = niter

        self.obs = obs
        self.n_obs = len(obs)
        self.u = u
        self.original_m = m
        self.m = m

        self.q1 = quantiles[0]
        self.q2 = quantiles[1]
        self.q3 = quantiles[2]

        self.orthogonal_param = orthogonal_param
        self.model = pm.Model()

    def run(self, verbose=False):
        with self.model:

            def get_step(text):
                if text[0:4] == "Metr":
                    return pm.Metropolis()
                elif text[0:4] == "NUTS":
                    return pm.NUTS()
                else:
                    print("Unknown step method given as input")

            if self.orthogonal_param:
                # PRIOR DEFINITION
                r = get_prior(self.priors[0], "r")
                nu = get_prior(self.priors[1], "nu")
                xi = get_prior(self.priors[2], "xi")
                if need_potential(self.priors):
                    prior = get_potential(self.priors, var=[r, nu, xi], orthogonal_param=True, u=self.u)

                # LIKELIHOOD DEFINITION (Poisson + GPD)
                n = pm.Poisson(name="n", mu=r, observed=self.n_obs)

                sig = nu / (1 + xi)

                def my_random_method(point=None, size=None):
                    sig_random, xi_random = draw_values([sig, xi], point=point, size=size)
                    return generate_samples(ss.stats.genpareto.rvs, c=xi_random, loc=self.u, scale=sig_random,
                                            size=size)

                gpd = pm.DensityDist('gpd', lambda value: gpd_logp(value=value, mu=self.u, sig=sig, xi=xi),
                                     observed=self.obs, random=my_random_method)

                if verbose:
                    print(tt.printing.Print("r")(r))
                    print(tt.printing.Print("nu")(nu))
                    print(tt.printing.Print("xi")(xi))
                    print(tt.printing.Print("prior")(prior))

            else:
                # PRIOR DEFINITION
                mu = get_prior(self.priors[0], "mu")
                sig = get_prior(self.priors[1], "sig")
                xi = get_prior(self.priors[2], "xi")
                if need_potential(self.priors):
                    prior = get_potential(self.priors, var=[mu, sig, xi], orthogonal_param=False, u=self.u)

                # LIKELIHOOD DEFINITION (Poisson + GPD)
                lam_value = self.m * tt.nnet.relu((1 + (xi + EPS) * (self.u - mu) / sig)) ** (-1 / (xi + EPS))
                lam = pm.Deterministic('lam', lam_value)

                n = pm.Poisson(name="n", mu=lam, observed=self.n_obs)

                sig_tilde = sig + xi * (self.u - mu)

                def my_random_method(point=None, size=None):
                    sig_random, xi_random = draw_values([sig_tilde, xi], point=point, size=size)
                    return generate_samples(ss.stats.genpareto.rvs, c=xi_random, loc=self.u, scale=sig_random,
                                            size=size)

                gpd = pm.DensityDist('gpd', lambda value: gpd_logp(value=value, mu=self.u, sig=sig_tilde, xi=xi),
                                     observed=self.obs, random=my_random_method)

                if verbose:
                    print(tt.printing.Print("mu")(mu))
                    print(tt.printing.Print("sig")(sig))
                    print(tt.printing.Print("xi")(xi))
                    print(tt.printing.Print("lam")(lam))

            if not self.orthogonal_param:
                if self.original_m != self.m:
                    mu_m = pm.Deterministic("mu_m",
                                            mu - (sig / (xi + EPS)) * (1 - (self.original_m / self.m) ** (-(xi + EPS))))
                    sig_m = pm.Deterministic("sig_m", sig * (self.original_m / self.m) ** (-xi))
                else:
                    mu_m = pm.Deterministic("mu_m", mu)
                    sig_m = pm.Deterministic("sig_m", sig)
            else:
                mu_m = pm.Deterministic("mu_m", self.u - (nu / ((xi + EPS) * (1 + xi))) * (
                            1 - (r / self.original_m) ** (xi + EPS)))
                sig_m = pm.Deterministic("sig_m", (nu / (1 + xi)) * (r / self.original_m) ** (xi + EPS))

            q1r = pm.Deterministic("q1r",
                                   gpd_quantile(prob=self.q1, mu=self.u, sig=sig_m + xi * (self.u - mu_m), xi=xi))
            q2r = pm.Deterministic("q2r",
                                   gpd_quantile(prob=self.q2, mu=self.u, sig=sig_m + xi * (self.u - mu_m), xi=xi))
            q3r = pm.Deterministic("q3r",
                                   gpd_quantile(prob=self.q3, mu=self.u, sig=sig_m + xi * (self.u - mu_m), xi=xi))

            if verbose:
                print(self.model.check_test_point())

            # STEP METHOD
            step = get_step(self.step_method)

            # SAMPLING
            trace = pm.sample(draws=self.niter,
                              step=step,
                              tune=1000,
                              discard_tuned_samples=False,
                              return_inferencedata=True,
                              progressbar=False)
        return trace

    def update_m(self, update_arg, xi=None):
        if update_arg == "n_obs":
            self.m = self.n_obs
        elif update_arg == "sharkey" and xi is not None:
            self.m = sharkey_optimal_m(xi=xi, n_obs=self.n_obs)
        else:
            print("The value of m given is not understood, try \"n_obs\" or \"sharkey\".")

    def prior_predictive_check(self, nsamples):
        with self.model:
            return pm.sample_prior_predictive(samples=nsamples)

    def posterior_predictive_check(self, trace):
        with self.model:
            return pm.sample_posterior_predictive(trace)


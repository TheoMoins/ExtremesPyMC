import pymc3 as pm
import theano.tensor as tt
import scipy as ss

from pymc3.distributions import draw_values, generate_samples
from src.priors import get_prior, need_potential, get_potential
from src.densities import gpd_logp, return_level
from src.poisson_process import sharkey_optimal_m

from src.densities import EPS

import logging

logger = logging.getLogger("pymc3")
# The different level of output to dismiss are:
# INFO, WARNING, ERROR, OR CRITICAL
logger.setLevel(logging.CRITICAL)

def get_step(text):
    if text[0:4] == "Metr":
        return pm.Metropolis()
    elif text[0:4] == "NUTS":
        return pm.NUTS()
    else:
        print("Unknown step method given as input")


class PoissonMCMC:
    def __init__(self, priors, step_method, niter, obs, u, m, period_range = None, orthogonal_param=True):

        self.priors = priors
        self.step_method = step_method
        self.niter = niter

        self.obs = obs
        self.n_obs = len(obs)
        self.u = u
        self.original_m = m
        self.m = m

        self.r_range = period_range

        self.orthogonal_param = orthogonal_param
        self.model = pm.Model()

    def define_prior(self):
        with self.model:

            if self.orthogonal_param:

                nu = get_prior(self.priors[0], "nu")
                xi = get_prior(self.priors[1], "xi")
                var_list = [nu, xi]
                if len(self.priors) == 3:
                    r = get_prior(self.priors[2], "r")
                    return r, nu, xi
                return nu, xi 
            
            else:

                sig = get_prior(self.priors[0], "sig")
                xi = get_prior(self.priors[1], "xi")
                if len(self.priors) == 3:
                    mu = get_prior(self.priors[0], "mu")
                    return mu, sig, xi
                return sig, xi 




    def run(self, verbose=False):
        with self.model:

            if self.orthogonal_param:
                # PRIOR DEFINITION
                if len(self.priors) == 3:
                    r, nu, xi = self.define_prior()
                else:
                    nu, xi = self.define_prior()
                
                var_list = [nu, xi]
                if len(self.priors) == 3:
                    var_list.append(r)
                if need_potential(self.priors):
                    prior = get_potential(self.priors, var=var_list, orthogonal_param=True, u=self.u)


                # LIKELIHOOD DEFINITION (Poisson + GPD)
                if len(self.priors) == 3:
                    n = pm.Poisson(name="n", mu=r, observed=self.n_obs)

                sig = nu / (1 + xi)

                def my_random_method(point=None, size=None):
                    sig_random, xi_random = draw_values([sig, xi], point=point, size=size)
                    return generate_samples(ss.stats.genpareto.rvs, c=xi_random, loc=self.u, scale=sig_random,
                                            size=size)

                gpd = pm.DensityDist('gpd', lambda value: gpd_logp(value=value, mu=self.u, sig=sig, xi=xi),
                                     observed=self.obs, random=my_random_method)

                if verbose:
                    # print(tt.printing.Print("r")(r))
                    print(tt.printing.Print("nu")(nu))
                    print(tt.printing.Print("xi")(xi))
                    print(tt.printing.Print("prior")(prior))

            else:
                # PRIOR DEFINITION
                if len(self.priors) == 3:
                    mu, sig, xi = self.define_prior()
                else:
                    sig, xi = self.define_prior()
                
                var_list = [sig, xi]
                if len(self.priors) == 3:
                    var_list.append(mu)
                if need_potential(self.priors):
                    prior = get_potential(self.priors, var=var_list, orthogonal_param=False, u=self.u)

                # LIKELIHOOD DEFINITION (Poisson + GPD)
                if len(self.priors) == 3:
                    lam_value = self.m * tt.nnet.relu((1 + (xi + EPS) * (self.u - mu) / sig)) ** (-1 / (xi + EPS))
                    lam = pm.Deterministic('lam', lam_value)

                    n = pm.Poisson(name="n", mu=lam, observed=self.n_obs)

                    sig_tilde = sig + xi * (self.u - mu)
                
                else: 
                    sig_tilde = sig

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
                if len(self.priors) == 3:
                    if self.original_m != self.m:
                        mu_m = pm.Deterministic("mu_m",
                                                mu - (sig / (xi + EPS)) * (1 - (self.original_m / self.m) ** (-(xi + EPS))))
                        sig_m = pm.Deterministic("sig_m", sig * (self.original_m / self.m) ** (-xi))
                    else:
                        mu_m = pm.Deterministic("mu_m", mu)
                        sig_m = pm.Deterministic("sig_m", sig)
                else:
                    mu_m = self.u
                    sig_m = pm.Deterministic("sig_m", sig)
            else:
                if len(self.priors) == 3:
                    mu_m = pm.Deterministic("mu_m", self.u - (nu / ((xi + EPS) * (1 + xi))) * (
                                1 - (r / self.original_m) ** (xi + EPS)))
                    sig_m = pm.Deterministic("sig_m", (nu / (1 + xi)) * (r / self.original_m) ** (xi + EPS))
                else:
                    mu_m = self.u
                    sig_m = pm.Deterministic("sig_m", (nu / (1 + xi)) )

            if self.r_range is not None:
                q = pm.Deterministic("q", return_level(self.r_range, mu=mu_m, sig=sig_m, xi=xi))

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


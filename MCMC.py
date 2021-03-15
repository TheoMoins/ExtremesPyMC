import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions.dist_math import bound

from priors import get_prior, need_potential, get_potential

EPS = 1e-20


class PoissonMCMC:
    def __init__(self, priors, step_method, niter, obs, u, m):

        self.priors = priors
        self.step_method = step_method
        self.niter = niter

        self.obs = obs
        self.n_obs = len(obs)
        self.u = u
        self.m = m

    def run(self):
        with pm.Model() as model:

            def get_step(text):
                if text[0:4] == "Metr":
                    return pm.Metropolis()
                elif text[0:4] == "NUTS":
                    return pm.NUTS()
                else:
                    print("Unknown step method given as input")
            # PRIOR DEFINITION
            mu = get_prior(self.priors[0], "mu")
            sig = get_prior(self.priors[1], "sig")
            xi = get_prior(self.priors[2], "xi")
            if need_potential(self.priors):
                prior = get_potential(self.priors, var=[mu, sig, xi])

            # LIKELIHOOD DEFINITION (Poisson + GPD)
            lam_value = self.m * tt.nnet.relu((1 + (xi + EPS) * (self.u - mu) / sig)) ** (-1 / (xi + EPS))
            lam = pm.Deterministic('lam', lam_value)

            n = pm.Poisson(name="n", mu=lam, observed=self.n_obs)

            def gpd_logp(value):
                sig_tilde = sig + xi * (self.u - mu)
                scaled = (value - self.u) / sig_tilde

                logp = -tt.log(sig_tilde) - ((xi + 1) / (xi + EPS)) * tt.log1p((xi + EPS) * scaled)
                alpha = tt.switch(tt.gt(xi, -1e-5),
                                  np.inf,
                                  mu - sig / xi)
                return bound(logp, value < alpha, value > self.u, xi != 0)

            gpd = pm.DensityDist('gpd', gpd_logp, observed=self.obs)

            # print(tt.printing.Print("mu")(mu))
            # print(tt.printing.Print("sig")(sig))
            # print(tt.printing.Print("xi")(xi))
            # print(tt.printing.Print("lam")(lam))
            # print(model.check_test_point())

            # STEP METHOD
            step = get_step(self.step_method)

            # SAMPLING
            trace = pm.sample(self.niter, step, return_inferencedata=False)

        print(pm.summary(trace, kind="stats"))
        return trace
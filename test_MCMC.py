import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
import arviz as az
import theano.tensor as tt
from pymc3.distributions.dist_math import bound

from poisson_process import NH_Poisson_process

EPS = 1e-20

# Observation

mu_obs = 80
sig_obs = 15
xi_obs = 0.05
u = 30
m = 1

PP = NH_Poisson_process(mu=mu_obs, sig=sig_obs, xi=xi_obs, u=u, m=m)

lam_obs = PP.get_measure()
n_obs = PP.gen_number_points()[0]

# PP.update_parameters(m=n_obs)

print("Expected number of points:", lam_obs)
print("Number of generated points:", n_obs)
obs = PP.gen_positions(n_obs=n_obs)
print(obs)
print("empirical mean:", np.mean(obs))
print("min:", np.min(obs))
print("max:", np.max(obs))

niter = 5000
# Model
with pm.Model() as model:
    sig = pm.HalfNormal(name='sig', sigma=10)
    xi = pm.Normal(name='xi', mu=0, sigma=5, testval=0.1)
    # xi = pm.HalfNormal(name='xi', sigma=1)
    # xi = np.array(0)

    mu = pm.Flat(name="mu")
    # lower_bound = tt.switch(tt.gt(xi, -1e-5),
    #                         np.array(-1e8), np.max(obs) + sig / (xi+EPS))
    # lower = pm.Deterministic("lower", lower_bound)
    # Bound_flat = pm.Bound(pm.Flat, lower=lower_bound)
    # mu = Bound_flat(name='mu', testval=u)

    def gpd_logp(value):
        sig_tilde = sig + xi * (u - mu)
        scaled = (value - u) / sig_tilde

        # logp = tt.switch(tt.lt(tt.abs_(xi), 1e-5),
        #                  -tt.log(sig_tilde) - scaled,
        #                  -tt.log(sig_tilde) - ((xi + 1) / (xi+EPS)) * tt.log1p((xi+EPS) * scaled))
        logp = -tt.log(sig_tilde) - ((xi + 1) / (xi+EPS)) * tt.log1p((xi+EPS) * scaled)
        alpha = tt.switch(tt.gt(xi, -1e-5),
                          np.inf,
                          mu - sig / xi)
        return bound(logp, value < alpha, value > u, xi != 0)


    # lam_value = tt.switch(tt.lt(tt.abs_(xi), 1e-5),
    #                       m * tt.exp((u - mu) / sig),
    #                       m * tt.nnet.relu((1 + (xi+EPS) * (u - mu) / sig)) ** (-1 / (xi+EPS)))
    lam_value = m * tt.nnet.relu((1 + (xi+EPS) * (u - mu) / sig)) ** (-1 / (xi+EPS))
    lam = pm.Deterministic('lam', lam_value)

    # print(tt.printing.Print("mu")(mu))
    # print(tt.printing.Print("sig")(sig))
    # print(tt.printing.Print("xi")(xi))
    # print(tt.printing.Print("lam")(lam))

    n = pm.Poisson(name="n", mu=lam, observed=n_obs)
    gpd = pm.DensityDist('gpd', gpd_logp, observed=obs)
    print(model.check_test_point())
    step = pm.Metropolis()
    # step = pm.NUTS()
    trace = pm.sample(niter, step, return_inferencedata=False)

print(pm.summary(trace, kind="stats"))

az.plot_trace(trace, var_names=["mu", "sig", "xi"])
az.plot_posterior(data=trace, var_names=["mu", "sig", "xi"])
# az.plot_autocorr(data=trace, var_names=["mu", "sig", "xi"])
plt.show()

import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
import arviz as az
import theano.tensor as tt
from pymc3.distributions.dist_math import bound

from poisson_process import NH_Poisson_process

# Observation

mu_obs = 25
sig_obs = 5
xi_obs = 0.2
u = 25
m = 80


# Lam_obs = (1 + xi_obs*(u - mu_obs)/sigma_obs)**(-1/xi_obs)
# obs = np.random.poisson(Lam_obs)

PP = NH_Poisson_process(mu=mu_obs, sig=sig_obs, xi=xi_obs, u=u, m=m)

lam_obs = PP.get_measure()
n_obs = PP.gen_number_points()[0]
print("Expected number of points:", lam_obs)
print("Number of generated points:", n_obs)
obs = PP.gen_positions(n_obs=n_obs)
print(obs)

#plt.figure()
#plt.plot(obs)
#plt.show()


niter = 10000
# Model
with pm.Model() as model:
    mu = pm.Flat(name='mu')
    sig = pm.HalfFlat(name='sig')
    xi = pm.HalfFlat(name='xi')

    def gpd_logp(value):
        scaled = (value - mu) / sig
        if xi != 0:
            logp = -(tt.log1p(sig) + ((xi + 1) / xi) * tt.log1p(xi * scaled))
        else:
            logp = tt.log1p(sig)*scaled
        alpha = mu - sig / xi
        bounds = tt.switch(xi > 0, value > mu, value < alpha)
        return bound(logp, bounds, value > mu)

    # Expected value
    lam = pm.Deterministic('lam', m * (1 + xi * (u - mu)/sig)**(-1/xi))
    n = pm.Poisson(mu=lam, name='n', observed=n_obs)
    # step = pm.Metropolis()
    gpd = pm.DensityDist('gpd', gpd_logp, observed=obs)
    trace = pm.sample(niter, return_inferencedata=False)


az.plot_trace(trace, var_names=["mu", "sig", "xi"])
az.plot_posterior(data=trace, var_names=["mu", "sig", "xi"])
az.plot_autocorr(data=trace, var_names=["mu", "sig", "xi"])
plt.show()

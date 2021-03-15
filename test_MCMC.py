import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt

from poisson_process import NHPoissonProcess
from MCMC import PoissonMCMC
from visualisation import plot_trace


# Data simulation

# Tony configuration
# mu_obs = 25
# sig_obs = 5
# xi_obs = 0.2
# u = 20
# m = 40

# Sharkey configuration
mu_obs = 80
sig_obs = 15
xi_obs = 0.05
u = 30
m = 1

PP = NHPoissonProcess(mu=mu_obs, sig=sig_obs, xi=xi_obs, u=u, m=m)

lam_obs = PP.get_measure()
n_obs = PP.gen_number_points()[0]

PP.update_parameters(m=n_obs)

print("Expected number of points: {:.2f}".format(lam_obs))
print("Number of generated points:", n_obs)
obs = PP.gen_positions(n_obs=n_obs)
print(obs)
times = PP.gen_time_events(n_obs=n_obs)

print("Min: {:.3f}".format(np.min(obs)))
print("Max: {:.3f}".format(np.max(obs)))

# PP.plot_simulation(times=times, positions=obs)

# Prior resp. for mu, sigma and xi:
# priors = ["Flat({})".format(u),
#           "HalfFlat()",
#           "Flat()"]
priors = ["Jeffreys({})".format(u),
          "Jeffreys()",
          "Jeffreys()"]


# Sampling method:
step_method = "NUTS"

# Number of iteration:
niter = 5000

MCMC = PoissonMCMC(priors=priors, step_method=step_method, niter=niter,
                   obs=obs, u=u, m=m)
trace = MCMC.run()
plot_trace(trace)

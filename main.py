import numpy as np
import os
import matplotlib.pyplot as plt
import arviz as az

from poisson_process import NHPoissonProcess
from mcmc import PoissonMCMC
from visualisation import plot_trace, plot_densities, plot_autocorr, plot_ess
from parameters.params import Params
from densities import gpd_quantile

# Parameters loading

poisson_params_directory = "parameters/poisson_simu/"

poisson_config = "sharkey_poisson_config"
# poisson_config = "tony_poisson_config"
# poisson_config = "negative_xi_config"

poisson_params = Params(poisson_params_directory + poisson_config + ".json")

mcmc_params_directory = "parameters/mcmc/"
mcmc_configs = ["Config5.json"]
# mcmc_configs = []
# for filename in os.listdir(mcmc_params_directory):
#     if "json" in filename:
#         mcmc_configs.append(filename)

# Data simulation
PP = NHPoissonProcess(mu=poisson_params.mu, sig=poisson_params.sigma, xi=poisson_params.xi,
                      u=poisson_params.u, m=poisson_params.m)

lam_obs = PP.get_measure()
n_obs = PP.gen_number_points()[0]

obs = PP.gen_positions(n_obs=n_obs)
times = PP.gen_time_events(n_obs=n_obs)

pp_params = PP.get_parameters()
print("Poisson process parameter: (mu = {}, sigma = {}, xi = {})".format(pp_params[2],
                                                                         pp_params[3],
                                                                         pp_params[4]))
ortho_params = PP.get_orthogonal_reparam()
print("Orthogonal version: (r = {:.2f}, nu = {:.2f}, xi = {})".format(ortho_params[0],
                                                                      ortho_params[1],
                                                                      ortho_params[2]))
print("Expected number of points: {:.2f}".format(lam_obs))
print("Number of generated points:", n_obs)
print("Min: {:.3f}".format(np.min(obs)))
print("Max: {:.3f}".format(np.max(obs)))

quantiles = (1/lam_obs, 1/(2*lam_obs), 1/(3*lam_obs))
print("Estimation of quantiles 1/{}, 1/{} and 1/{} ".format(int(lam_obs), 2*int(lam_obs), 3*int(lam_obs)))
sig_tilde = pp_params[3]+pp_params[4]*(pp_params[0]-pp_params[2])
real_q1 = gpd_quantile(prob=quantiles[0], mu=pp_params[0], sig=sig_tilde, xi=pp_params[4])
real_q2 = gpd_quantile(prob=quantiles[1], mu=pp_params[0], sig=sig_tilde, xi=pp_params[4])
real_q3 = gpd_quantile(prob=quantiles[2], mu=pp_params[0], sig=sig_tilde, xi=pp_params[4])
print("Theoretical values of quantiles: {:.3f}, {:.3f}, {:.3f}".format(real_q1, real_q2, real_q3))

PP.plot_simulation(times=times, positions=obs)

# MCMC

traces = []
traces_orthogonal = []

names = []
names_orthogonal = []

for filename in mcmc_configs:
    print("\nConfig file: ", filename)
    mcmc_params = Params(mcmc_params_directory + filename)
    print(mcmc_params.name)
    print("")

    priors = [mcmc_params.priors["p1"],
              mcmc_params.priors["p2"],
              mcmc_params.priors["p3"]]

    if mcmc_params.init_p1_by_u:
        init_val = n_obs if mcmc_params.orthogonal_param else poisson_params.u
        priors[0] = priors[0].replace("u", str(init_val))

    MCMC = PoissonMCMC(priors=priors, step_method=mcmc_params.step_method, niter=mcmc_params.niter,
                       obs=obs, u=poisson_params.u, m=poisson_params.m, quantiles=quantiles,
                       orthogonal_param=mcmc_params.orthogonal_param)
    if mcmc_params.update_m != "":
        MCMC.update_m(update_arg=mcmc_params.update_m, xi=poisson_params.xi)
    print("Choice of m = {} for MCMC".format(MCMC.m))

    trace = MCMC.run(verbose=False)

    names.append(mcmc_params.name)
    traces.append(trace)

    print("\nRank normalized R-hat:")
    print(az.rhat(data=trace))

    print("\nRank normalized ESS:")
    print(az.ess(data=trace))

    print("\n Summary:")
    print(az.summary(data=trace, var_names=["mu_m", "sig_m", "xi"]))

    print("\n Quantiles Summary:")
    print(az.summary(data=trace, var_names=["q1r", "q2r", "q3r"]))

    # if MCMC.orthogonal_param:
    #     plot_trace(trace, var_names=["r", "nu", "xi"], title=mcmc_params.name)
    #     names_orthogonal.append(mcmc_params.name)
    #     traces_orthogonal.append(trace)

    plot_trace(trace, var_names=["mu_m", "sig_m", "xi"], title=mcmc_params.name, real_value=pp_params[2:5])
    plot_trace(trace, var_names=["q1r", "q2r", "q3r"], title=mcmc_params.name,
               real_value=[real_q1, real_q2, real_q3])

plot_autocorr(traces=traces, labels=names, var_names=["mu_m", "sig_m", "xi"])
plot_autocorr(traces=traces, labels=names, var_names=["q1r", "q2r", "q3r"])
plot_ess(traces=traces, labels=names, var_names=["mu_m", "sig_m", "xi"])
plot_ess(traces=traces, labels=names, var_names=["q1r", "q2r", "q3r"])
# plot_densities(traces=traces, labels=names, var_names=["mu_m", "sig_m", "xi"])

if traces_orthogonal:
    plot_autocorr(traces=traces_orthogonal, labels=names_orthogonal, var_names=["r", "nu", "xi"])
    plot_ess(traces=traces_orthogonal, labels=names_orthogonal, var_names=["r", "nu", "xi"])
    # plot_densities(traces=traces_orthogonal, labels=names_orthogonal, var_names=["r", "nu", "xi"])

# figs = [plt.figure(n) for n in plt.get_fignums()]
# for i, fig in enumerate(figs):
#     fig.savefig("Figures/{}/Figure{}.pdf".format(poisson_config, i + 1))
# plt.show()
print("Done!")

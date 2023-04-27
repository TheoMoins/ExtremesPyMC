import numpy as np
import os
import matplotlib.pyplot as plt
import arviz as az

from src.poisson_process import NHPoissonProcess
from src.mcmc import PoissonMCMC
from src.visualisation import *
from parameters.params import Params
from src.densities import gpd_quantile
from src.rhat_infinity import *

# Parameters loading

poisson_params_directory = "parameters/poisson_simu/"

# poisson_config_list = ["positive_xi_", "null_xi_", "negative_xi_"]
# poisson_config_list = ["positive_xi_"]
poisson_config_list = ["gpd_"]

mcmc_params_directory = "parameters/mcmc/"

# mcmc_configs = ["Config1a", "Config1b", "Config1c", "Config1d"]
# mcmc_configs = ["Config2a", "Config2b", "Config2c", "Config2d"]
# mcmc_configs = ["Config4", "Config5",  "Config6"]
mcmc_configs = ["GPDConfig1", "GPDConfig2"]

# mcmc_configs = []
# for filename in os.listdir(mcmc_params_directory):
#     if "json" in filename:
#         mcmc_configs.append(filename)

for poisson_config in poisson_config_list:
    poisson_params = Params(poisson_params_directory + poisson_config + ".json")

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

    PP.plot_simulation(times=times, positions=obs)

    # MCMC

    traces = []
    traces_orthogonal = []

    names = []
    names_orthogonal = []

    for filename in mcmc_configs:
        print("\nConfig file: ", filename)
        mcmc_params = Params(mcmc_params_directory + filename + ".json")
        print(mcmc_params.name)

        nb_dim = len(mcmc_params.priors)
        priors = [mcmc_params.priors["p1"],
                  mcmc_params.priors["p2"]]

        if nb_dim == 3:
            priors.append(mcmc_params.priors["p3"])

            if mcmc_params.init_p1_by_u:
                init_val = n_obs if mcmc_params.orthogonal_param else poisson_params.u
                priors[2] = priors[2].replace("u", str(init_val))

        MCMC = PoissonMCMC(priors=priors, step_method=mcmc_params.step_method, niter=mcmc_params.niter,
                           obs=obs, u=poisson_params.u, m=poisson_params.m,
                           orthogonal_param=mcmc_params.orthogonal_param)
        if mcmc_params.update_m != "":
            MCMC.update_m(update_arg=mcmc_params.update_m, xi=poisson_params.xi)
        print("Choice of m = {} for MCMC".format(MCMC.m))

        trace = MCMC.run(verbose=False)

        names.append(mcmc_params.name)
        traces.append(trace)

        if nb_dim == 3:
            real_values = pp_params[2:5]
            var_names = ["mu_m", "sig_m", "xi"]
        else:
            real_values = pp_params[3:5]
            var_names = ["sig_m", "xi"]
                
        # plot_trace(trace, var_names = var_names, title = mcmc_params.name, real_value = real_values)

    plot_autocorr(traces=traces, labels=names, var_names = var_names)
    plot_ess(traces=traces, labels=names, var_names = var_names)

    if "Config1c" in mcmc_configs or "Config2c" in mcmc_configs and poisson_config == "positive_xi_":
        plot_r_hat_x(traces=traces, labels=names, var_names = var_names, ymax = 1.032)
    else:
        plot_r_hat_x(traces=traces, labels=names, var_names = var_names)

figs = [plt.figure(n) for n in plt.get_fignums()]
for i, fig in enumerate(figs):
    config_idx = int(i//(len(figs)/len(poisson_config_list)))
    fig.savefig("Figures/{}{}_Figure{}.pdf".format(poisson_config_list[config_idx], filename, i + 1), bbox_inches="tight")
    # plt.show()
print("Done!")

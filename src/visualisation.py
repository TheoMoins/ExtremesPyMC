import numpy as np
import matplotlib.pylab as plt
import pymc3 as pm
import arviz as az

from src.rhat_infinity import *

COLOR_LIST = ["xkcd:coral", "xkcd:blue violet", "xkcd:shamrock"]


def plot_trace(trace, var_names, title, real_value=None):

    chain_prop = {"color": ['lightseagreen', 'lightsalmon', 'goldenrod', 'yellowgreen']}
    # hist_kwargs = {"bw": "scott"}
    # plot_kwargs = {"asdfcswflpha": 0.5}

    if real_value is not None:
        lines = [(var_names[0], {}, [real_value[0]]),
                 (var_names[1], {}, [real_value[1]])]
        if len(var_names)==3:
            lines.append((var_names[2], {}, [real_value[2]]))
        az.plot_trace(trace, var_names=var_names, chain_prop=chain_prop, combined=True,
                      lines=lines, rug=False, figsize=(20,12))
    else:
        az.plot_trace(trace, var_names=var_names, chain_prop=chain_prop, combined=False,
                      rug=False, figsize=(20,12))
    plt.suptitle(title, fontsize=16)


# def plot_autocorr_(traces, var_names, labels):
#     nb_traces = len(traces)
#     if nb_traces == 1:
#         az.plot_autocorr(traces[0], var_names=var_names, combined=True)
#         plt.suptitle(labels[0], fontsize=16)
#     else:
#         _, ax = plt.subplots(nb_traces, len(var_names), figsize=(20, nb_traces * 6))
#         for i, trace in enumerate(traces):
#             for j in range(len(var_names)):
#                 az.plot_autocorr(trace, var_names=var_names[j], ax=ax[i, j], combined=True)
#                 ax[i, j].set_title(var_names[j] + " : " + labels[i])

#         plt.suptitle("Autocorrelations", fontsize=18)


def plot_autocorr(traces, var_names, labels, max_lag=50):
    _, ax = plt.subplots(1, len(var_names), figsize=(7*len(var_names), 5))

    for j, var_name in enumerate(var_names):
        #ax[j].set_ylim(-0.1,1.05)
        #ax[j].set_xlim(0,max_lag)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].xaxis.set_tick_params(labelsize=14)
        ax[j].yaxis.set_tick_params(labelsize=14)
        for i, trace in enumerate(traces):
            x_array = np.asarray(trace.posterior[var_name]).flatten()
            y = az.autocorr(x_array)
            ax[j].plot(np.arange(0, max_lag), y[0:max_lag], ".--", 
                    alpha = 0.7, linewidth=1, ms = 10, color=COLOR_LIST[i])
            
    ax[0].set_ylabel("Autocorrelations", fontsize = 15)
    ax[1].set_xlabel("Lag", fontsize = 15)
    plt.legend(labels, loc=(0.5,0.5), fontsize=14)



# def plot_ess(traces, var_names, labels):
#     nb_traces = len(traces)
#     if nb_traces == 1:
#         az.plot_ess(traces[0], var_names=var_names, kind="evolution")
#         plt.suptitle(labels[0], fontsize=16)
#     else:
#         _, ax = plt.subplots(nb_traces, len(var_names), figsize=(20, nb_traces * 6))
#         for i, trace in enumerate(traces):
#             for j in range(len(var_names)):
#                 az.plot_ess(trace, var_names=var_names[j], ax=ax[i, j], kind="evolution")
#                 ax[i, j].set_title(var_names[j] + " : " + labels[i])

#         plt.suptitle("ESS", fontsize=16)

def plot_ess(traces, var_names, labels, nb_points = 50):
    chain_length = traces[0].posterior[var_names[0]].shape[1]
    chains_idx = np.linspace(0, chain_length, nb_points, dtype=int)[1:]

    _, ax = plt.subplots(1, len(var_names), figsize=(7*len(var_names), 5))

    for j, var_name in enumerate(var_names):
        #ax[j].set_ylim(-0.1,1.05)
        #ax[j].set_xlim(0,chain_length)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].xaxis.set_tick_params(labelsize=14)
        ax[j].yaxis.set_tick_params(labelsize=14)
        ax[j].axhline(y=400, ls = ":", color="grey", lw=2)
        # ax[j].set_xlabel(var_list_latex[j], fontsize = 15)
        for i, trace in enumerate(traces):
            ess_list = []
            for n in chains_idx:
                ess_list.append(float(az.ess(trace.posterior[var_name][:,0:n], method="mean")[var_name]))
            ax[j].plot(chains_idx, ess_list, ".--", 
                    alpha = 0.7, linewidth=1, ms = 10, color=COLOR_LIST[i])
            
    ax[0].set_ylabel("ESS", fontsize = 15)
    ax[1].set_xlabel("Number of draws", fontsize = 15)
    plt.legend(labels, loc=(0.05,0.8), fontsize=14)


def plot_densities(traces, labels, var_names):
    az.plot_density(data=traces,
                    data_labels=labels,
                    var_names=var_names,
                    hdi_prob=0.99,
                    bw="scott",
                    shade=0.1)
    plt.suptitle("Posterior distributions", fontsize=18)


def plot_r_hat_x(traces, var_names, labels, ymax = None):
    _, ax = plt.subplots(1, len(var_names), figsize=(7*len(var_names), 5))

    for j, var_name in enumerate(var_names):
        if ymax is not None:
            ax[j].set_ylim(0.999,ymax)
        #ax[j].set_xlim(0,chain_length)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].xaxis.set_tick_params(labelsize=14)
        ax[j].yaxis.set_tick_params(labelsize=14)

        for i, trace in enumerate(traces):
            chains = np.asarray(trace.posterior[var_name]).transpose(1, 0)
            grid, r_val = r_x_values(chains)
            ax[j].plot(grid, r_val, "-", 
                    alpha = 0.7, linewidth=2, color=COLOR_LIST[i])
        ax[j].axhline(y=get_threshold(2 * len(traces[0].posterior.chain)), ls = ":", color="grey", lw=3)

    ax[0].set_ylabel(r"$\hat R (x)$", fontsize = 15)
    ax[1].set_xlabel(r"$x$", fontsize = 15)
    plt.legend(labels, loc=(-0.7,0.8), fontsize=14)

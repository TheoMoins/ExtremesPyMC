import numpy as np
import matplotlib.pylab as plt
import pymc3 as pm
import arviz as az
from sympy import var

from src.rhat_infinity import *

COLOR_LIST = ["xkcd:coral", "xkcd:shamrock", "xkcd:blue violet"]


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


def var_name_to_latex(var):
    if var == "sig_m":
        var = "sigma"
    if var == "mu_m":
        var = "mu"
    return r"${}$".format("\\"+var)

def plot_autocorr(traces, var_names, labels, max_lag=50, plot_legend=True):
    _, ax = plt.subplots(1, len(var_names), figsize=(7*len(var_names), 5),
                         sharex=True, sharey=True)

    for j, var_name in enumerate(var_names):
        #ax[j].set_ylim(-0.1,1.05)
        #ax[j].set_xlim(0,max_lag)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].xaxis.set_tick_params(labelsize=18)
        ax[j].yaxis.set_tick_params(labelsize=18)
        ax[j].set_title(var_name_to_latex(var_name), fontsize = 17)
        for i, trace in enumerate(traces):
            x_array = np.asarray(trace.posterior[var_name]).flatten()
            y = az.autocorr(x_array)
            ax[j].plot(np.arange(0, max_lag), y[0:max_lag], ".--", 
                    alpha = 0.7, linewidth=1, ms = 10, color=COLOR_LIST[i])
            
    ax[0].set_ylabel("Autocorrelations", fontsize = 17)
    # ax[0].set_xlabel("Lag", fontsize = 17)
    ax[1].set_xlabel("Lag", fontsize = 17)
    if plot_legend:
        plt.legend(labels, loc=(0.5,0.5), fontsize=16)


def plot_ess(traces, var_names, labels, nb_points = 50, plot_legend=False):
    chain_length = traces[0].posterior[var_names[0]].shape[1]
    chains_idx = np.linspace(0, chain_length, nb_points, dtype=int)[1:]

    _, ax = plt.subplots(1, len(var_names), figsize=(7*len(var_names), 5),
                         sharex=True, sharey=True)

    for j, var_name in enumerate(var_names):
        #ax[j].set_ylim(-0.1,1.05)
        #ax[j].set_xlim(0,chain_length)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].xaxis.set_tick_params(labelsize=18)
        ax[j].yaxis.set_tick_params(labelsize=18)
        # ax[j].set_xlabel(var_list_latex[j], fontsize = 15)
        for i, trace in enumerate(traces):
            ess_list = []
            for n in chains_idx:
                ess_list.append(float(az.ess(trace.posterior[var_name][:,0:n], method="mean")[var_name]))
            ax[j].plot(chains_idx, ess_list, ".--", 
                    alpha = 0.7, linewidth=1, ms = 10, color=COLOR_LIST[i])
        ax[j].axhline(y=400, ls = ":", color="grey", lw=3)   
    ax[0].set_ylabel("ESS", fontsize = 17)
    # ax[0].set_xlabel("Number of draws", fontsize = 17)
    ax[1].set_xlabel("Number of draws", fontsize = 17)
    if plot_legend:
        plt.legend(labels, loc=(0.05,0.8), fontsize=16)


def plot_densities(traces, labels, var_names):
    az.plot_density(data=traces,
                    data_labels=labels,
                    var_names=var_names,
                    hdi_prob=0.99,
                    bw="scott",
                    shade=0.1)
    plt.suptitle("Posterior distributions", fontsize=18)


def plot_r_hat_x(traces, var_names, labels, ymax = None, plot_legend=False):
    _, ax = plt.subplots(1, len(var_names), figsize=(7*len(var_names), 5),
                         sharex=False, sharey=True)

    for j, var_name in enumerate(var_names):
        if ymax is not None:
            ax[j].set_ylim(0.995,ymax)
        #ax[j].set_xlim(0,chain_length)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].xaxis.set_tick_params(labelsize=18)
        ax[j].yaxis.set_tick_params(labelsize=18)

        for i, trace in enumerate(traces):
            chains = np.asarray(trace.posterior[var_name]).transpose(1, 0)
            grid, r_val = r_x_values(chains)
            ax[j].plot(grid, r_val, "-", 
                    alpha = 0.7, linewidth=2, color=COLOR_LIST[i])
        ax[j].axhline(y=get_threshold(2 * len(traces[0].posterior.chain)), ls = ":", color="grey", lw=3)

    ax[0].set_ylabel(r"$\hat R (x)$", fontsize = 17)
    # ax[0].set_xlabel(r"$x$", fontsize = 17)
    ax[1].set_xlabel(r"$x$", fontsize = 17)
    if plot_legend:
        plt.legend(labels, loc=(-0.7,0.8), fontsize=16)

import numpy as np
import matplotlib.pylab as plt
import pymc3 as pm
import arviz as az

from src.rhat_infinity import *


def plot_trace(trace, var_names, title, real_value=None):

    chain_prop = {"color": ['lightseagreen', 'lightsalmon', 'goldenrod', 'yellowgreen']}
    # hist_kwargs = {"bw": "scott"}
    # plot_kwargs = {"asdfcswflpha": 0.5}

    if real_value is not None:
        lines = [(var_names[0], {}, [real_value[0]]),
                 (var_names[1], {}, [real_value[1]]),
                 (var_names[2], {}, [real_value[2]])]
        az.plot_trace(trace, var_names=var_names, chain_prop=chain_prop, combined=False,
                      lines=lines, rug=False, figsize=(20,12))
    else:
        az.plot_trace(trace, var_names=var_names, chain_prop=chain_prop, combined=False,
                      rug=False, figsize=(20,12))
    plt.suptitle(title, fontsize=16)


def plot_autocorr(traces, var_names, labels):
    nb_traces = len(traces)
    if nb_traces == 1:
        az.plot_autocorr(traces[0], var_names=var_names, combined=True)
        plt.suptitle(labels[0], fontsize=16)
    else:
        _, ax = plt.subplots(nb_traces, 3, figsize=(20, nb_traces * 6))
        for i, trace in enumerate(traces):
            for j in range(3):
                az.plot_autocorr(trace, var_names=var_names[j], ax=ax[i, j], combined=True)
                ax[i, j].set_title(var_names[j] + " : " + labels[i])

        plt.suptitle("Autocorrelations", fontsize=18)


def plot_ess(traces, var_names, labels):
    nb_traces = len(traces)
    if nb_traces == 1:
        az.plot_ess(traces[0], var_names=var_names, kind="evolution")
        plt.suptitle(labels[0], fontsize=16)
    else:
        _, ax = plt.subplots(nb_traces, 3, figsize=(20, nb_traces * 6))
        for i, trace in enumerate(traces):
            for j in range(3):
                az.plot_ess(trace, var_names=var_names[j], ax=ax[i, j], kind="evolution")
                ax[i, j].set_title(var_names[j] + " : " + labels[i])

        plt.suptitle("ESS", fontsize=16)


def plot_densities(traces, labels, var_names):
    az.plot_density(data=traces,
                    data_labels=labels,
                    var_names=var_names,
                    hdi_prob=0.99,
                    bw="scott",
                    shade=0.1)
    plt.suptitle("Posterior distributions", fontsize=18)


def plot_r_hat_x(traces, var_names, labels):
    nb_traces = len(traces)
    if nb_traces == 1:
        chains = np.asarray(traces[0].posterior[var_names[0]]).transpose(1, 0)
        grid, r_val = r_x_values(chains)
        plt.plot(grid, r_val)
        plt.plot(grid, [get_threshold(2*len(traces[0].posterior.chain))]*len(grid), linestyle="--")
        plt.title(labels[0], fontsize=16)
    else:
        _, ax = plt.subplots(nb_traces, 3, figsize=(20, nb_traces * 6))
        for i, trace in enumerate(traces):
            for j in range(3):
                chains = np.asarray(traces[i].posterior[var_names[j]]).transpose(1, 0)
                grid, r_val = r_x_values(chains)
                ax[i, j].plot(grid, r_val)
                ax[i, j].plot(grid, [get_threshold(2 * len(trace.posterior.chain))] * len(grid), linestyle="--")

                ax[i, j].xaxis.set_tick_params(labelsize=15)
                ax[i, j].yaxis.set_tick_params(labelsize=15)
                ax[i, j].set_title(var_names[j] + " : " + labels[i])

        plt.suptitle("ESS", fontsize=18)


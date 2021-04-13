import numpy as np
import matplotlib.pylab as plt
import pymc3 as pm
import arviz as az


def plot_trace(trace, var_names, title, real_value=None):
    chain_prop = {"color": ['lightseagreen', 'lightsalmon', 'goldenrod', 'yellowgreen']}
    if real_value is not None:
        lines = [(var_names[0], {}, [real_value[0]]),
                 (var_names[1], {}, [real_value[1]]),
                 (var_names[2], {}, [real_value[2]])]
        az.plot_trace(trace, var_names=var_names, chain_prop=chain_prop, combined=True,
                      lines=lines, rug=True)
    else:
        az.plot_trace(trace, var_names=var_names, chain_prop=chain_prop, combined=True, rug=True)
    plt.suptitle(title, fontsize=16)


def plot_autocorr(traces, var_names, labels):
    nb_traces = len(traces)
    if nb_traces == 1:
        az.plot_autocorr(traces[0], var_names=var_names, combined=True)
        plt.suptitle(labels[0], fontsize=16)
    else:
        _, ax = plt.subplots(nb_traces, 3, figsize=(20, nb_traces*6))
        for i, trace in enumerate(traces):
            for j in range(3):
                az.plot_autocorr(trace, var_names=var_names[j], ax=ax[i, j], combined=True)
                ax[i, j].set_title(var_names[j] + " : " + labels[i])

        plt.suptitle("Autocorrelations", fontsize=16)


def plot_ess(traces, var_names, labels):
    nb_traces = len(traces)
    if nb_traces == 1:
        az.plot_ess(traces[0], var_names=var_names, kind="evolution")
        plt.suptitle(labels[0], fontsize=16)
    else:
        _, ax = plt.subplots(nb_traces, 3, figsize=(20, nb_traces*6))
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
                    shade=0.1)
    plt.suptitle("Posterior distributions", fontsize=18)


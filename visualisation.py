import numpy as np
import matplotlib.pylab as plt
import arviz as az


def plot_trace(trace):
    az.plot_trace(trace, var_names=["mu", "sig", "xi"])
    az.plot_posterior(data=trace, var_names=["mu", "sig", "xi"])
    az.plot_autocorr(data=trace, var_names=["mu", "sig", "xi"])
    plt.show()


def plot_ess(trace):
    az.plot_ess(idata=trace, var_names=["mu", "sig", "xi"], kind="local")
    az.plot_ess(idata=trace, var_names=["mu", "sig", "xi"], kind="quantile")
    az.plot_ess(idata=trace, var_names=["mu", "sig", "xi"], kind="evolution")
    plt.show()

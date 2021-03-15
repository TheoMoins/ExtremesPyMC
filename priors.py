import numpy as np
import pymc3 as pm
import theano.tensor as tt

EPS = 1e-20


def get_prior(text, variable):
    params = text[text.find('(') + 1:text.find(')')]
    if params != "":
        params = [float(s) for s in params.split(',')]
    # Flat prior over the whole real line:
    if text[0:4] == "Flat" or ("Jeffreys" in text and variable == "mu"):
        if params != "":
            return pm.Flat(name=variable, testval=params[0])
        else:
            return pm.Flat(name=variable)
    # Flat prior over the real positive line:
    elif "HalfFlat" in text or ("Jeffreys" in text and variable == "sig"):
        if params != "":
            return pm.HalfFlat(name=variable, testval=params[0])
        else:
            return pm.HalfFlat(name=variable)
    # Truncated normal prior for real positive values:
    elif "HalfNormal" in text:
        if len(params) > 1:
            return pm.HalfNormal(name=variable, sigma=params[0], testval=params[1])
        else:
            return pm.HalfNormal(name=variable, sigma=params[0])
    # Normal prior:
    elif "Norm" in text:
        if len(params) == 3:
            return pm.Normal(name=variable, mu=params[0], sigma=params[1], testval=params[2])
        else:
            return pm.Normal(name=variable, mu=params[0], sigma=params[1])
    # Flat prior in [-1/2 ; +oo[:
    elif "Jeffreys" in text and variable == "xi":
        TruncFlat = pm.Bound(pm.Flat, lower=-0.5)
        if params != "":
            return TruncFlat(name="xi", testval=params[0])
        else:
            return TruncFlat(name="xi")
    else:
        print("Unknown prior given as input")


def jeffreys_logp(mu, sig, xi, u):
    logp = (-1 - 3 / (2 * xi + EPS)) * tt.log((1 + (xi + EPS) * (u - mu) / sig))
    logp += -2 * tt.log(sig)
    logp += - tt.log(1 + xi) - 0.5 * tt.log(1 + 2 * xi + EPS)
    return logp


def need_potential(priors_name):
    for p in priors_name:
        if "HalfFlat" in p:
            return True
    return False


def get_potential(priors_name, var, u=None):
    logp = 0
    for i, p in enumerate(priors_name):
        if "HalfFlat" in p:
            logp += -tt.log(var[i])
    if "Jeffreys" in priors_name[0]:
        if u is None:
            print("u is required to compute Jeffreys prior")
        logp = jeffreys_logp(var[0], var[1], var[2], u)
    return pm.Potential("prior", logp)

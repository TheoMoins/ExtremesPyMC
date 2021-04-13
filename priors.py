import numpy as np
import pymc3 as pm
import theano.tensor as tt
from densities import jeffreys_logp, jeffreys_orthogonal_logp

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
    # Half flat prior, flat over the real positive line:
    elif "HalfFlat" in text or ("Jeffreys" in text and variable in ["sig", "r", "nu"]):
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


def need_potential(priors_name):
    for p in priors_name:
        if "HalfFlat" in p or "Jeffreys" in p:
            return True
    return False


def get_potential(priors_name, var, orthogonal_param=True, u=None):
    logp = 0
    for i, p in enumerate(priors_name):
        if "HalfFlat" in p:
            logp += -tt.log(var[i])
    if "Jeffreys" in priors_name[0]:
        if orthogonal_param:
            logp = jeffreys_orthogonal_logp(var[0], var[1], var[2])
        else:
            if u is None:
                print("u is required to compute Jeffreys prior")
            logp = jeffreys_logp(var[0], var[1], var[2], u)
    return pm.Potential("prior", logp)

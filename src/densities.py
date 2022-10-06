import tty
import numpy as np
import pymc3 as pm
import theano.tensor as tt

from pymc3.distributions.dist_math import bound

EPS = 1e-10


def jeffreys_logp(mu, sig, xi, u):
    logp = (-1 - 3 / (2 * xi + EPS)) * tt.log((1 + (xi + EPS) * (u - mu) / sig))
    logp += -2 * tt.log(sig)
    logp += - tt.log(1 + xi) - 0.5 * tt.log(1 + 2 * xi + EPS)
    return logp


def jeffreys_orthogonal_logp(r, nu, xi):
    logp = 0.5 * tt.log(r)
    logp += -tt.log(nu) - tt.log(1 + xi) - 0.5 * tt.log(1 + 2 * xi + EPS)
    return logp

def jeffreys_gpd_logp(sig, xi):
    return - tt.log(sig) - tt.log(1 + xi) - 0.5 * tt.log(1 + 2 * xi + EPS)


def jeffreys_orthogonal_gpd_logp(nu, xi):
    return -tt.log(nu) - tt.log(1 + xi) - 0.5 * tt.log(1 + 2 * xi + EPS)

def gpd_logp(value, mu, sig, xi):
    scaled = (value - mu) / sig

    logp = -tt.log(sig) - ((xi + 1) / (xi + EPS)) * tt.log1p((xi + EPS) * scaled)
    alpha = tt.switch(tt.gt(xi, -1e-5),
                      np.inf,
                      mu - sig / xi)
    return bound(logp, value < alpha, value > mu, xi != 0)


def gpd_quantile(prob, mu, sig, xi):
    return mu + (sig/(xi+EPS))*(prob**(-(xi+EPS))-1)

def return_level(r, mu, sig, xi):
    return mu + (sig/(xi+EPS))*((-tt.log(1-1/r))**(-(xi+EPS))-1)

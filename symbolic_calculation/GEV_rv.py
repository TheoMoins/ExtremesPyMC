import numpy as np

from sympy import *
from sympy.stats import *


class GEV_Variable:
    def __init__(self):
        self.x = Symbol("x")
        self.mu = Symbol("mu")
        self.sig = Symbol("sigma")
        self.gam = Symbol("gamma")

    def eval_params(self, Mu=None, Sig=None, Gam=None):
        if Mu is not None:
            self.mu = Mu
        if Sig is not None:
            self.sig = Sig
        if Gam is not None:
            self.gam = Gam

    def reparameterize(self):
        self.mu = Symbol("mu")
        self.sig = Symbol("sigma")
        self.gam = Symbol("gamma")

    def get_pdf(self):
        if (type(self.gam) is int) and Eq(self.gam, 0):
            return (1 / self.sig) * exp(-(self.x - self.mu) / self.sig) * exp(-exp(-(self.x - self.mu) / self.sig))
        else:
            return Piecewise(
                ((1 / self.sig) * exp(-(self.x - self.mu) / self.sig) * exp(-exp(-(self.x - self.mu) / self.sig)),
                 Eq(self.gam, 0)),
                ((1 / self.sig) * ((1 + self.gam * (self.x - self.mu) / self.sig) ** (-1 / self.gam - 1))
                 * exp(-(1 + self.gam * (self.x - self.mu) / self.sig) ** (-1 / self.gam)),
                 self.gam != 0)
            )

    def get_cdf(self):
        if (type(self.gam) is int) and Eq(self.gam, 0):
            return exp(- exp(-(self.x - self.mu) / self.sig))
        else:
            return Piecewise(
                (exp(- exp(-(self.x - self.mu) / self.sig)), Eq(self.gam, 0)),
                (exp(- (1 + self.gam * (self.x - self.mu) / self.sig) ** (-1 / self.gam)), self.gam != 0))

    def eval_pdf(self, val):
        return self.get_pdf().subs(self.x, val)

    def eval_cdf(self, val):
        return self.get_cdf().subs(self.x, val)

    def def_set(self):
        if (type(self.gam) is int) and Eq(self.gam, 0):
            return Interval(-oo, oo)
        else:
            return Piecewise((Interval(-oo, self.mu - self.sig / self.gam), self.gam < 0),
                             (Interval(-oo, oo), Eq(self.gam, 0)),
                             (Interval(self.mu - self.sig / self.gam, oo), True)
                             )

    def sympy_rv(self):
        return ContinuousRV(self.x, self.get_pdf(), set=self.def_set())

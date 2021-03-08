import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from poisson_process import NH_Poisson_process

mu = 25
sig = 5
xi = 0.2
u = 25
m = 80

PP = NH_Poisson_process(mu=mu, sig=sig, xi=xi, u=u, m=m)

n = PP.gen_number_points()[0]
print("Expected number of points:", PP.get_measure())
print("Number of generated points:", n)

positions = PP.gen_positions(n_obs=n)
times = PP.gen_time_events(n_obs=n)

fig, ax = plt.subplots(figsize=(10, 6))

ax.vlines(times, [0], positions)
ax.hlines(
    u,
    0.0,
    m,
    colors="r")
ax.vlines(
    0,
    u/max(positions),
    1,
    transform=ax.get_xaxis_transform(),
    colors="r")

ax.vlines(
    m,
    u/max(positions),
    1,
    transform=ax.get_xaxis_transform(),
    colors="r")

ax.set(
    xlabel="Time",
    ylabel="Position")

fig.suptitle("Simulation of the NHPP in $[0;m]$ x $[u ; +\infty[$ \n Number of generated points: " + str(n), fontsize=14)


plt.figure()
QQplot = st.probplot(positions, dist=st.genpareto(c=xi, loc=mu, scale=sig+xi*(u-mu)), fit=False, plot=plt)
plt.show()


import numpy as np
from phy4910 import move_photon
import matplotlib.pyplot as plt

rng = np.random.default_rng()
n = 1000
nr_ar = np.zeros(n)
ns_ar = np.zeros(n)

for i in range (n):
    x, y, z, theta, phi, nr, ns = move_photon(tau_max = 5, log = False, rng = rng)
    nr_ar[i] = nr
    ns_ar[i] = ns

#print(nr_ar)

avg = np.mean(nr_ar)
print(avg/(avg+1))

avg_ns = np.mean(ns_ar)
dev_ns = np.std(ns_ar)

print(f"2b photons that make it out of the surface take {avg_ns} steps on average and the standard deviation is {dev_ns}")
    

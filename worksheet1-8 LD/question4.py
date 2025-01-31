import numpy as np
from phy4910 import move_photon
import matplotlib.pyplot as plt

rng = np.random.default_rng()
n = 100000
theta_ar = np.zeros(n)
phi_ar = np.zeros(n)

for i in range (n):
    x, y, z, theta, phi, nr, ns = move_photon(tau_max = 5, log = False, rng = rng)
    theta_ar[i] = theta
    phi_ar[i] = phi
np.savetxt("angle data.txt",np.column_stack((theta_ar,phi_ar)))



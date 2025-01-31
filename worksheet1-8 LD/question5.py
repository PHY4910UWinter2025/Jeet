import numpy as np
import matplotlib.pyplot as plt

thetas, phis = np.loadtxt("angle data.txt", unpack=True)

mus = np.cos(thetas)

#plt.plot(mus, ',')
#plt.show()

# bin the mus
N = len(mus)
nbin = 20
I = np.zeros(nbin)

mu_max = np.max(mus) + 1e-6
mu_min = np.min(mus)

for i in range(N):
    b = int( (mus[i] - mu_min) / (mu_max - mu_min) * nbin)
    I[b] += 1
    
#plt.plot(I)
#plt.show()

bin_size = (mu_max - mu_min) / nbin
mu_bin = np.arange(mu_min, mu_max, bin_size) + 0.5 * bin_size

# normalize
for i in range(nbin):
    I0 = 2 * mu_bin[i] * N / nbin
    I[i] /= I0
    
plt.plot(mu_bin, I, ".")
plt.plot(mu_bin, 0.5 + 0.75 * mu_bin)
plt.show()

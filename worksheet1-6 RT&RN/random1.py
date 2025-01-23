import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

N = 1000000
rnums = rng.normal(size = N)

print(rnums)

#plt.plot(rnums)

#plt.show()

# Data binning

Nbin = 50
bins = np.zeros(Nbin)

data_max = np.max(rnums) + 1e-10 
data_min = np.min(rnums)

for i in range(N):
	b = int((rnums[i] - data_min)/ (data_max) - (data_min) * Nbin)
	bins[b] += 1.0
	
plt.histogram(range(Nbin), bins)
plt.show()

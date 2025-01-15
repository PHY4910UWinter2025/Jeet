import numpy as np

def ode_euler(x_start, x_end, h, y0, z0, f, g):

	x = np.arange(x_start, x_end, h)
	N = len(x)
	y = np.zeros(N)
	z = np.zeros(N)

	y[0] = 1
	z[0] = 1

	for i in arange(N-1):
		k1 = h * f(x[i], y[i], z[i])
		ell1 = h * g(x[i], y[i], z[i])
		
		y[i+1] = y[i] + k1
		z[i+1] = z[i] + ell1
		
	return x, y, z

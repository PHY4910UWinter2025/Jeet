import numpy as np

def ode_euler(x_start, x_end, h, y0, z0, f, g):

	x = np.arange(x_start, x_end, h)
	N = len(x)
	y = np.zeros(N)
	z = np.zeros(N)

	y[0] = 1
	z[0] = 1

	for i in range(N-1):
		k1 = h * f(x[i], y[i], z[i])
		ell1 = h * g(x[i], y[i], z[i])
		
		y[i+1] = y[i] + k1
		z[i+1] = z[i] + ell1
		
	return x, y, z
	
def ode_rk4(x_start, x_end, h, y0, z0, f, g):

	x = np.arange(x_start, x_end, h)
	N = len(x)
	y = np.zeros(N)
	z = np.zeros(N)

	y[0] = y0
	z[0] = z0

	for i in range(N-1):
		k1 = h * f(x[i], y[i], z[i])
		ell1 = h * g(x[i], y[i], z[i])
		
		k2 = h * f(x[i] + 0.5*h, y[i] + 0.5*k1, z[i] + 0.5* ell1)
		ell2 = h * g(x[i] + 0.5*h, y[i] + 0.5*k1, z[i] + 0.5* ell1)
		
		k3 = h * f(x[i] + 0.5*h, y[i] + 0.5*k2, z[i] + 0.5* ell2)
		ell3 = h * g(x[i] + 0.5*h, y[i] + 0.5*k2, z[i] + 0.5* ell2)
		
		k4 = h * f(x[i] + 0.5*h, y[i] + 0.5*k3, z[i] + 0.5* ell3)
		ell4 = h * g(x[i] + 0.5*h, y[i] + 0.5*k3, z[i] + 0.5* ell3)
		
		y[i+1] = y[i] + k1/6 + k2/3 + k3/3 +k4/6
		z[i+1] = z[i] + ell1/6 + ell2/3 + ell3/3 + ell4/6
		
	return x, y, z

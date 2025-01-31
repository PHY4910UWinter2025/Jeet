import numpy as np
import matplotlib.pyplot as plt

def ode_euler(x_start, x_end, h, y0, z0, f, g):

	x = np.arange(x_start, x_end, h)
	N = len(x)
	y = np.zeros(N)
	z = np.zeros(N)

	y[0] = 1
	z[0] = 1

	for i in range(0, N-1):
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

	for i in range(0, N-1):
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
	

def pick_direction(rng):

	theta = np.arccos(1 - 2 * rng.random())
	phi =  2*np.pi * rng.random()
	
	return theta, phi
	
def pick_optical_depth(rng):

	x = rng.random()
	return -np.log(1-x)
	
	
def move_photon(tau_max = 10, z_max = 1, rng = np.random.default_rng(), log = True):

	x_list = [0]
	y_list = [0]
	z_list = [0]
	
	n_scatter = 0
	n_restart = 0
	while True:
	
		theta, phi = pick_direction(rng)
		tau = pick_optical_depth(rng)
		
		s = tau/tau_max
		
		x = x_list[-1] + s * np.sin(theta) * np.cos(phi)
		y = y_list[-1] + s * np.sin(theta) * np.sin(phi)
		z = z_list[-1] + s * np.cos(theta)
		
		x_list.append(x)
		y_list.append(y)
		z_list.append(z)
		
		n_scatter += 1
		
		if log: print(f"Scatter {n_scatter}: Photon at {x:.3f}, {y:.3f}, {z:.3f}")
		
		if z < 0:
			# start again
			x_list = [0]
			y_list = [0]
			z_list = [0]
			n_scatter = 0
			n_restart += 1
			
		if z > z_max:
			if log: print(f"All done!")
			return x_list, y_list, z_list, theta, phi, n_restart, n_scatter



if __name__ == '__main__':

	# test!
	
    rng = np.random.default_rng()

    x, y, z, theta, phi, nr, ns = move_photon()
    fig  = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
        
    ax.plot(x, y, z, '-')
    plt.show()


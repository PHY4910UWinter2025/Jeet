import numpy as np
from phy4910 import ode_euler

#solve the ode via euler

def f(x, y, z):
	return z
	
def g(x, y, z):
	return -1/y * z**2
	
x, y , z = ode_euler(0, 10, 0.001, 1, 1, f, g)
	
	
np.savetxt("euler_data.txt", np.column_stack((x,y,z)))



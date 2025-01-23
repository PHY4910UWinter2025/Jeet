import matplotlib.pyplot as plt
import numpy as np
from phy4910 import ode_euler, ode_rk4

# solve the ode via euler

def f(x, y, z):
	return z
	
def g(x, y, z):
	return -1/y * z**2

x1, y1, z1 = ode_euler(0, 10, 0.001, 1, 1, f, g)

x2, y2, z2 = ode_rk4(0, 10, 0.001, 1, 1, f, g)

plt.plot(x1, y1)

plt.plot(x2, y2)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import phy4910


def f(x, y, z):
	return z
	
def g(x, y, z):
	return -1/y * z**2

x, y, z = phy4910.ode_rk4(0, 1, 0.001, 1, 12.012, f, g)

plt.plot(x, y)


plt.show()

print(y[-1])

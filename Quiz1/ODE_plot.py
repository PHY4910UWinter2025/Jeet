import matplotlib.pyplot as plt
import numpy as np
from phy4910 import ode_euler

# solve the ode via euler

def f(x, y, z):
	return z
	
def g(x, y, z):
	return -z + 6*y

x, y, z = ode_euler(0, 1, 0.001, 1, -2, f, g)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.savefig("quiz1plot.pdf")

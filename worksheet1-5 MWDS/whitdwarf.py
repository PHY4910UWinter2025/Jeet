import numpy as np
import matplotlib.pyplot as plt
from phy4910 import ode_rk4

# A. Let's build a nonrelativistic model of a white dwarf!

#
#  Part 1 - Solve Lane-Emden
#

# polytropic index of the model
n = 1.5

# these two functions define the Lane-Emden differential equation
def f(eta, varrho, z):
	return z
	
def g(eta, varrho, z):
	return -varrho**n - 2*z/eta

# solve the Lane-Emden equation
eta, varrho, z = ode_rk4(0.0001, 4, 0.0001, 1, 0, f, g)

# uh oh, we get some NaNs thanks to varrho going negative.  Remove them:
positivevalues = varrho > 0
eta = eta[positivevalues]
varrho = varrho[positivevalues]
z = z[positivevalues]

plt.plot(eta, varrho)
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\varrho$")
plt.show()

# find the surface in "scaled radius"
eta_s = eta[-1]

print(f"The surface is at {eta_s:.3f}")

#
# Part 2 - Calculating the dimensionless mass m
#

m = np.trapz(varrho**n * eta**2, eta)

print(f"The dimensionless mass is {m:.3f}")

#
# Part 3 - Converting to real units
#

# Part a

# some physical numbers, in cgs units
# we can change rho_c to make different mass white dwarfs
rho_c = 4.045E6
k_nr = 3.166E12
G = 6.6743E-8
M_sun = 1.989E33

# calculate the radial scale factor
lam = np.sqrt((n+1) * k_nr * rho_c**((1-n)/n)/4/np.pi/G) 

print(f"The radial scale factor is {lam/1e5:.3f} km")

# Part b

# create arrays for the physical radius and density
r = lam * eta 
rho = rho_c * varrho**n 

plt.plot(r/1e5, rho)
plt.xlabel(r"$r$ (km)")
plt.ylabel(r"$\rho$ (g/cm$^3$)")
plt.show()

print(f"The radius of the white dwarf is {r[-1]/1e5:.3f} km")

# Part c

m_dwarf = 4*np.pi*rho_c*lam**3*m

print(f"The mass of the white dwarf is {m_dwarf/M_sun:.3f} solar masses")








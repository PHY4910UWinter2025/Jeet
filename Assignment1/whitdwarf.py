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
    if eta == 0:  # Avoid division by zero
        return -varrho**n
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






import numpy as np
import matplotlib.pyplot as plt
from phy4910 import ode_rk4

# Define functions for A(varrho) and B(varrho)
def A(varrho):
    return (-5/9) * varrho**(-4/3) * (1 + varrho**(2/3))**(-1/2) - (2/3) * varrho**(-2/3) * (1 + varrho**(2/3))**(-3/2) + (1/3) * (1 + varrho**(2/3))**(-5/2)

def B(varrho):
    return (5/3) * varrho**(-1/3) * (1 + varrho**(2/3))**(-1/2) - (1/3) * varrho**(1/3) * (1 + varrho**(2/3))**(-3/2)

# Define system of ODEs
def f(eta, varrho, z):
    return z

def g(eta, varrho, z):
    if eta == 0:  # Avoid division by zero
        return -varrho**1.5
    return -varrho**1.5 - (2/eta) * z

def solve_white_dwarf(varrho_c, eta_max=20, deta=0.01):
    eta_vals = []
    varrho_vals = []
    
    varrho = varrho_c  # Initial central density
    z = 0  # Initial slope
    eta = 0
    
    while varrho > 0.001 * varrho_c and eta < eta_max:
        eta_vals.append(eta)
        varrho_vals.append(varrho)
        
        eta_step, varrho_step, z_step = ode_rk4(eta, eta + deta, deta, varrho, z, f, g)
        varrho, z = varrho_step[-1], z_step[-1]
        
        eta += deta
    
    return np.array(eta_vals), np.array(varrho_vals)

# Compute mass-radius relationship
n = 1.5
k_nr = 3.166E12
G = 6.6743E-8
M_sun = 1.989E33

rho_c_values = np.logspace(5, 9, 25)  # 25 values from 10^5 to 10^9 g/cm³
radii = []
masses = []

for rho_c in rho_c_values:
    eta_vals, varrho_vals = solve_white_dwarf(1)
    eta_s = eta_vals[-1]  # Surface radius
    m = np.trapz(varrho_vals**n * eta_vals**2, eta_vals)
    
    lam = np.sqrt((n+1) * k_nr * rho_c**((1-n)/n) / (4 * np.pi * G))
    R = lam * eta_s
    M = 4 * np.pi * rho_c * lam**3 * m
    
    radii.append(R / 1e5)  # Convert to km
    masses.append(M / M_sun)  # Convert to solar masses

# Plot mass-radius relationship
plt.figure(figsize=(8,6))
plt.plot(radii, masses, marker='o', linestyle='-', label='Theoretical Mass-Radius')
plt.xlabel("Radius (km)")
plt.ylabel("Mass (Solar Masses)")
plt.title("White Dwarf Mass-Radius Relationship")
plt.legend()
plt.grid()
plt.show()




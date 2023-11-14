import numpy as np
import matplotlib.pyplot as plt

##########################################################
##########################################################

filename = "pattern1.txt"
dims = 60,80 #theta, phi

##########################################################
##########################################################
# Generation of the sphre grid

theta, dTheta = np.linspace(0, np.pi, dims[0], retstep=True)
phi, dPhi = np.linspace(0,2*np.pi, dims[1], endpoint=False, retstep=True)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

sphere_grid=(theta_grid, phi_grid, dTheta, dPhi)



##########################################################
# Loading the values of the file to compute the F field

# i_theta = np.loadtxt(filename, usecols=0).reshape(dims)
# i_phi = np.loadtxt(filename, usecols=1).reshape(dims)

Freal = np.loadtxt(filename, usecols=2).reshape(dims)
Fimag = np.loadtxt(filename, usecols=3).reshape(dims)

F = Freal+1j*Fimag



##########################################################
# Radiation pattern

sphere_F = SphericalShape(np.abs(F), *sphere_grid)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sphere_F.plot_surface(fig, ax, title="Radiation pattern", colorbar=True)
plt.show()



##########################################################
# Directivity

D = 4*np.pi*np.power(np.abs(F),2)/sphere_F.integrate(1)

sphere_D = SphericalShape(D, *sphere_grid)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sphere_D.plot_surface(fig, ax, title="Directivity", colorbar=True)
plt.show()

Dmax=np.max(D)
print(f"D_max = {Dmax:.2f} = {10*np.log10(Dmax):.2f} [dB]")
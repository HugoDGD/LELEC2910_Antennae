from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
from shapebis import *

phi_grid = loadmat("Metasurface patterns/phi_grid_matrix.mat")["phi_grid_matrix"]
theta_grid = loadmat("Metasurface patterns/Theta_grid_matrix.mat")["Theta_grid_matrix"]
dTheta,dPhi = np.deg2rad(4), np.deg2rad(5)

sphere_grid = (theta_grid, phi_grid, dTheta, dPhi)

TE_grid = loadmat("Metasurface patterns/S12_grid_matrix_TE.mat")["S12_grid_matrix_TE"]
TM_grid = loadmat("Metasurface patterns/S12_grid_matrix_TM.mat")["S12_grid_matrix_TM"]

R = np.abs(TE_grid+TM_grid) #Norm of the radiated field

########################################################################################################
#TE and TM components

sphere_TE = SphericalShape(np.abs(TE_grid), *sphere_grid)
sphere_TM = SphericalShape(np.abs(TM_grid), *sphere_grid)

fig, axes = plt.subplots(1,2, subplot_kw={"projection": "3d"})
sphere_TE.plot_surface(fig, axes[0], title="TE component", colorbar=True)
sphere_TM.plot_surface(fig, axes[1], title="TM component", colorbar=True)



########################################################################################################
#Total directivity

sphere_R = SphericalShape(R,*sphere_grid)
rad_power = sphere_R.integrate(1)

D=4*np.pi*np.divide(np.power(R,2),rad_power)

sphere_D = SphericalShape(D, *sphere_grid)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sphere_D.plot_surface(fig, ax, title="Directivity", colorbar=True)



########################################################################################################
#LHCP and RHCP components

LHCP_grid = 1/np.sqrt(2)*(TM_grid-1j*TE_grid)
RHCP_grid = 1/np.sqrt(2)*(TM_grid+1j*TE_grid)

sphere_LHCP = SphericalShape(np.abs(LHCP_grid), *sphere_grid)
sphere_RHCP = SphericalShape(np.abs(RHCP_grid), *sphere_grid)

fig, axes = plt.subplots(1,2, subplot_kw={"projection": "3d"})
sphere_LHCP.plot_surface(fig, axes[0], title="LHCP component", colorbar=True)
sphere_RHCP.plot_surface(fig, axes[1], title="RHCP component", colorbar=True)

index_max = np.argmax(D)
LHCP_max=np.abs(LHCP_grid).flatten()[index_max]
RHCP_max=np.abs(RHCP_grid).flatten()[index_max]

print(f"LHCP:{LHCP_max:.3e}, RHCP:{RHCP_max:.3e}")



########################################################################################################
#LHCP and RHCP directivity

LHCP_D_grid = 4*np.pi*np.power(np.abs(LHCP_grid),2)/sphere_LHCP.integrate(1)
RHCP_D_grid = 4*np.pi*np.power(np.abs(RHCP_grid),2)/sphere_RHCP.integrate(1)

sphere_LHCP_D = SphericalShape(LHCP_D_grid, theta_grid,phi_grid,dTheta, dPhi)
sphere_RHCP_D = SphericalShape(RHCP_D_grid, theta_grid,phi_grid,dTheta, dPhi)

fig, axes = plt.subplots(1,2, subplot_kw={"projection": "3d"})
sphere_LHCP_D.plot_surface(fig,axes[0], title="LHCP directivity", colorbar=True)
sphere_RHCP_D.plot_surface(fig,axes[1], title="RHCP directivity", colorbar=True)

plt.show()
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
#Total directivity

sphere_R = SphericalShape(R,*sphere_grid)
rad_power = sphere_R.integrate(1)

D=4*np.pi*np.divide(np.power(R,2),rad_power)

sphere_D = SphericalShape(D, *sphere_grid)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sphere_D.plot_surface(fig, ax, title="Directivity", colorbar=True)



########################################################################################################
#TE and TM components

sphere_TE = SphericalShape(np.abs(TE_grid), *sphere_grid)
sphere_TM = SphericalShape(np.abs(TM_grid), *sphere_grid)

fig, axes = plt.subplots(1,2, subplot_kw={"projection": "3d"})
sphere_TE.plot_surface(fig, axes[0], title="TE component", colorbar=True)
sphere_TM.plot_surface(fig, axes[1], title="TM component", colorbar=True)



########################################################################################################
#TE and TM directivities

TE_D_grid = 4*np.pi*np.power(np.abs(TE_grid),2)/rad_power
TM_D_grid = 4*np.pi*np.power(np.abs(TM_grid),2)/rad_power

sphere_TE_D = SphericalShape(np.abs(TE_D_grid), *sphere_grid)
sphere_TM_D = SphericalShape(np.abs(TM_D_grid), *sphere_grid)

fig, axes = plt.subplots(1,2, subplot_kw={"projection": "3d"})
sphere_TE_D.plot_surface(fig, axes[0], title="TE directivity", colorbar=True)
sphere_TM_D.plot_surface(fig, axes[1], title="TM directivity", colorbar=True)



########################################################################################################
#Question 3
theta_30 = 30//2
phi_90 = 90//5

theta = theta_grid[theta_30, phi_90]
phi = phi_grid[theta_30, phi_90]

norm = np.sqrt(np.power(np.abs(TE_grid[theta_30, phi_90]),2)+np.power(np.abs(TM_grid[theta_30, phi_90]),2))

print(f"Directivity at theta={np.rad2deg(theta)}° and phi={np.rad2deg(phi)}°:\n\
    D = {D[theta_30, phi_90]}\n\
    |TE|/norm = {np.abs(TE_grid[theta_30, phi_90])/norm}; |TM|/norm = {np.abs(TM_grid[theta_30, phi_90])/norm}")

########################################################################################################
#LHCP and RHCP components
LHCP_grid = 1/np.sqrt(2)*(TE_grid-1j*TM_grid)
RHCP_grid = 1/np.sqrt(2)*(TE_grid+1j*TM_grid)

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

LHCP_D_grid = 4*np.pi*np.power(np.abs(LHCP_grid),2)/rad_power
RHCP_D_grid = 4*np.pi*np.power(np.abs(RHCP_grid),2)/rad_power

sphere_LHCP_D = SphericalShape(LHCP_D_grid, theta_grid,phi_grid,dTheta, dPhi)
sphere_RHCP_D = SphericalShape(RHCP_D_grid, theta_grid,phi_grid,dTheta, dPhi)

fig, axes = plt.subplots(1,2, subplot_kw={"projection": "3d"})
sphere_LHCP_D.plot_surface(fig,axes[0], title="Cross-pol directivity")
sphere_RHCP_D.plot_surface(fig,axes[1], title="Co-pol directivity")

plt.show()



########################################################################################################
#Direction cosine plane


def plot_uv_plane(fig, ax, X, Y , Z, zmin=0, zmax=1,
                    title="", title_size=20,
                    xlabel="x", ylabel="y", label_size=15,
                    cmap="magma", cbar_label="", cbar_label_size=15, cbar_ticks_size=15):

    surf = ax.contourf(X,Y, Z, cmap=cmap, vmin=zmin, vmax=zmax)
    ax.set_aspect("equal")

    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_title(title, fontsize=title_size)

    cbar = fig.colorbar(surf, location="left")
    cbar.ax.set_ylabel(cbar_label, fontsize=cbar_label_size)
    cbar.ax.tick_params("y", labelsize=cbar_ticks_size)

LHCP_D_dB = 10*np.log(LHCP_D_grid)
RHCP_D_dB = 10*np.log(RHCP_D_grid)

min = min(np.min(LHCP_D_dB), np.min(RHCP_D_dB))
max = max(np.max(LHCP_D_dB), np.max(RHCP_D_dB))

print(min, max)

fig, axes = plt.subplots(1,2)
X,Y,_ = sphere2cartesian(1,theta_grid, phi_grid)
plot_uv_plane(fig, axes[0], X,Y, LHCP_D_dB, zmin=min, zmax=max, title="Cross-pol directivity", xlabel="u", ylabel="v", cbar_label="[dB]")
plot_uv_plane(fig, axes[1], X,Y, RHCP_D_dB, zmin=min, zmax=max, title="Co-pol directivity", xlabel="u", ylabel="v", cbar_label="[dB]")
plt.show()
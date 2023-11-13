import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

# def sphere2cartesian(R, theta,phi):
#     X=R*np.sin(theta)*np.cos(phi).T
#     Y=R*np.sin(theta)*np.sin(phi).T
#     Z=R*np.cos(theta)

#     return np.asarray((X,Y,Z), float)

def sphere2cartesian(R,theta,phi):
    """
        Compute the X,Y,Z matrices from R, theta, phi, element wise

        pre: R, theta, phi must be matrices with the same shape
        post: return a tuple containing the 3 matrices X,Y,Z
    """
    X = np.multiply(R,np.multiply(np.sin(theta),np.cos(phi)))
    Y = np.multiply(R,np.multiply(np.sin(theta),np.sin(phi)))
    Z = np.multiply(R,np.cos(theta))

    return (X,Y,Z)

def cbar_ticklabels_formating(labels: np.ndarray, cbar_ticks_format:str):
    return [f"{label:{cbar_ticks_format}}" for label in labels]

def min_max_scale(a:np.ndarray):
    return (a-np.min(a))/(np.max(a)-np.min(a))

def scale(a:np.ndarray, min, max):
    return (a-min)/(max-min)


class Shape:
    def __init__(self, points:np.ndarray, dS:np.ndarray=None):
        """
            Create a shape from its points.

            points: a tuple (X,Y,Z) with (X[index],Y[index],Z[index]) the cartesian coordinates of the point at index.
                    These can be either arrays or matrices/grid.

            dS: at each point, represents the integration area of this point
        """

        self.points=points
        self.dS = dS

    def integrate(self, f_values:np.ndarray):
        """Integrate a function on the surface of the shape.
            f_values:   the values of the function to integrate at each points of the shape (=self.points). f_values.shape must be the same as self.points.shape

            return a single value wich is equal to sum(f_values*dS) with dS the integration surfaces of the shape
        """
        if self.dS is None:
            raise AttributeError("The integration areas dS of the shape are not defined")

        return np.sum(np.multiply(f_values,self.dS))
    
#################################################################################################################

class SphericalShape(Shape):
    def __init__(self, R_grid:np.ndarray, theta_grid:np.ndarray, phi_grid:np.ndarray, dTheta:float, dPhi:float):
        """
        Create a shape from the spherical coordinates R, theta, phi.

        R:      the radius of each points
                theta, phi: the grids of the sphere points in spherical coordinates
                dTheta, dPhi: the sampling step of theta and phi respectively

        Note: if you integrate f(x)=1 on a SphericalShape, the result will be the integral on a sphere of radius 1 of the radius squared
        """
        self.R = R_grid

        points = sphere2cartesian(R_grid,theta_grid,phi_grid)
        jacobian = (lambda r,theta,_: np.multiply(np.power(r,2),np.sin(theta))*dTheta*dPhi)
        dS = jacobian(R_grid, theta_grid, phi_grid)

        super().__init__(points, dS)

    def plot_surface(self, fig:plt.Figure, ax:plt.Axes, title="Spherical plot", title_size=25, xlabel="X", ylabel="Y", zlabel="Z", label_size=15, cmap:colors.Colormap=plt.get_cmap("magma"), colorbar=False, cbar_label="", nb_cbar_ticks=5, ticks_size=12, cbar_ticks_format="1.2e"):
        """
            Plot the radius of the shape over the cartesian plane. The aspect ratio of the plot is 1, which means that all 3 axis have the same scale
            The colormap use the normalized radius as metric.
            The ticks of the colormap are taken linearly between max(Radius) and min(Radius)


            fig,ax: Respectively figure and axis on which to plot
            title: The title of the plot

            xlabel, ylabel, zlabel: The label of the 3 cartesian axis

            cmap: The colormap used to color the surface

            colorbar: True to add the colorbar to the plot.
            cbar_label: The label of the colorbar
            nb_cbar_ticks: Number of ticks on the colorbar
            cbar_ticks_format: str format used for the ticks formatting
        """

        Rmax = np.amax(self.R)
        Rmin = np.amin(self.R)
        colors = cmap(min_max_scale(self.R))

        surf = ax.plot_surface(*self.points, cmap="magma", facecolors=colors)

        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.set_zlabel(zlabel, fontsize=label_size)


        ax.set_title(title, fontsize=title_size)
        ax.set_aspect("equal")

        if colorbar:
            cbar_ticks = np.linspace(0,1,nb_cbar_ticks)
            cbar_ticks_label = cbar_ticklabels_formating(np.linspace(Rmin,Rmax,nb_cbar_ticks), cbar_ticks_format)
            cbar = fig.colorbar(surf, location="left")

            cbar.ax.set_yticks(cbar_ticks)
            cbar.ax.set_yticklabels(cbar_ticks_label, fontsize=ticks_size)
            cbar.ax.set_ylabel(cbar_label)
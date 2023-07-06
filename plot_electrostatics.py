import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from create_mesh_device import unpack_geom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d


def plot_V_xy(field, geometry, zp, N=200, vmax=None, vmin = None):
    """
    Plots the electrostatic potential of the device in the xy plane at a certain height z,
    also plotting the pattern of the pg
    Args:
        -field: problem.fields[0] of diffusive_solver after solving the problem (problem.solve)
        -geometry: dictionary containing the geometry of the device
        -z: height at which we want to plot
    """
    eps = 1e-5
    Lx, Ly, _, _, _, w, r, d, h, diam, w_g = unpack_geom(geometry)
    x = np.linspace(0+eps, Lx-eps, N)
    y = np.linspace(0+eps, Ly-eps, N)
    z = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
                 z[i,j] = field(xi, yj, zp)
    
    if vmin is None: vmin = (-z).min()
    if vmax is None: vmax = (-z).max()
    plt.imshow(-z.T,origin='lower', extent =[x.min(), x.max(), y.min(), y.max()],
               cmap='plasma', vmin = vmin, vmax = vmax)
    
    # Plot of the patterned gate
    lx2 = (Lx-d-2*h) / 2    
    ly2 = (Ly-w) / 2
    w2 = (w-r) / 2
    plt.plot([0, lx2], [ly2, ly2], c = "black")
    plt.plot([lx2, lx2+h], [ly2, ly2+w2], c = "black")
    plt.plot([lx2+h, lx2+h], [ly2+w2, ly2+w2+r], c = "black")
    plt.plot([lx2, lx2+h], [ly2, ly2+w2], c = "black")
    plt.plot([lx2, lx2+h], [ly2+w, ly2+w-w2], c = "black")
    plt.plot([0, lx2], [Ly-ly2, Ly-ly2], c = "black")

    plt.plot([Lx-lx2, Lx], [ly2, ly2], c = "black")
    plt.plot([Lx-lx2-h, Lx-lx2], [ly2+w2, ly2], c = "black")
    plt.plot([Lx-lx2-h, Lx-lx2-h], [ly2+w2, ly2+w2+r], c = "black")
    plt.plot([Lx-lx2-h, Lx-lx2], [Ly-ly2-w2, Ly-ly2], c = "black")
    plt.plot([Lx-lx2, Lx], [Ly-ly2, Ly-ly2], c = "black")
    theta = np.linspace(0, 2*np.pi)
    plt.plot([(Lx/2 + diam/2*np.cos(t)) for t in theta], [Ly/2 + diam/2*np.sin(t) for t in theta],  c = "black")
    
    # Plot of the golden gate
    wg2 = (Lx-w_g)/2
    plt.vlines(wg2, y.min(), y.max(), colors=mpl.colormaps['winter'](1.0), lw=1.3)
    plt.vlines(Lx-wg2, y.min(), y.max(), colors=mpl.colormaps['winter'](1.0), lw=1.3)

    # Labels 
    plt.title(r'$\phi$ at z='+f' {zp} nm')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.colorbar(label=r'$V$ (eV)')

    plt.show()
    
def plot_V_xz(field, geometry, yp, N=100, vmin = None, vmax = None):
    """
    Plots the electrostatic potential of the device in the xz plane a certain height z,
    also plotting the pattern of the pg
    Args:
        -field: problem.fields[0] of diffusive_solver after solving the problem (problem.solve)
        -geometry: dictionary containing the geometry of the device
        -z: height at which we want to plot
    """
    eps = 1e-5
    Lx, _, t1, t2, t3, _, _, _, _, diam, wg = unpack_geom(geometry)
    Lz = t1+t2+t3
    
    x = np.linspace(0+eps, Lx-eps, N)
    z = np.linspace(0+eps, Lz-eps, N)
    
    y = np.zeros((len(x), len(z)))
    for i, xi in enumerate(x):
        for j, zj in enumerate(z):
                 y[i,j] = field(xi, yp, zj)

    fig, ax = plt.subplots(figsize=(8, 6))
    if vmin is None: vmin = (-y).min()
    if vmax is None: vmax = (-y).max()
    im = ax.imshow(-y.T,origin='lower', interpolation='nearest',
                   extent =[x.min()-Lx/2, x.max()-Lx/2, z.min(), z.max()], 
                   aspect='auto', cmap='plasma', vmin = vmin, vmax = vmax)
    cs = ax.contour(-y.T, 10, colors='black', alpha = 0.8, vmin = vmin, vmax = vmax, linewidths=1,  # equipotential lines
                    extent =[x.min()-Lx/2, x.max()-Lx/2, z.min(), z.max()])
    ax.clabel(cs, fontsize=10, inline=True, fmt='%.1f', colors='k')
    
    #  Labels
    ax.set_title(fr'View in the $xz$ plane of $\phi$ for r={diam/2}nm')
    ax.set_xlabel('x (nm)', fontsize=18)
    ax.set_ylabel('z (nm)', fontsize=18)
    ax.tick_params(which='major', labelsize=18)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size = "5%", pad = 0.5)
    fig.add_axes(cax)
    
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label( label= r'$V$ (eV)', fontsize=18)
    
    #  Different gates and the BLG
    ax.hlines(0, x.min()-Lx/2, x.max()-Lx/2, lw=4.0, colors='k')  # BG
    ax.hlines(t1, x.min()-Lx/2, x.max()-Lx/2, lw=2.0, colors=mpl.colormaps['winter'](0.5))  # BLG
    ax.hlines(t1+t2, x.min()-Lx/2, x.max()-Lx/2, lw=1.5, colors='k', ls='--')  # PG
    ax.hlines(t1+t2+t3, x.min()-wg/2, x.min()+wg/2, lw=4.0, colors=mpl.colormaps['winter'](1.0))  #GG
    
    plt.show()


def plot_delta_xy(current, geometry, zp, N=200, vmax = None, vmin = None):
    """
    Plots at a certain height z, also plotting the pattern of the pg
    Args:
        -current: "problem.currents[0] de diffusive solver (después de hacer problem.solve)
        -geometry = [Lx, Ly, t1, t2, t3, w, r, d, h, diam, w_g]
        -zp: height at which we want to plot
    """
    eps = 1e-5
    Lx, Ly, _, _, _, w, r, d, h, diam, w_g = unpack_geom(geometry)
    x = np.linspace(0+eps, Lx-eps, N)
    y = np.linspace(0+eps, Ly-eps, N)
    z = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
                 z[i,j] = current(xi, yj, zp)[2]
    z = gaussian_filter1d(z, 1.5)  # to get a better image
    if vmin is None: vmin = (0.34*z).min()
    if vmax is None: vmax = (0.34*z).max()
    # multiply by 0.34 that is the interlayer distance
    plt.imshow(0.34*z.T,origin='lower', extent =[x.min(), x.max(), y.min(), y.max()],
               vmax = vmax, vmin = vmin, cmap='plasma')
    
    # Plot of the patterned gate
    lx2 = (Lx-d-2*h) / 2    
    ly2 = (Ly-w) / 2
    w2 = (w-r) / 2
    plt.plot([0, lx2], [ly2, ly2], c = "black")
    plt.plot([lx2, lx2+h], [ly2, ly2+w2], c = "black")
    plt.plot([lx2+h, lx2+h], [ly2+w2, ly2+w2+r], c = "black")
    plt.plot([lx2, lx2+h], [ly2, ly2+w2], c = "black")
    plt.plot([lx2, lx2+h], [ly2+w, ly2+w-w2], c = "black")
    plt.plot([0, lx2], [Ly-ly2, Ly-ly2], c = "black")

    plt.plot([Lx-lx2, Lx], [ly2, ly2], c = "black")
    plt.plot([Lx-lx2-h, Lx-lx2], [ly2+w2, ly2], c = "black")
    plt.plot([Lx-lx2-h, Lx-lx2-h], [ly2+w2, ly2+w2+r], c = "black")
    plt.plot([Lx-lx2-h, Lx-lx2], [Ly-ly2-w2, Ly-ly2], c = "black")
    plt.plot([Lx-lx2, Lx], [Ly-ly2, Ly-ly2], c = "black")
    theta = np.linspace(0, 2*np.pi)
    plt.plot([(Lx/2 + diam/2*np.cos(t)) for t in theta], [Ly/2 + diam/2*np.sin(t) for t in theta],  c = "black")


    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    

    wg2 = (Lx-w_g)/2
    plt.vlines(wg2, y.min(), y.max(), colors =mpl.colormaps['winter'](1.0), lw=1.3) #, colors='orange'
    plt.vlines(Lx-wg2, y.min(), y.max(), colors=mpl.colormaps['winter'](1.0), lw=1.3)
    
    plt.colorbar(label=r'$\Delta$ (eV)')
    plt.show()
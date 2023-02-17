import numpy as np
import matplotlib.pyplot as plt

def reconstructEGF1D(model):
    G = model.modeset @ np.diag(model.dcoeffs.flatten()) @ model.modeset.T
    return G

# def computeError(G_reconstruction, G):
#     # Errors for 1D problems. For 2D problems, one has to compute norm(w * (G_reconstruction-G) * w')/norm(w * G * w')
#     return np.linalg.norm(G_reconstruction - G, ord = 2)/np.linalg.norm(G, ord = 2)

def normL2(G, meshweights):
    meshweights = meshweights.reshape((-1,1))
    return np.sqrt(np.sum(meshweights * (G*G) * meshweights.T))

def errorL2(G_emp, G, meshweights):
    return normL2(G_emp-G, meshweights)/normL2(G, meshweights)

def plotGreen1D(model, vmin = None, vmax = None):
    domain = model.mesh.coordinates()
    x, s = np.meshgrid(domain,domain)
    plt.figure(figsize = (7,5))
    
    G_reconstruction = reconstructEGF1D(model)
    
    surf = plt.contourf(x, s, G_reconstruction, 20, cmap = 'jet', vmin = vmin, vmax = vmax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(surf)
    plt.xlabel('x')
    plt.ylabel('s', rotation='horizontal', labelpad=5)
    plt.title("Empirical Green's function")

def compareGreen1D(model, exactGreen, vmin = None, vmax = None):
    domain = model.mesh.coordinates()
    x, s = np.meshgrid(domain,domain)
    plt.figure(figsize = (14,5))
    plt.subplot(121)
    
    G_reconstruction = reconstructEGF1D(model)
    
    surf = plt.contourf(x, s, G_reconstruction, 20, cmap = 'jet', vmin = vmin, vmax = vmax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(surf)
    plt.xlabel('x')
    plt.ylabel('s', rotation='horizontal', labelpad=5)
    plt.title("Empirical Green's function")
    
    G = exactGreen(domain, model.params[0])
    
    plt.subplot(122)
    plt.gca().set_aspect('equal', adjustable='box')
    surf = plt.contourf(x, s, G, 20, cmap = 'jet', vmin = vmin, vmax = vmax)
    plt.colorbar(surf)
    plt.xlabel('x')
    plt.ylabel('s', rotation='horizontal', labelpad=5)
    plt.title("Exact Green's function")

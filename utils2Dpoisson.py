from fenics import *
import mshr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def reconstructGxx00(model):
    xloc = model.mesh.coordinates()
    sloc = np.zeros((1,2))
    
    V = FunctionSpace(model.mesh, 'P', 1)
    d2v = dof_to_vertex_map(V)
    modes_at_xloc = np.empty((xloc.shape[0],model.rank))
    modes_at_sloc = np.empty((sloc.shape[0],model.rank))
    for mode in range(model.rank):
        u = Function(V)
        vals = model.modeset[:,mode]
        u.vector()[:] = vals[d2v]
        for xind in range(xloc.shape[0]):
            modes_at_xloc[xind,mode] = u(*xloc[xind,:])
        for sind in range(sloc.shape[0]):
            modes_at_sloc[sind,mode] = u(*sloc[sind,:])
    G = modes_at_xloc @ np.diag(model.dcoeffs.flatten()) @ modes_at_sloc.T
    return G

def plotGxx00(model):
    G = reconstructGxx00(model)
    
    # print(G)
    V = FunctionSpace(model.mesh, 'P', 1)
    d2v = dof_to_vertex_map(V)
    u = Function(V)
    temp = np.squeeze(G)
    u.vector()[:] = temp[d2v]
    
    plt.figure(figsize = (14,5))
    plt.subplot(121)
    
    p = plot(u, levels = 30, cmap = 'jet', vmin = -0.81, vmax = 0.0) # 
    plt.colorbar(p)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if model.type == "coefficient-fit":
        plt.title("Empirical Green's function")
    else:
        plt.title("Randomized SVD")
    
    
    # Compute exact G
    domain = np.linspace(-1,1,301)
    
    x, s = np.meshgrid(domain,domain)
    G_exact = np.zeros(np.shape(x))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xx, ss = x[i,j], s[i,j]
            if xx**2 + ss**2 <= 1.0 and (xx != 0 or ss != 0):
                G_exact[i,j] = (1/(4*pi))* np.log(xx**2 + ss**2)
            else:
                G_exact[i,j] = np.nan
            
    plt.subplot(122)
    plt.gca().set_aspect('equal', adjustable = 'box')
    surf = plt.contourf(x, s, G_exact, levels = 30, cmap = 'jet', vmin = -0.81, vmax = 0.0)
    circ = patches.Circle((0, 0), 1.0, transform = plt.gca().transData, facecolor = 'none')
    plt.gca().add_patch(circ)
    for col in surf.collections:
        col.set_clip_path(circ)

    plt.colorbar(surf)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if model.type == "coefficient-fit":
        plt.title("Empirical Green's function")
    else:
        plt.title("Randomized SVD")

def reconstructGx0s0(model, samples):
    eps = 1e-3
    xloc = np.hstack((np.linspace(-1+eps, 1-eps, num = samples).reshape(samples,1),np.zeros((samples,1))))
    sloc = np.hstack((np.linspace(-1+eps, 1-eps, num = samples).reshape(samples,1),np.zeros((samples,1))))

    V = FunctionSpace(model.mesh, 'P', 1)
    d2v = dof_to_vertex_map(V)
    modes_at_xloc = np.empty((xloc.shape[0],model.rank))
    modes_at_sloc = np.empty((sloc.shape[0],model.rank))
    
    for mode in range(model.rank):
        u = Function(V)
        vals = model.modeset[:,mode]
        u.vector()[:] = vals[d2v]
        for xind in range(xloc.shape[0]):
            modes_at_xloc[xind,mode] = u(*xloc[xind,:])
        for sind in range(sloc.shape[0]):
            modes_at_sloc[sind,mode] = u(*sloc[sind,:])
    G = modes_at_xloc @ np.diag(model.dcoeffs.flatten()) @ modes_at_sloc.T
    
    x, s = np.meshgrid(np.linspace(-1, 1, num = samples), np.linspace(-1, 1, num = samples))
    G_true = np.zeros(np.shape(x))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xx, ss = x[i,j], s[i,j]
            if abs(xx - ss) > 1e-5:
                G_true[i,j] = (1/(4*pi))* np.log((xx-ss)**2/(xx*ss-1)**2)
            else:
                G_true[i,j] = np.nan
    
    return G, G_true

def plotGx0s0(model):
    samples = 100
    G, G_true = reconstructGx0s0(model, samples)
    
    x, s = np.meshgrid(np.linspace(-1, 1, num = samples), np.linspace(-1, 1, num = samples))
    plt.figure(figsize = (14,5))
    plt.subplot(121)

    surf = plt.contourf(x, s, G, 30, cmap = 'jet', vmin = -0.8, vmax = 0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(surf)
    plt.xlabel('$x_1$')
    plt.ylabel('$s_1$')
    plt.title("Empirical Green's function")
    
    plt.subplot(122)
    plt.gca().set_aspect('equal', adjustable='box')
    surf = plt.contourf(x, s, G_true, levels = 30, cmap = 'jet', vmin = -0.8, vmax = 0)
    # surf = plt.imshow(G, interpolation='lanczos', cmap='jet', extent=[-1,1,-1,1], vmin = -0.8, vmax=0)
    plt.colorbar(surf)
    plt.xlabel('$x_1$')
    plt.ylabel('$s_1$')
    plt.title("Exact Green's function")

# def compareGxx00(modelA, modelB, modelAname = None, modelBname = None, vmin = None, vmax = None):
#     Ga = reconstructGxx00(modelA)
#     Gb = reconstructGxx00(modelB)
    
#     V = FunctionSpace(modelA.mesh, 'P', 1)
#     d2v = dof_to_vertex_map(V)
#     u = Function(V)
#     temp = np.squeeze(Ga)
#     u.vector()[:] = temp[d2v]
    
#     plt.figure(figsize = (14,5))
#     plt.subplot(121)
    
#     p = plot(u, levels = 30, cmap = 'jet', vmin = vmin, vmax = vmax)
#     plt.colorbar(p)
#     plt.xlabel('$x_1$')
#     plt.ylabel('$x_2$')
#     if modelAname is not None:
#         plt.title(modelAname)
#     else:
#         plt.title("Model 1")
    
#     V = FunctionSpace(modelB.mesh, 'P', 1)
#     d2v = dof_to_vertex_map(V)
#     u = Function(V)
#     temp = np.squeeze(Gb)
#     u.vector()[:] = temp[d2v]
    
#     plt.subplot(122)
    
#     p = plot(u, levels = 30, cmap = 'jet', vmin = vmin, vmax = vmax)
#     plt.colorbar(p)
#     plt.xlabel('$x_1$')
#     plt.ylabel('$x_2$')
#     if modelBname is not None:
#         plt.title(modelBname)
#     else:
#         plt.title("Model 2")
    
#     plt.suptitle("$G(x_1,x_2,0,0)$")

# def compareGx0s0(modelA, modelB, modelAname = None, modelBname = None, vmin = None, vmax = None):
#     samples = 100
#     Ga, _ = reconstructGx0s0(modelA, samples)
#     Gb, _ = reconstructGx0s0(modelB, samples)
#     print(Ga.shape)
#     x, s = np.meshgrid(np.linspace(-1, 1, num = samples), np.linspace(-1, 1, num = samples))
#     plt.figure(figsize = (14,5))
#     plt.subplot(121)

#     surf = plt.contourf(x, s, Ga, 30, cmap = 'jet', vmin = vmin, vmax = vmax)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar(surf)
#     plt.xlabel('$x_1$')
#     plt.ylabel('$s_1$')
#     if modelAname is not None:
#         plt.title(modelAname)
#     else:
#         plt.title("Model 1")
    
#     plt.subplot(122)
#     plt.gca().set_aspect('equal', adjustable='box')
#     surf = plt.contourf(x, s, Gb, levels = 30, cmap = 'jet', vmin = vmin, vmax = vmax)
#     # surf = plt.imshow(G, interpolation='lanczos', cmap='jet', extent=[-1,1,-1,1], vmin = -0.8, vmax=0)
#     plt.colorbar(surf)
#     plt.xlabel('$x_1$')
#     plt.ylabel('$s_1$')
#     if modelBname is not None:
#         plt.title(modelBname)
#     else:
#         plt.title("Model 2")

#     plt.suptitle("G(x,0,xbar, 0)")
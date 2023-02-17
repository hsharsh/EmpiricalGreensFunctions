from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

def transform_coordinates(A, theta_1, theta_2):
    B = A
    a = (pi-theta_2)/(pi-theta_1)
    b = pi-a*pi
    
    # Change to polar coordinates
    r = np.sqrt(B[:,0]**2+B[:,1]**2)
    t = np.arctan2(B[:,1], B[:,0])
    
    # Find index of positive y
    i_p = B[:,1] >= 0
    i_m = B[:,1] < 0
    
    # Interpolate angles
    t[i_p] = a*t[i_p] + b   
    t[i_m] = a*t[i_m] - b
    
    # Convert to cartesian coordinates
    B[:,0] = r*np.cos(t)
    B[:,1] = r*np.sin(t)
    
    return B

class EGF:
    def __init__(self,
                type        : str,
                params      : np.array,
                rank        : int,
                mesh        : any,
                inputdata   : np.array,
                outputdata  : np.array = None,
                noise_level : float     = None,
                modeset     : np.array  = None,
                dcoeffs     : np.array  = None,
                Sim         : any       = None,
                verbose     : bool      = False,
                ):
        """
        Arguments:
            type: _description_. Defaults to "egf".
                params: An array of size (n_{param_dimension} x 1) array, which defines the parameters for
                the model.
            rank: The number of columns in the "modeset" of the model. In case of an "egf" model, this is the number
                of columns at which the Singular Value decomposition of the output ensemble (outputdata) is truncated,
                or equivalently, it is the number of modes used to represent our model for the system. For a
                "randomized-svd" model, it is the number of left singular vectors and the singular values (potentially
                with flipped signs) used to represent the model.
            mesh: A FENICS mesh object used to generate the weights corresponding to each node and to generate a Finite
                Element function space (DOLFIN object) to perform any integration necessary.
            inputdata: A collection of forcing/input functions sampled at N_sample location. Each discreteized forcing
                function is repsented by a column vector and these columns are stacked together to create this input
                ensemble which is an array of size (N_sensors x N_samples). 
            outputdata: A collection of responses from the system when it is probed with the forcing functions defined
                in inputdata. Each column, outputdata[:,i], is the response corresponding to inputdata[:,i], which is
                then sampled at N_sensor locations to create this output ensemble. This is an array of size
                (N_sensors x N_samples). In case of a "randomized-egf" model, the outputdata generation is chosen to be
                a part of the modelling procedure (design choice) since we need the Simulator (the true solution
                operator) for the algorithm. Defaults to None.
            noise_level: This is the fraction of additive Gaussian White noise with mean 0 and variance equal to the
                average intensity of the response. This is defined only for the "randomized-egf" model as the simulator
                is a part of the algorithm. Defaults to None.
            modeset: The set of eigenfunction for the model, sampled at N_sensors locations, represented as a matrix of
                stacked columns of size N_sensors. This argument is only used to define an EGF model when you know the
                mode set  beforehand (in case of interpolation). Defaults to None.
            dcoeffs: The set of eigenvalues for the model. This argument is only used to define an EGF model
                when you know the mode set beforehand (in case of interpolation). Defaults to None.
            Sim: An object of Simulator class used to generate discretized system responses corresponding to a forcing
                forcing (in case of "randomized-svd" model). The main function for the class is Sim.solve(), which takes
                the params, noise_level, and a forcing vector as input and generates the corresponding system response.
                Note that this process can be implemented more efficiently by reducing the overheads due to multiple
                function calls but for the sake of a better understanding of the method, the solve method takes only one
                forcing vector at a time. Defaults to None.
            verbose: If set to True, the algorithm generates more debugging info. Defaults to False.
        """
        self.type = type
        self.forcing = inputdata
        self.params = params
        self.rank = rank
        self.nSens, self.nIO = inputdata.shape
        self.mesh = mesh

        # Compute the weights corresponding to each node in the mesh using FENICS.
        self.meshweights = Sim.meshweights

        # Model type "coefficient-fit" computes an Empirical Green's Function by computing the eigenfunction by finding
        # the left singular vectors of the output ensemble, outputdata. Subsequently, the eigenvalues (referred to as
        # coefficients/diagonal coefficients) are fit so that the modeset and the diagonal coefficient construct the
        # spectral decomposition of the Green's function. This is described in more detail in the paper.
        if type == "randomized-svd":

            param = self.params[0]
            self.signal = outputdata
                
            # Normalize the solution ensemble using the meshweights from FENICS. L2 orthonormality
            # (\sum_i weights(i) * q_j(i) = 1 for all j)
            # q, _= np.linalg.qr(self.signal)
            q, _= np.linalg.qr(np.sqrt(self.meshweights)*self.signal)
            q = q/np.sqrt(self.meshweights)
            
            if verbose:
                print("QR decomposition")
            
            v = np.zeros(q.shape)
            for i in range(q.shape[1]):
                if verbose:
                    print("Solving again, i = %d / %d" % (i+1,q.shape[1]))
                if noise_level is not None:
                    v[:,i] = Sim.solve(q[:,i], noise_level, param)
                else:
                    v[:,i] = Sim.solve(q[:,i], None, param)
            
            if verbose:
                print("Size G = %d" %q.shape[0])
            # G = np.zeros((q.shape[0],q.shape[0]))
            # for i in range(self.rank):
            #     G += q[:,i:i+1] * v[:,i:i+1].T

            # self.G = 0.5*(G+G.T)
            # Store the learnt empirical Green's function as a modeset and coefficient vector.
            #U, S, Vt = np.linalg.svd(self.G, full_matrices = False)
            
            # Construct Low rank approximation for G (https://gregorygundersen.com/blog/2019/01/17/randomized-svd/)
            U_tilde, S, Vt = np.linalg.svd(v.T * np.sqrt(self.meshweights).T, full_matrices = False)
            U = np.matmul(q,U_tilde)
            Vt = Vt / np.sqrt(self.meshweights).T

            # self.G = U @ np.diag(S) @ Vt
            # self.G = 0.5*(self.G + self.G.T)
            
            self.modeset = U[:,:self.rank]
                
            # signs = Vt @ np.diag(self.meshweights.squeeze()) @ U
            # print(signs)
            # print(np.diag(signs))
                  
            signs = np.ones(self.rank)
            for i in range(self.rank):
                if np.dot(self.meshweights.squeeze() * U[:,i], Vt[i,:]) < 0:
                    signs[i] = -1
            
            
            self.dcoeffs = (signs * S[:self.rank]).reshape((-1,1))

        # For defining an interpolated model
        elif type == None:
            self.modeset = modeset
            self.dcoeffs = dcoeffs
            # self.signal = self.reconstruct_signal()
        
        else:
            RuntimeError('Unknown model type!')

#     def reconstruct_signal(self, forcing: np.array = None) -> np.array:
#         """ For a model of type "coefficient-fit", this function uses the Empirical Green's function to reconstruct the
#         system response by passing the specified forcing. If this is not set, then the model uses the same set of
#         forcing functions which were used to construct the model. 

#         ----------------------------------------------------------------------------------------------------------------
#         Arguments:
#             forcing: A collection of forcing/input functions sampled at N_sample location. Each discreteized forcing
#                 function is repsented by a column vector and these columns are stacked together to create this input
#                 ensemble which is an array of size (N_sensors x N_samples). Defaults to None

#         ----------------------------------------------------------------------------------------------------------------
#         Returns:
#             A collection of responses where each column, outputdata[:,i], is the response corresponding to forcing[:,i],
#             which is then sampled at N_sensor locations to reconstruct the signal using an Empirical Green's Function.
#             This is an array of size (N_sensors x N_samples).
#         """
#         # Define the Finite Element Function space need to perform the integration over the mesh.
#         V = FunctionSpace(self.mesh, 'P', 1)
#         d2v = dof_to_vertex_map(V)
        
#         # Set the forcing to inputdata used to fit the model in case it is not specified by the user.
#         if forcing is None:
#             forcing = self.forcing

#         outputdata = np.empty((self.nSens, forcing.shape[1]))

#         # Use FENICS to evaluate the responses by passing the forcing through an integral with the Empirical Green's
#         # Function as the kernel
#         for i in range(forcing.shape[1]):
#             intvals = np.zeros((self.rank,))

#             for mode in range(self.rank):
#                 temp = self.modeset[:,mode]*forcing[:,i]
#                 product = Function(V)
#                 product.vector()[:] = temp[d2v]
#                 intvals[mode] = assemble(product*dx)

#             modecoeffs = (self.dcoeffs.reshape((self.rank,)) * intvals).reshape((self.rank,))

#             outputdata[:,i] = np.sum(self.modeset[:,:] * modecoeffs, axis = 1)
#         return outputdata

def sampleforcing2D(sigma, nSamples):
    """
    Sample nSamples random functions generated from a GP with squared-exp kernel with length scale parameter sigma using Chebfun.
    Ensure that a data1D.mat (mesh locations where the randomly generated chebfun function is sampled at) by initializing a Simulator class.
    """
    # if not(os.path.exists("dat2D.mat")):
    matlab_path = "/Applications/MATLAB_R2022a.app/bin/matlab"
    os.system(f"{matlab_path} -nodisplay -nosplash -nodesktop -r \"run('sample2D({int(sigma*10000)},{nSamples})'); exit;\" | tail -n +11")
    data = scipy.io.loadmat("dat2D.mat")
    forcing = data['F']
    return forcing

class Simulator:
    
    def __init__(self, theta):
        # Define the domain and mesh for the solving the PDE.
        basemesh = Mesh("pacman.msh")
        theta_0 = pi/2
        Vc = basemesh.coordinates.function_space()
        x, y = SpatialCoordinate(basemesh)
        m = Function(Vc).interpolate(as_vector([x, y]))

        m.dat.data[:] = transform_coordinates(m.dat.data[:], theta_0/2, theta/2)
        self.mesh = Mesh(m)      
        self.V = FunctionSpace(self.mesh, "CG", 3)
        
        # Store the meshweights for computation
        m = self.V.ufl_domain()
        W = VectorFunctionSpace(m, self.V.ufl_element())
        X = interpolate(m.coordinates, W)
        X_samples = X.dat.data_ro
        mesh_dict = {"X": X_samples}
        scipy.io.savemat("mesh2D.mat", mesh_dict)
        v = TestFunction(self.V)
        temp = assemble(v*dx)
        self.meshweights = temp.vector().get_local().reshape(-1,1)
        
        # Solution to Laplace equation
        self.u = Function(self.V, name="u")
        v = TestFunction(self.V)
        self.f = Function(self.V)
        
        # Weak form
        self.F = inner(grad(self.u),grad(v))*dx - self.f*v*dx

        # Boundary conditions
        self.bc = [DirichletBC(self.V, Constant(0.0), "on_boundary")]

    
    def solve(self, forcing, noise_level = None, param = None):
        """
        Given a (N_sensors x 1) forcing vector, solve the a 2D Helmholtz problem on a unit disc.
        """
        
        self.f.dat.data[:] = forcing # Instantiate the source term in the variational form by interpolating the sampled sourcing term.
        # Solve PDE
        solve(self.F==0, self.u, self.bc)

        # Sample the solution at the nodes of the mesh.
        solution = self.u.dat.data
        
        # As specified, add IID Gaussian white noise.
        if noise_level is not None:
            noise =  noise_level*np.random.normal(0,np.abs(solution.mean()),solution.shape)
            solution += noise
        
        return solution
    
def compute_order_and_signs(R0: np.array, R1: np.array) -> tuple([np.array, np.array]):
    """ Given two orthonormal matrices R0 and R1, this function computes the "correct" ordering and signs of the columns
    (modes) of R1 using R0 as a reference. The assumption is that these are orthonormal matrices, the columns of which
    are eigenmodes of systems which are close to each other and hence the eigenmodes will close to each other as well.
    We thus find an order such that the modes of the second matrix have the maximum inner product (in magnitude) with
    the corresponding mode from the first matrix. If such an ordering doesn't exist the function raises a runtime error.

    Once such an ordering is found, one can flip the signs for the modes of R1, if the inner product is not positive.
    This is necessary when we want to interpolate.

    --------------------------------------------------------------------------------------------------------------------
    Args:
        R0: Orthonormal matrix, the modes of which are used as the reference to re-order and find signs
        R1: Orthonormal matrix for which the modes are supposed to be reordered.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        New ordering and the signs (sign flips) for matrix R1.
    """
    rank = R0.shape[1]
    order = -1*np.ones(rank).astype(int)
    signs = np.ones(rank)
    
    used = set()
    # For each mode in R1, Search over all modes of R0 for the best matching mode.
    for i in range(rank):
        basemode = R0[:,i]
        maxidx, maxval = -1, -1
        for j in range(rank):
            current = np.abs(np.dot(basemode, R1[:,j])) # Compute the magnitude of the inner product
            if current >= maxval and (j not in used):
                maxidx = j
                maxval = current
        order[i] = maxidx
        used.add(maxidx)
    
    # Raise an error if the ordering of modes is not a permutation.
    check = set()
    for i in range(rank):
        check.add(order[i])

    if len(check) != rank:
        raise RuntimeError('No valid ordering of modes found')
        
    # Signs are determined according to the correct ordering of modes
    for i in range(rank):
        if np.dot(R0[:,i],R1[:,order[i]]) < 0:
            signs[i] = -1
    
    return order, signs

def compute_interp_coeffs(models : np.array, targetParam: np.array) -> np.array:
    """Computes the interpolation coefficients (based on fitting Lagrange polynomials) for performing interpolation,
    when the parameteric space is 1D. Note that the function takes in parameters of any dimnesions

    --------------------------------------------------------------------------------------------------------------------
    Arguments:
        models: Set of models (EGF objects) which are used to generate an interoplated model at the parameter
            targetParam.
        targetParam: An array of size (n_{param_dimension} x 1) array which defines the parameters for the model which
            we want to interpolate.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        A numpy array of Interpolation coefficents index in the same way as in the set models.
    """
    assert(not models == True)

    if (models[0].params.shape[0]!= 1):
        raise RuntimeError("Lagrange Polynomial based interpolation requires the parameteric space to be 1D.")


    a = np.ones(len(models))
    if len(models[0].params) == 1:
        thetas = [model.params[0] for model in models]
        theta = targetParam[0] # Define the target parameter
        for i,t1 in enumerate(thetas):
            for j,t2 in enumerate(thetas):
                if i != j:
                    a[i] = a[i]*((theta - t2)/(t1 - t2))

    return a

def model_interp(models, inputdata, targetParam, verbose = False):
    """
    Interpolation for the models (orthonormal basis + coefficients). THe orthonormal basis is interpolated in the
    tangent space of compact Stiefel manifold using a QR based retraction map. The coefficients are interpolated
    directly (entry-by-entry) using a Lagrange polynomial based inteporlation. Note that currently the interpolation
    only supports 1D parameteric spaces for the model as the method for interpolation within the tangent space only
    supports 1D parameteric spaces but this can be easily extended to higher dimensions. The lifting and retraction to
    the tangent space of an "origin" has no dependendence on the dimensionality of the parameteric space. The current
    implementation takes models of type randomized-svd. Again, this is not a limitation of the method as the core
    interpolation scheme is not dependent on the model type but only needs orthonormal bases.

    --------------------------------------------------------------------------------------------------------------------
    Arguments:
        models: Set of models (EGF objects) which are used to generate an interoplated model at the parameter
            targetParam.
        Sim: An object of Simulator class used to generate discretized system responses corresponding to a forcing
                forcing (in case of "randomized-svd" model). The main function for the class is Sim.solve(), which takes
                the params, noise_level, and a forcing vector as input and generates the corresponding system response.
                Note that this process can be implemented more efficiently by reducing the overheads due to multiple
                function calls but for the sake of a better understanding of the method, the solve method takes only one
                forcing vector at a time. Note that the manifold interpolation doesn't require the simulator object. We
                use it to define the EGF objects and then do integrations for visualization. When applying this method
                to real world problems, we don't need it. Defaults to None.
        inputdata: A collection of forcing/input functions sampled at N_sample location. Each discreteized forcing
                function is repsented by a column vector and these columns are stacked together to create this input
                ensemble which is an array of size (N_sensors x N_samples). Note that the manifold interpolation does
                not require the input ensemble. These are used for computing the empirical errors in the model is of
                type "coefficient-fit".
        targetParam: An array of size (n_{param_dimension} x 1) array which defines the parameters for the model which
            we want to interpolate.
        verbose: If set to True, the algorithm generates more debugging info. Defaults to False.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        An EGF object with the interpolated basis.
    """
    dists = []

    assert(not models == True)
    assert(models[0].type == "randomized-svd")

    nSens = models[0].nSens
    # nIO = models[0].nIO
    rank = models[0].rank
    
    # V = FunctionSpace(Sim.mesh,'P',1)
    # u = TestFunction(V)
    # temp = assemble(u*dx)
    # meshweights = (temp.get_local()[vertex_to_dof_map(V)]).reshape(-1,1)
    

    # Find the model which is closest to target parameter. Note that the distance is calculated in terms of norm of the
    # normalized parameters. 
    for model in models:
        dists.append(np.linalg.norm((model.params-targetParam)/targetParam)) # Element-wise divide is a design choice.
    
    refIndex = np.argmin(dists)
    numModels = len(models)
    
    # Define the basis which is used as the origin.
    # U0, _, _ = np.linalg.svd(np.sqrt(models[refIndex].meshweights) * \
    #                               models[refIndex].G * np.sqrt(models[refIndex].meshweights).T, full_matrices = False)
    U0 = models[refIndex].modeset * np.sqrt(models[refIndex].meshweights)

    # # Interpolation coefficients
    # dists = np.array(dists)
    # rip = np.exp(-dists/np.mean(dists))
    # print(rip)

    a = compute_interp_coeffs(models, targetParam)

    if verbose:
        print(f"Interpolation for dcoeffs: {a}")
    
    coeffs_interp = np.zeros(rank)
    P_interp = np.zeros((nSens, rank))
    
    U_set = [] # Storing these for debugging.
    coeffs_set = [] # Storing these for debugging
    for m in range(numModels):
        # Store the learnt empirical Green's function as a modeset and coefficient matrix.
        # U, S, Vt = np.linalg.svd( np.sqrt(models[m].meshweights) * \
        #                                         models[m].G * np.sqrt(models[m].meshweights).T, full_matrices = False)
        
        Vc = models[m].mesh.coordinates.function_space()
        x, y = SpatialCoordinate(models[m].mesh)
        
        # Interpolator function for changing mesh coordinates
        f = Function(Vc).interpolate(as_vector([x, y]))
        
        f.dat.data[:] = transform_coordinates(f.dat.data[:], models[m].params[0]/2, models[refIndex].params[0]/2)
        
        # Compute determinant of inverse Jacobian of transformation for orthonormalization
        Vdet = FunctionSpace(models[m].mesh, "CG", 2)
        detDf = Function(Vdet, name="det").interpolate(det(inv(grad(f))))
        V1 = FunctionSpace(models[m].mesh, "CG", 3)
        
        detDf = Function(V1, name="det").interpolate(sqrt(detDf))
        
        U = (detDf.dat.data[:].reshape(-1,1)*models[m].modeset) * np.sqrt(models[m].meshweights)
        dcoeffs = models[m].dcoeffs
        
        # Match the modes and subsequently signs with the basepoint
        order, signs = compute_order_and_signs(U0, U)
        U = U[:,order]*signs
        dcoeffs = dcoeffs[order]
        
        # Match the basepoints signs
        for i in range(rank):
            if(U[:,i].T @ U0[:,i] < 0):
                U[:,i] = -U[:,i]
        
        # Project to tangent space of compact Stiefel manifold.
        P = U - U0 @ (U0.T @ U + U.T @ U0)*0.5 # Project and add along with the weight in the interpolation
        
        # Interpolate the matrices in tangent space and diagonal coefficient set with same interpolation coefficients.
        P_interp += P * a[m]
        coeffs_interp += dcoeffs.squeeze() * a[m]
        U_set.append(U)
        coeffs_set.append(dcoeffs)
    
    # Retract back to compact Stiefel manifold using a QR-decomposition based map.
    modeset, _ = np.linalg.qr(U0 + P_interp)
    
    # Reorder and flip signs. Note that we haven't seen an instance which requires reordering with a QR based map which
    # indeed an issue in case of an SVD based map.
    order, signs = compute_order_and_signs(U0, modeset)
    
    Sim = Simulator(targetParam[0])
    
    modeset = modeset[:,order] * signs
    modeset = modeset / np.sqrt(Sim.meshweights)
    coeffs_interp = coeffs_interp[order]
    
    Vc = Sim.mesh.coordinates.function_space()
    x, y = SpatialCoordinate(Sim.mesh)

    # Interpolator function for changing mesh coordinates
    f = Function(Vc).interpolate(as_vector([x, y]))

    f.dat.data[:] = transform_coordinates(f.dat.data[:], models[refIndex].params[0]/2, targetParam[0]/2)

    # Compute determinant of inverse Jacobian of transformation for orthonormalization
    Vdet = FunctionSpace(Sim.mesh, "CG", 2)
    detDf = Function(Vdet, name="det").interpolate(det(inv(grad(f))))
    V1 = FunctionSpace(Sim.mesh, "CG", 3)

    detDf = Function(V1, name="det").interpolate(sqrt(detDf))

    U = (detDf.dat.data[:].reshape(-1,1)*modeset)
    

    # Create an EGF object to return
    interp_model = EGF(None, targetParam, rank, Sim.mesh, inputdata, \
                            None, None, modeset, coeffs_interp, Sim, verbose = False)
    return interp_model, U_set, coeffs_set
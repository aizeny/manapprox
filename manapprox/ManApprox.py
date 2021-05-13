'''
Written by Barak Sober & Yariv Aizenbud. 2019.
If you are using this code please cite:
1) Y. Aizenbud and B. Sober "Non-Parametric Estimation of Manifolds from Noisy Data"
2) B. Sober and D. Levin. "Manifold Approximation by Moving Least-Squares Projection (MMLS)." 
3) B. Sober,Y. Aizenbud, and D. Levin. "Approximation of functions over manifolds: A moving least-squares approach." Journal of Computational and Applied Mathematics
'''
###########
# Imports #
###########
import numpy as np
from scipy import linalg as LA
from scipy.special import comb as nchoosek
import scipy
import numpy.matlib as nlib
from scipy.spatial import cKDTree, distance_matrix
import sys
from time import time as _t


class ManApprox(object):
    """
    Manifold Moving Least-Squares: approximation of manifolds based upon scattered data
    Implements the algoritm described in:
    Sober, B., & Levin, D. (2016). Manifold Approximation by Moving Least-Squares Projection (MMLS). arXiv preprint arXiv:1606.07104.
    """
    def __init__(self, data=[], manifold_dim = None, sparse_factor = 5, sigma = None, poly_deg = 1, thresholdH = 10**-3, init_tree = True, calculate_sigma = True):
        self.EPSILON = 1.e-13
        self.data = data
        self.data_dim = data.shape[0]
        self.approximated_points = []
        self.approximated_points_res = []
        self.manifold_dim = manifold_dim
        self.sparse_factor = sparse_factor
        self.sigma = sigma
        self.poly_deg = poly_deg
        self.thresholdH = thresholdH 
        self.recompute_sigma = False #recalculates sigma based on init_q instead of the given sigma/sigma calculated based on the given point
        self.initSVD = False
        if manifold_dim is None:
            self.kd_tree = None
            return
        if init_tree:
            self.createTree()
        else:
            self.kd_tree = None
        if calculate_sigma:
            self.calculateSigma()
        

    def __repr__(self):
        if self.kd_tree == None:
            is_KDtree = "No"
        else:
            is_KDtree = "Yes"

        output =    """
        Number of data points: %d
        Dimension of data: %d
        Manifold dimension: %s
        KDTree pre-calculated: %s

        Number of approximated points so far: %d

        Sigma: %s 
        threshhold H: %f
        Polynomial degree: %d
        Sparse factor: %d
        
        """%(self.data.shape[1], self.data_dim, str(self.manifold_dim), is_KDtree, len(self.approximated_points), str(self.sigma), self.thresholdH, self.poly_deg, self.sparse_factor )

        return output
    
    def setManifoldDim(self, manifold_dim):
        self.manifold_dim = manifold_dim

    def createTree(self, leaf_size = 10):
        self.kd_tree = MyKDTree(self.data, leaf_size)

    def compactWeightFunc(self, r, k = 1.5):
        '''Compactly supported RBF weight function
        The support size is k*s and the value of at (k-1)s is approximately 0.1
        Notice: this is not a distribution!'''
        
        ret = np.zeros_like(r)
        ret[r < k*self.sigma] = np.exp((-r[r < k*self.sigma]**2)/(r[r < k*self.sigma] - k*self.sigma)**2) 
        return ret
    def compactWeightFuncDot(self, r, k = 1.5):
        '''Compactly supported RBF weight function
        The support size is k*s and the value of at (k-1)s is approximately 0.1
        Notice: this is not a distribution!'''
        indices = r < k*self.sigma
        ret = np.zeros_like(r)
        ret[indices] = self.compactWeightFunc(r,k)[indices] * 2 * r[indices] * k * self.sigma / (r[indices] - k*self.sigma)**2
        return ret

    def calculateSigma(self, n_iter=100):
        '''calculating an approximate distance for the weight function
        This is a very naive implementation!!!'''
        N = self.data.shape[1]    
        N_THRESH = (self.sparse_factor * nchoosek(self.manifold_dim + self.poly_deg,self.manifold_dim)) + 1
        N_PERC = max(min(100*np.float(N_THRESH)/N, 100), 0)
        sig_approximation = np.zeros(n_iter)
        for r_index, i in zip(np.random.randint(3,N-3, n_iter), range(n_iter)):
            q = np.asarray(self.data[:,r_index])
            DISTS = np.linalg.norm(self.data - nlib.repmat(q,self.data.shape[1],1).T, axis = 0)
            sig_approximation[i] = np.percentile(DISTS, N_PERC)
        
        self.sigma = np.max(sig_approximation)
        
    def calculateSigmaFromPoint(self, point):
        '''calculating an approximate distance for the weight function
        This is a very naive implementation!!!'''
        point = point.squeeze()
        N = self.data.shape[1]    
        N_THRESH = min([(self.sparse_factor * nchoosek(self.manifold_dim+self.poly_deg,self.manifold_dim)) + 1, N])
        N_PERC = 100*np.float(N_THRESH)/N
        DISTS = np.linalg.norm(self.data - nlib.repmat(point,self.data.shape[1],1).T, axis = 0)
        sig_approximation = np.percentile(DISTS, N_PERC)
        self.sigma = sig_approximation
    
    def genInitQ(self, point):
        if self.sigma == None:
            #print("Warning: sigma not initialized - initializing with calculateSigma()")
            self.calculateSigmaFromPoint(point)
        if self.kd_tree == None:
            #print("Warning: No KD tree. Looking for neighbors one by one.")
            #print("currently not implemented")
            raise Exception("no KD tree")
        D,I = self.kd_tree.query(point, self.sigma)
        W = self.compactWeightFunc(D[0])
        TI = I[0][W>self.EPSILON]
        if TI.shape[0] == 0:
            print('Point has no neighbors - choosing the nearest data point')
            _, p_i = self.kd_tree.query_k(point,1)
            D,I = self.kd_tree.query(self.data[:,p_i[0]:p_i[0]+1], self.sigma)
            W = self.compactWeightFunc(D[0])
            TI = I[0][W>self.EPSILON]
            if TI.shape[0] == 0:
                raise Exception("No points in neighborhood. Sigma may be too small")
        init_q = np.mean(self.data[:,TI],axis=1)
        return init_q[np.newaxis].T

    def projectPointsGetPoly(self, point, U0 = None, num_bias_removal_iter = 0):
        '''
        projectPointsGetPoly(point, U0 = None, num_bias_removal_iter = 0) - projects points on the manifold and return the local polynomial approximation for the manifold
        Input:
        point - an arrat of a single or multiple points to project (1d array or Dxk array where k is the number of points)
        U0 - Initial coordinate system. if None(default) initialized by SVD, if initSVD == True, and with qr otherwise.
        num_bias_removal_iter - number of iterations of "step 2" of the algorithm that removes the bias. By default no iterations are done.
        '''
        if self.sigma == None:
            #print("Warning: sigma not initialized - initializing with calculateSigma()")
            self.calculateSigmaFromPoint(point)
        if self.kd_tree == None:
            #print("Warning: No KD tree. Looking for neighbors one by one.")
            #print("currently not implemented")
            raise Exception("no KD tree")
        isArray = False
        if len(point.shape) == 1:
            point = np.array([point]).T
            isArray = True

        init_q = self.genInitQ(point)

        if self.recompute_sigma == True:
            self.calculateSigmaFromPoint(init_q)
        U, q, PX0, X0, _, _ = self.findLocalCoordinatesThresh( point, init_q, self.compactWeightFunc, U0, initSVD = self.initSVD)

        D, I = self.kd_tree.query(q, self.sigma) # work with neighbors according to the SIGMA_Q neighborhood (so we'll have enough)
        D = D[0]
        I = I[0]
        TI = I[D<np.inf]
        X0 = self.data[:,TI] - nlib.repmat(q, 1, len(TI)) 
        PX0 = np.dot(U.T, X0) 
        W = self.compactWeightFunc(np.linalg.norm(PX0, axis=0)) # Computing weights in the low dimensional space 

        coeffs, Base = self.weightedLeastSquares(PX0[:,W>self.EPSILON], W[W>self.EPSILON], X0[:,W>self.EPSILON], self.poly_deg)
        projected_r0 = self.evaluatePolynomialAtZero(coeffs, Base)

        projected_p = projected_r0 + q

        #Starting Step 2 of the algorithm.
        for it in range(num_bias_removal_iter):
            # Updating init and coordiante system (H in the manuscript)
            q = projected_p
            if len(Base.shape)>1:
                poly_differential = coeffs[np.sum(Base, axis=1) == 1,:].T
            else:
                poly_differential = coeffs[Base == 1,:].T
            U = LA.qr(poly_differential,mode='economic')[0]
            # Recomputing least squares.
            D, I = self.kd_tree.query(q, self.sigma) # work with neighbors according to the SIGMA_Q neighborhood (so we'll have enough)
            D = D[0]
            I = I[0]
            TI = I[D<np.inf]
            X0 = self.data[:,TI] - nlib.repmat(q, 1, len(TI)) 
            PX0 = np.dot(U.T, X0) 
            W = self.compactWeightFunc(np.linalg.norm(PX0, axis=0), k=0.9**it) # Computing weights in the low dimensional space 

            coeffs, Base = self.weightedLeastSquares(PX0[:,W>self.EPSILON], W[W>self.EPSILON], X0[:,W>self.EPSILON], self.poly_deg)
            projected_r0 = self.evaluatePolynomialAtZero(coeffs, Base)
            projected_p = projected_r0 + q

        #adding point to list of projected points
        self.approximated_points.append(point)
        self.approximated_points_res.append(projected_p)
        if isArray:
            return projected_p.T[0], q.T[0], coeffs, Base, U 
        else:
            return projected_p, q, coeffs, Base, U 
    
    def projectPoints(self, point, U0 = None):
         projected_p, _, _, _, _  = self.projectPointsGetPoly(point, U0 = U0)
         return projected_p
    
    def projectManyPoints(self, points, consecutive_data = True, print_every = 100):
        self.approximated_points = []
        self.approximated_points_res = []
        U0 = None
        t0 = _t()
        for p, i in zip(points.T, range(len(points.T))):
            if consecutive_data:
                _, _, _, _, U0 = self.projectPointsGetPoly(p, U0)
            else:
                _, _, _, _, _ = self.projectPointsGetPoly(p, U0)
            if i%print_every == print_every-1:
                print("projecting point number", i, "Av. proj. time is:", (_t()-t0)/(i+1))
        return np.asarray(self.approximated_points_res).T[0]

    def findLocalCoordinatesThresh(self, point, init_q, weight_func, U0 = None, initSVD = True, mx_iter=100):
        if len(init_q.shape) == 1:
            init_q = np.array([init_q]).T
        if len(point.shape) == 1:
            point = np.array([point]).T
        
        # get neighbors in the support
        D, I = self.kd_tree.query(init_q, self.sigma)
        W = weight_func(D[0])
        neighbors = I[0,W>0]
        if U0 is not None:
            U = U0
        elif initSVD:
            # perform weighted SVD as a first guess
            U, _, _, X0= self.weightedSVD(self.data, W, init_q)
            # choose a DIM dimensional basis
            U = U[:,:self.manifold_dim]
        else:
            U = LA.qr(np.random.rand(self.data.shape[0], self.manifold_dim),mode='economic')[0]
        q = (U@(U.T@(point - init_q))) + init_q 
        # get neighbors in the support
        D, I = self.kd_tree.query(q, self.sigma)
        W = weight_func(D[0])
        neighbors = I[0,W>0]
        X0 = self.data[:,neighbors] - nlib.repmat(q, 1, len(neighbors))
        # Projecting the points on the new coordinates
        PX0 = np.dot(U.T,X0)
        # Approximating the data on the new axes
        new_U, new_PX0, new_X0, new_W, new_q = U, PX0, X0, W[W>0], q
        # assign an initial value for former_q
        former_q = new_q + np.ones_like(new_q)
        itr = 0
        while (LA.norm(former_q - new_q) > self.thresholdH) and (itr < mx_iter):
            former_q = new_q
            R_tilde = new_X0 * np.sqrt(new_W)
            Xnd = R_tilde.T @ new_U
            X_tilde = np.ones(np.array(Xnd.shape) + [0,1])
            X_tilde[:,1:] = Xnd
            alpha = LA.solve(X_tilde.T @ X_tilde, X_tilde.T @ R_tilde.T)
            q_tilde = new_q + alpha[0,:][np.newaxis].T
            new_U = LA.qr(alpha[1:,:].T,mode='economic')[0]
            new_q = q_tilde + (new_U @ (new_U.T @ (point - q_tilde)))
            
            # get neighbors in the support
            D, I = self.kd_tree.query(new_q, self.sigma)
            new_W = weight_func(D[0])
            neighbors = I[0,new_W>0]
            new_W = new_W[new_W>0]
            new_X0 = self.data[:,neighbors] - new_q
            # projecting the points on the tangential space
            new_PX0 = new_U.T @ new_X0
            # advance the iteration (for debugging purposes)
            itr = itr+1
            #print(LA.norm(former_q - new_q)/LA.norm(former_q))
        #print "Local Coordinates number of iterations = ", itr
        if itr >= mx_iter:
            print("FindLocalCoordinatesThresh:: Exceeded max iterations (=", mx_iter, ")")
        return new_U, new_q, new_PX0, new_X0, new_W, itr

    def findLocalCoordinatesThresh_old(self, r, init_q, weight_func, U0 = None, initSVD = False, mx_iter=1000):

        if len(init_q.shape) == 1:
            init_q = np.array([init_q]).T
        if len(r.shape) == 1:
            r = np.array([r]).T
        # get neighbors in the support
        D, I = self.kd_tree.query(init_q, self.sigma)
        W = weight_func(D[0])
        if U0 is not None:
            U = U0
        elif initSVD:
            # perform weighted SVD as a first guess
            U, _, _, X0= self.weightedSVD(self.data, W, init_q)
            # choose a DIM dimensional basis
            U = U[:,:self.manifold_dim]
        else:
            U = LA.qr(np.random.rand(self.data.shape[0], self.manifold_dim),mode='economic')[0]
        q = np.dot(U, np.dot(U.T, r - init_q)) + init_q # q = U@U.T@(r - init_q) + init_q 
        # get neighbors in the support
        D, I = self.kd_tree.query(q, self.sigma)
        W = weight_func(D[0])
        neighbors = I[0,W>0]
        X0 = self.data[:,neighbors] - nlib.repmat(q, 1, len(neighbors))
        # Projecting the points on the new coordinates
        PX0 = np.dot(U.T,X0)
        # Approximating the data on the new axes
        new_U, new_PX0, new_X0, new_W, new_q = U, PX0, X0, W, q
        # assign an initial value for former_q
        former_q = new_q + np.ones_like(new_q)
        itr = 0
        while (LA.norm(former_q - new_q) > self.thresholdH) and (itr < mx_iter):
            former_q = new_q
            new_U, new_q = self.linearApproximation(new_U, new_PX0, new_X0, r, new_q, new_W[0:neighbors.shape[0]], self.manifold_dim)
            # get neighbors in the support
            D, I = self.kd_tree.query(new_q, self.sigma)
            new_W = weight_func(D[0])
            neighbors = I[0,new_W>0]
            new_X0 = self.data[:,neighbors] - nlib.repmat(new_q, 1, len(neighbors))
            # projecting the points on the tangential space
            new_PX0 = np.dot(new_U.T, new_X0)
            # advance the iteration (for debugging purposes)
            itr = itr+1
            #print LA.norm(former_q - new_q)
        #print "Local Coordinates number of iterations = ", itr
        if itr >= mx_iter:
            print("FindLocalCoordinatesThresh:: Exceeded max iterations (=", mx_iter, ")")
        return new_U, new_q, new_PX0, new_X0, new_W, itr
    
    def MMLSCurvatureTensor2D3D(self, U, PX0, X0, W, deg=2):
        '''
        The function assumes we are dealing with 2d surface in R^3.
        '''
        #dim = 2
        Normal = np.cross(U[:,0], U[:,1])
        y_data = np.dot(Normal.T, X0)
        coeffs, Base = self.weightedLeastSquares(PX0, W, y_data, self.poly_deg)
        Curv_tensor = np.zeros((np.int(nchoosek(self.manifold_dim, 2)+1), np.int(nchoosek(self.manifold_dim, 2)+1)))
        for c, b in zip(coeffs.T[0], Base):
            if sum(b) == 2:
                indices = np.arange(len(b))[np.array(b, dtype=bool)]
                if len(indices)>1:
                    Curv_tensor[indices[0], indices[1]] = c/2
                    Curv_tensor[indices[1], indices[0]] = c/2
                else:
                    Curv_tensor[indices[0], indices[0]] = c
        return Curv_tensor

    def weightedSVD(self, x_vec, W, q = None):
        '''expects x_vec to be an array with input vectors as column.
        q - The center around which we are working
        W - non-negative decreasing weights
        Returns - the principal components shifted around  0 as column vectors
        '''
        if q is None:
            q = np.zeros([x_vec.shape[0], 1])
        # verify that we have a 1XN array
        elif len(q.shape)<2:
            q = np.array([q]).T
        #dim = x_vec.shape[0]
        N = x_vec.shape[1]
        # deduct the center
        X = x_vec - nlib.repmat(q, 1, N)
        WXT = np.asarray(W[W>0] * X[:,W>0])
        U, S , Vh = LA.svd(WXT, full_matrices=False)
        return U, S , Vh, X
    
    def linearApproximation(self, U, PX0, X0, r, q, W, dim):
        coeffs, Base = self.weightedLeastSquares(PX0, W, X0, deg=1)
        x_vals = np.concatenate((np.asarray([np.zeros(dim)]).T, np.eye(dim)), axis=1)    
        new_z = self.evaluatePolynomial(coeffs, Base, x_vals)
        almost_p0 = new_z[:,0:1]
        p0 = almost_p0 + q
        almost_new_U = new_z[:,1:] - nlib.repmat(almost_p0, 1, dim)
        almost_new_U = almost_new_U / np.linalg.norm(almost_new_U, axis = 0)
        # now we wish to have an orthonormal basis for the tangential space
        new_U, _ = np.linalg.qr(np.asarray(almost_new_U), mode='reduced')
        new_q = np.dot(new_U, np.dot(new_U.T, r - p0)) + p0
        return new_U, new_q
    
    def createBase(self, dim, deg, intercept=True):
        ranges_l = []
        for i in range(dim):
            ranges_l.append(range(deg + 1))
        if dim > 1:
            M = np.meshgrid(*ranges_l)
            Base = zip(*[M[i][np.sum(M, axis=0) <= deg] for i in range(dim)])
            if not intercept:
                Base = np.array([tuple(x) for x in Base if x != tuple(np.zeros((1, dim))[0])])
                Base = zip(*Base)
        else:
            if intercept:
                Base = range(deg + 1)
            else:
                Base = range(1, deg + 1)
        Base = list(Base)
        Base.sort()
        return np.array(Base)

    def weightedLeastSquares(self, x_data, thetas, y_data, deg=2, print_cond=False, intercept=True):
        ''' x_data: is expected to be a matrix of column vectors
            thetas: is expected to be a row vector that assigns the weights for each column vector in x_data
            y_data: is expected to be a row vector that fits a scalar value to each row row in x_data
            (y_data can also be a matrix of different sampled data as column vectors - to save multiple calculations - i.e., each row corresponds to a scalar sample)

            RETURNS
            coeffs = [wi * sum(b(xi) b(xi)^T)]^-1 * f
            each column in coeffs is a vector of coefficients for the polynomial in the respective coordinate
            for example: coeffs.T[0] will be a vector in the size of b_xi such that they are the corresponding coefficients in the Basis for the first coordinate

            and

            Base = order of powers in the base

            These are the coeffs in the basis Base (the coeffs are ordered in column vectorrs - so for the first dimension the first column etc...
        '''
        isMultiple_f = (len(y_data.shape) > 1)  # a flag to indicate multi dimentional sample
        if len(x_data.shape) > 1:
            dim = x_data.shape[0]
            x_num = x_data.shape[1]
        else:
            dim = 1
            x_num = x_data.shape[0]
        # the weights for is sqrt since it apears in both sides of the outer product down below...
        # and as well it apears in the yi on the right hand side
        weights = np.sqrt(thetas)
        Base = self.createBase(dim, deg, intercept)
        B = None
        for x, w, i in zip(x_data.T, weights, range(x_num)):
            b_xi = [np.prod(x ** m) for m in Base]
            b_xi = np.asarray(b_xi) * w  # this is where I use the weights
            # the following yi are supposed either scalar values or vectors
            yi = y_data.T[i] * w  # this is where I use the weights
            if B is None:
                B = np.outer(b_xi, b_xi)
                if isMultiple_f:
                    f = np.array([y * b_xi for y in yi]).T
                else:
                    f = yi * b_xi
            else:
                B = B + np.outer(b_xi, b_xi)
                if isMultiple_f:
                    f = f + np.array([y * b_xi for y in yi]).T
                else:
                    f = f + yi * b_xi
        if print_cond: print("Cond = ", np.linalg.cond(B))
        if np.linalg.cond(B) < 1 / sys.float_info.epsilon:
            coeffs = LA.solve(B, f)
        else:
            print ("WARNING! Weighted least squares matrix is not invertible - Performing least squares")
            coeffs = LA.lstsq(B, f)[0]
        if not isMultiple_f:
            coeffs = np.asarray([coeffs])
            coeffs = coeffs.T
        return coeffs, Base
    def evaluatePolynomialAtZero(self, coeffs, Base):
        '''Base - is expected to be an array of tuples (if dim>1) or array of integers (if dim=1)
        the elements in Base represent the powers of the different elements
        For example: the tuple (1,2,0) (in dim=3) corresponds to the monom x^1*y^2*z^0

        coeffs - is expected to be an array of column vectors each at the size of the len(Base) so that it contains a coeff for each monom in the respective Basis

        --> returns the evaluation of the polynomial at 0
        NOTE: the output data will be a matrix (unless if the function is scalar valued we will return an array
        '''
        return coeffs[0:1,:].T
        
    def evaluatePolynomial(self, coeffs, Base, X):
        '''Base - is expected to be an array of tuples (if dim>1) or array of integers (if dim=1)
        the elements in Base represent the powers of the different elements
        For example: the tuple (1,2,0) (in dim=3) corresponds to the monom x^1*y^2*z^0

        coeffs - is expected to be an array of column vectors each at the size of the len(Base) so that it contains a coeff for each monom in the respective Basis
        X - are the values at which we wish to evaluate the polynomial' - should be a matrix with the column vectors

        --> returns the evaluation of the polynomial at each of the points (i.e., for each column in X)
        NOTE: the output data will be a matrix (unless if the function is scalar valued we will return an array      
        '''
        if len(X.shape)<=1:
            X = np.asarray([X])
        #dim = X.shape[0]
        X_num = X.shape[1]

        # it is easier to work with row vector arrays
        Xt = X.T
        coeffsT = coeffs.T
        evaluation = np.zeros([coeffsT.shape[0], X_num])
        for coeff, i in zip(coeffsT, range(coeffsT.shape[0])):
            for c, powers in zip(coeff,Base):
                evaluation[i] = evaluation[i] + c *( Xt ** powers).prod(axis=1)
        if coeffsT.shape[0]>1:
            return evaluation
        else:
            return evaluation[0]
    def genGeodesDirection(self, point, direction, num_of_steps, step_size_x, step_size_y = None, num_bias_removal_iter=0, verbose = False):
        '''
        gen_geodes_direction(self, point, direction) generates a set of points from a 
        point "point" on (or near) the manifold, and a direction "direction", that approximates a geodesic line on 
        the manifold from the point in the given direction.

        The method is an iterative process that: 
            1) moves a step in a direction and than project the new point on to the manifold (thus generating anew point).
            2) parallel transports the direction to the new point, to get the new direction.
        
        Input:
        point - Dx1 array (D is the dimension of the data)
        Direction - dx1 array (d is the manifold dimension)
        num_of_steps - the number of steps to talk.
        step_size_x - if step_size_y == 0 than step_size_x is the step size on the tangent for each iteration
        step_size_y - if given, insrtead of moving on the tangent in each iteration, the points moves long the local polynomial approxiamtion, and step_size_y is the pount on the move on the "y-axis". In this case step_size_x is the initial/upperbound to the step size
        '''
        projected_p = []
        
        if step_size_y==None: 
            move_with_polynomail_approx = False 
        else:
            move_with_polynomail_approx = True
        
        U = None
        tt = _t()
        for ind in range(0,num_of_steps):
            if verbose: print("Starting iteration:",ind, "time for iteration: ",_t() - tt)
            tt = _t()             
            #Projecting the point 
            xx, q, coeffs, Base,U  = self.projectPointsGetPoly(point, U0 = U, num_bias_removal_iter = num_bias_removal_iter)
            projected_p.append(xx)
            if ind == 0:
                direction_vec_proj = np.array([step_size_x])
            else:
                direction_vec = projected_p[-1] - projected_p[-2]
                direction_vec_proj = np.dot(U.T, direction_vec)
            direction_vec_proj = direction_vec_proj/np.linalg.norm(direction_vec_proj) * step_size_x
            if move_with_polynomail_approx == True:
                #calculating the new point which will be delta_x away
                projected_r0 = self.evaluatePolynomial(coeffs, Base, np.asarray([direction_vec_proj]).T)[:,0]
                while np.linalg.norm( projected_r0 - xx+q)> step_size_y:
                    direction_vec_proj = direction_vec_proj/2
                    if verbose: print('reducing step size to ', np.linalg.norm(direction_vec_proj), '    distance on y-axis:,', np.linalg.norm( projected_r0 - xx+q))
                    projected_r0 = self.evaluatePolynomial(coeffs, Base, np.asarray([direction_vec_proj]).T)[:,0]
                point = projected_r0 + q
            else:
                point = q + np.dot(U,direction_vec_proj)
            
        return projected_p

class MyKDTree():
    '''
 |  Parameters
 |  ----------
 |  data : (N,K) array_like
 |      The data points to be indexed. This array is not copied, and
 |      so modifying this data will result in bogus results.
 |      In this subclass (as opose to the parent) the dimension are rows and the number of points are the columns
 |  leafsize : int, optional
 |      The number of points at which the algorithm switches over to
 |      brute-force.  Has to be positive.
    '''
    def __init__(self, data, leafsize = 10):
        self.real_tree = cKDTree(data.T, leafsize)

    def query(self, q, radius):
        if len(q.shape) == 1:
            return self.real_tree.query(q,k=self.real_tree.n,distance_upper_bound=radius)
        else:
            return self.real_tree.query(q.T,k=self.real_tree.n,distance_upper_bound=radius)
    def query_k(self, q, k):
        if len(q.shape) == 1:
            return self.real_tree.query(q,k=k)
        else:
            return self.real_tree.query(q.T,k=k)

if __name__ == '__main__':
    pass

'''
Written by Barak Sober & Yariv Aizenbud. 2019.
If you are using this code please cite:
1) B. Sober and D. Levin. "Manifold Approximation by Moving Least-Squares Projection (MMLS)." 
2) Y. Aizenbud and B. Sober "Estimation of Manifolds from Noisy Samples"
3) B. Sober Y. Aizenbud and D. Levin. "Approximation of functions over manifolds: A moving least-squares approach." 
'''

###########
# Imports #
###########
import numpy as np
from . import ManApprox
import numpy.matlib as nlib
from time import time as _t
from scipy import linalg as LA
import sys
from time import time as _t
EPSILON = 1.e-12


###########
# Classes #
###########
class ManApproxF(ManApprox):
    """
    Function over Manifold approximation through MMLS: based upon scattered data
    Implements the algoritm described in:
    Sober, B., Aizenbud, Y., & Levin, D. (2021). Approximation of functions over manifolds: A moving least-squares approach. Journal of Computational and Applied Mathematics, 383, 113140.
    """
    def __init__(self, data=[], f_data = [], manifold_dim = None, sparse_factor = 5, sigma = None, poly_deg = 1, thresholdH = 10**-3, init_tree = True, calculate_sigma = True):
        super().__init__(data, manifold_dim, sparse_factor, sigma, poly_deg, thresholdH, init_tree, calculate_sigma)
        if not(f_data.shape[0] == data.shape[1]):
            raise "Different number of points in f_data and data "
        self.f_data = f_data
        self.f_approximated_points = []
        self.f_approximated_points_res = []

    def approximateAtPoints(self, point, U0=None, flag_project=False):
        approximated_p, _, _, _, _, projected_p = self.approximateAtPointsGetPoly(point,U0=U0, flag_project=flag_project)
        return projected_p, approximated_p
    
    
    
    
    def approximateMany(self, point_set, flag_project = False):
        '''
        func_value_arr = approximateMany(point_set, flag_project = False)
        Approximate multiple points in one function call
        '''
        self.f_approximated_points = []
        self.f_approximated_points_res = []
        U0 = None
        t0 = _t()
        for p, i in zip(point_set.T, range(len(point_set.T))):
            _, _, _, _, U0, _ = self.approximateAtPointsGetPoly(p, U0=U0, flag_project=flag_project)
            if i%10 == 0:
                print("\tprojectAllData:: projecting point number", i)
                print("\tAverage ptojection time is:", (_t()-t0)/(i+1))
        return np.asarray(self.f_approximated_points_res).T[0]

    def approximateAtPointsGetPoly(self, point, U0=None, flag_project=False):
        '''
        approximated_p, q, coeffs, Base, U, projected_p = approximateAtPointsGetPoly(point,U0=None, flag_project=False)
        Approximate a point, and return all approximation parameters
        '''
        q, X0, PX0, F0, W, U = self.getLocalChart(point, U0=U0)
        if F0.ndim == 1:
            F0_filtered = F0[W>self.EPSILON][np.newaxis]
        else:
            F0_filtered = F0[W>self.EPSILON].T
        coeffs, Base = self.weightedLeastSquares(PX0[:,W>self.EPSILON], W[W>self.EPSILON], F0_filtered, self.poly_deg)
        approximated_p = self.evaluatePolynomialAtZero(coeffs, Base)
        
        #adding point to list of projected points
        self.f_approximated_points.append(point)
        self.f_approximated_points_res.append(approximated_p)

        if flag_project:
            coeffs, Base = self.weightedLeastSquares(PX0[:,W>self.EPSILON], W[W>self.EPSILON], X0[:,W>self.EPSILON], self.poly_deg)
            projected_r0 = self.evaluatePolynomialAtZero(coeffs, Base)

            projected_p = projected_r0 + q

            #adding point to list of projected points
            self.approximated_points.append(point)
            self.approximated_points_res.append(projected_p)
        else:
            projected_p = None

        return approximated_p, q, coeffs, Base, U, projected_p
    
    def StandardMLS(self, point, flag_centered=False, recompute_sigma = False, k_weight = 1.5):
        '''
        Here we assume that there is no underlying manifold.
        '''
        if len(point.shape)==1:
            q = point[np.newaxis].T
        else:
            q = point
        if not flag_centered:
            PX0 = self.data - q
        else:
            PX0 = self.data
        if recompute_sigma:
            self.calculateSigmaFromPoint(q)
        D, I = self.kd_tree.query(q, self.sigma) # work with neighbors according to the SIGMA_Q neighborhood (so we'll have enough)
        D = D[0]
        I = I[0][D<np.inf]
        PX0 = PX0[:,I] 
        if len(self.f_data.shape) == 1:
            F0 = self.f_data[I][np.newaxis]
        else:
            F0 = self.f_data[:,I]
        W = self.compactWeightFunc(np.linalg.norm(PX0, axis=0), k_weight)
        coeffs, Base = self.weightedLeastSquares(PX0[:,W>self.EPSILON], W[W>self.EPSILON], F0[:,W>self.EPSILON], self.poly_deg)
        approximated_p = self.evaluatePolynomialAtZero(coeffs, Base)
        
        #adding point to list of projected points
        self.f_approximated_points.append(point)
        self.f_approximated_points_res.append(approximated_p)

        return approximated_p, coeffs, Base
    
    def StandardDeg1MLS(self, point, flag_centered=False, recompute_sigma = False, k_weight = 1.5):
        '''
        Here we assume that there is no underlying manifold.
        '''
        if len(point.shape)==1:
            q = point[np.newaxis].T
        else:
            q = point
        if not flag_centered:
            PX0 = self.data - q
        else:
            PX0 = self.data
        if recompute_sigma:
            self.calculateSigmaFromPoint(q)
        D, I = self.kd_tree.query(q, self.sigma) # work with neighbors according to the SIGMA_Q neighborhood (so we'll have enough)
        D = D[0]
        I = I[0][D<np.inf]
        PX0 = PX0[:,I] 
        if len(self.f_data.shape) == 1:
            F0 = self.f_data[I][np.newaxis]
        else:
            F0 = self.f_data[:,I]
        W = self.compactWeightFunc(np.linalg.norm(PX0, axis=0), k_weight)
        PX0 = PX0[:,W>self.EPSILON]
        W = W[W>self.EPSILON]
        F0 = F0[:,W>self.EPSILON]
        X = np.ones(PX0.shape + np.array([1,0])).T
        X[:,1:] = PX0.T
        A = np.dot(X.T*W,X)
        b = np.dot(X.T * W, F0.T)
        if np.linalg.cond(A) < 1 / sys.float_info.epsilon:
            coeffs = LA.solve(A, b)
        else:
            print ("WARNING! Weighted least squares matrix is not invertible")
            print ("Performing least squares")
            print (A)
            coeffs = LA.lstsq(A, b)
        approximated_p = self.evaluatePolynomialAtZero(coeffs, Base)
        
        #adding point to list of projected points
        self.f_approximated_points.append(point)
        self.f_approximated_points_res.append(approximated_p)

        return approximated_p, coeffs

    def approximateDerivative(self, point, U0=None, flag_project=False):
        _, _, PX0, F0, _, U = self.getLocalChart(point, U0=U0)
        _, Base, Jacobian_MLS = self.movingLeastSquaresJacobian(PX0, np.linalg.norm(PX0,axis=0), F0, deg=self.poly_deg)
        return Jacobian_MLS, Base, U
    

    def movingLeastSquaresJacobian(self, x_data, distances, y_data, deg=2, print_cond=False, intercept=True):
        '''
        Computes the gradient at zero.
        Returns the accurate gradient in the standard basis
        '''
        if len(x_data.shape) > 1:
            dim = x_data.shape[0]
            x_num = x_data.shape[1]
        else:
            dim = 1
            x_num = x_data.shape[0]

        Base = self.createBase(dim, deg, intercept)
        thetas = self.compactWeightFunc(distances)
        thetas_dot = self.compactWeightFuncDot(distances)
        A = np.zeros((len(Base),len(Base)))
        grad_A = np.zeros((dim,len(Base),len(Base)))
        B = np.zeros((len(Base),1)) 
        grad_B = np.zeros((dim,len(Base)))
        for x, w, w_dot, i in zip(x_data.T, thetas, thetas_dot, range(x_num)):
            if w < EPSILON and w_dot < EPSILON:
                continue
            b_xi = np.zeros((len(Base),1))
            for j,m in enumerate(Base):
                b_xi[j] = np.prod(x ** m)
            
            # the following yi are supposed either scalar values or vectors
            yi = y_data.T[i:i+1]
            A = A + w*np.outer(b_xi, b_xi)
            B = B + w*yi*b_xi
            for d in range(dim):
                grad_A[d,:,:] = grad_A[d,:,:] - 2*x[d]*w_dot*np.outer(b_xi, b_xi)
                grad_B[d,:] = grad_B[d,:] - 2*x[d]*w_dot*yi*b_xi.T[0]
        if print_cond: print("Cond = ", np.linalg.cond(B))

        if np.linalg.cond(A) <= 1 / sys.float_info.epsilon:
            A_inv = LA.inv(A)
            A_inv_B = np.dot(A_inv, B)
            coeffs = A_inv_B
        else:
            print ("Error! Weighted least squares matrix is not invertible")
            print (A)
            assert False
        term_I   = np.zeros((dim,B.shape[1]))
        term_II  = np.zeros((dim,B.shape[1]))
        term_III = np.zeros((dim,B.shape[1]))
        for d in range(dim):
            # build the derivative base
            ed = np.zeros((1,Base.shape[1]))
            ed[0,d] = 1
            relevant_indices = np.arange(Base.shape[0])[Base[:,d]>0]
            deriv_Base = Base[relevant_indices,:]
            deriv_Base = deriv_Base - ed
            deriv_coeff = Base[relevant_indices,d]
            real_indices = []
            for b in deriv_Base:
                real_indices.append(np.arange(len(Base))[np.prod(Base == b,axis=1)==1][0])
            real_indices = np.array(real_indices)
            deriv_real_coeff = np.zeros((len(Base),1))
            deriv_real_coeff[real_indices,:] = deriv_coeff[np.newaxis].T
            # Now the terms of the gradient - we need only the constant term since we are around zero.
            term_I[d,:]   = A_inv_B[relevant_indices[0]]#np.dot(deriv_real_coeff.T, A_inv_B)
            term_II[d,:]  = - np.dot(A_inv, np.dot(grad_A[d,:,:], A_inv_B))[0]
            term_III[d,:] = np.dot(A_inv, grad_B[d,:])[0]
        # and the grad is
        Jacobian_MLS = term_I + term_II + term_III

        return coeffs, Base, Jacobian_MLS
    
    def approximateJacobian(self, x_data, distances, y_data, deg=2, print_cond=False, intercept=True):
        '''
        Computes the Jacobian at zero!!!
        Returns the approximate gradient in the standard basis
        This method is much faster than MovingLeastSquaresJacobian and more stable numerically
        However, it gives an approximate to the original function's Jacobian and not the MLS' Jacobian
        '''
        if len(x_data.shape) > 1:
            dim = x_data.shape[0]
            x_num = x_data.shape[1]
        else:
            dim = 1
            x_num = x_data.shape[0]

        Base = self.createBase(dim, deg, intercept)
        thetas = self.compactWeightFunc(distances)
        A = np.zeros((len(Base),len(Base)))
        B = np.zeros((len(Base),1)) 
        for x, w, i in zip(x_data.T, thetas, range(x_num)):
            if w < EPSILON:
                continue
            b_xi = np.zeros((len(Base),1))
            for j,m in enumerate(Base):
                b_xi[j] = np.prod(x ** m)
            
            # the following yi are supposed either scalar values or vectors
            yi = y_data.T[i:i+1]
            A = A + w*np.outer(b_xi, b_xi)
            B = B + w*yi*b_xi
        if print_cond: print("Cond = ", np.linalg.cond(B))

        if np.linalg.cond(A) <= 1 / sys.float_info.epsilon:
            coeffs =  LA.solve(A, B)
        else:
            print ("Error! Weighted least squares matrix is not invertible")
            print (A)
            assert False
        Jacobian = np.zeros((dim,B.shape[1]))
        for d in range(dim):
            # build the derivative base
            ed = np.zeros((1,Base.shape[1]))
            ed[0,d] = 1
            relevant_indices = np.arange(Base.shape[0])[Base[:,d]>0]
            # and the grad is
            Jacobian[d,:] = coeffs[relevant_indices[0]]
        return coeffs, Base, Jacobian
    
    def getLocalChart(self, point, U0=None):
        """Find the local coordinate chart to perform approximations"""
        if self.sigma == None:
            print("Warning: sigma not initialized - initializing with calculateSigma()")
            self.calculateSigmaFromPoint(point)
        if self.kd_tree == None:
            print("Warning: No KD tree. Looking for neighbors one by one.")
            print("currently not implemented")
            raise Exception("no KD tree")

        init_q = self.genInitQ(point)

        if self.recompute_sigma == True:
            self.calculateSigmaFromPoint(init_q)
        U, q, PX0, X0, W, _ = self.findLocalCoordinatesThresh( point, init_q, self.compactWeightFunc, U0, initSVD = self.initSVD)
        D, I = self.kd_tree.query(q, self.sigma) # work with neighbors according to the SIGMA_Q neighborhood (so we'll have enough)
        D = D[0]
        I = I[0][D<np.inf]
        X0 = self.data[:,I] - nlib.repmat(q, 1, len(I))
        PX0 = np.dot(U.T, X0)
        F0 = self.f_data[I]
        W = self.compactWeightFunc(np.linalg.norm(PX0, axis=0))
        return q, X0, PX0, F0, W, U


if __name__ == "__main__":
    pass
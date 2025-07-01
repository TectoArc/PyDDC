import numpy as np
from numba import jit
import V
import random_field as field
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree
# import cvxpy as cp
import cvxopt
from cvxopt import matrix, solvers 

class helpers:
    @staticmethod
    def reflection(plst, bdr=None, axis:str=None):
        # define matrices for reflection along y and x axis
        ry = np.array([[-1, 0], [0, 1]]) 
        rx = np.array([[1, 0], [0, -1]])       
        
        wts, loc = plst[:, 0], plst[:, 1:]
        if bdr is None:
            raise Exception("Boundary must be mentioned for performing reflection")
        elif axis =="x":
            slst = (bdr.reshape(-1, 1) + (rx @ (loc.T - bdr.reshape(-1, 1)))).T
        elif axis == "y":
            slst = (bdr.reshape(-1, 1) +(ry @ (loc.T - bdr.reshape(-1, 1)))).T
        slst = np.stack((wts, slst[:, 0], slst[:, 1]), axis=1)
        return slst
        
    @staticmethod
    def interpolation(pts, vals, ppos, type="vec"):
        if type == "vec":
            interp = LinearNDInterpolator(pts, vals.ravel())    
        elif type == "scalar":
            interp = NearestNDInterpolator(pts, vals.ravel())    
        return interp(ppos).reshape(-1, 1)
    
    @staticmethod
    def resample(plst, w0, max_iter=50, tol=1e-8):
        TC = len(plst)/V.DSF # Net target particle count
        w0 = w0.reshape(-1, 1)
        tree = cKDTree(V.En_int)
        _, ii = tree.query(plst, k=1)
        ii_unq, p2g = np.unique(ii, return_inverse=True)

        w = np.empty([0, 1])
        Ln = np.empty([0, 2])

        for i in range(len(ii_unq)):
            p_id = np.where(p2g==i)[0]
            TargetBinCount = int(TC * w0[p_id].sum()/w0.sum()) # make target distribution proportional to initial distribution
            if np.count_nonzero(w0[p_id]) <TargetBinCount:
                w = np.concatenate((w, w0[p_id])); Ln = np.concatenate((Ln, plst[p_id]))
                continue
            if TargetBinCount >=25:
                if TargetBinCount >= len(plst[p_id]):
                    TargetBinCount = len(plst[p_id]) - 1  # avoid singularity in case of too few particles
 
                id = np.random.choice(len(w0[p_id]), size=TargetBinCount, replace=False, p=(w0[p_id]/w0[p_id].sum()).ravel())
                M = plst[p_id][id]
                A = np.vstack([
                    np.ones(M.shape[0]),
                    M[:, 0], 
                    M[:, 1]
                ])
                W = np.sum(w0[p_id])
                S = np.sum(plst[p_id]*w0[p_id], axis=0)/W
                b = np.array([W, W*S[0], W*S[1]]).reshape(-1, 1)
                
                w_iter = 1/M.shape[0]*np.ones(M.shape[0])*w0[p_id].sum()
                converged = False
                for _ in range(max_iter):
                    w_iter = w_iter.reshape(-1, 1)
                    w_iter = np.maximum(w_iter, 0)  # ensure non-negativity
                    lamda = np.linalg.solve(A@A.T, A@w_iter-b)

                    w_iter -= A.T@lamda
                    if np.all(w_iter>-1e-10) and np.linalg.norm(A@w_iter-b) < tol:
                        w = np.concatenate((w, w_iter)); Ln = np.concatenate((Ln, M))
                        converged = True
                        break
                if not converged:
                    w = np.concatenate((w, w0[p_id])); Ln = np.concatenate((Ln, plst[p_id])) 
            else:
                w = np.concatenate((w, w0[p_id])); Ln = np.concatenate((Ln, plst[p_id])) 
        w = np.maximum(w, 0)  # ensure non-negativity
        P = np.stack([w.ravel(), Ln[:, 0], Ln[:, 1]], axis=1)
        weight_zero_ids = np.where(P[:, 0]==0)
        P = np.delete(P, weight_zero_ids, axis=0)
        return P

class PT:
    def __init__(self, c_sat, phi, dt):
        self.dt = dt
        self.phi = phi
        self.c = lambda mu: mu/(V.dv[1:-1, 1:-1]*self.phi[1:-1, 1:-1])
        self.c_sat = c_sat

    def particle_injection(self, c):
        phi = self.phi[-2, 1:-1][V.dirichlet_dofs]
        J = -V.D*phi*(c[-2, 1:-1][V.dirichlet_dofs] - self.c_sat)/V.dy[-2][0]*2
        Np = (J*phi*V.dx[1:-1][V.dirichlet_dofs].T*self.dt/V.mpp).astype("int64").flatten()

        S = np.empty([0, 2]) 
        for j, i in enumerate(V.dirichlet_dofs):
            if Np[j] == 0:
                continue
            px = np.random.uniform(V.xf[1:-1][i], V.xf[1:-1][i+1], Np[j])
            py = np.random.normal(V.H, np.sqrt(2*V.D*phi[j]*self.dt), Np[j])            
            pxy = np.array(list(zip(px, py)))
            S = np.concatenate((S, pxy))

        w0 = np.ones(S.shape[0])*V.mpp
        P = np.stack([w0, S[:, 0], S[:, 1]], axis=1)
        out = P[:, 2]> V.H
        P[out] = helpers.reflection(P[out], np.array(V.H), "x")
        return P, J

    def diffusion(self, plst): # considering effective diffusion coefficient D/phi
        D_dx = np.random.normal(0, np.sqrt(2*V.D*self.phi[1:-1, 1:-1]*self.dt))
        D_dy = np.random.normal(0, np.sqrt(2*V.D*self.phi[1:-1, 1:-1]*self.dt))
       
        return np.vstack([helpers.interpolation(V.En_int, D_dx.ravel(), plst, type="scalar").ravel(), 
                          helpers.interpolation(V.En_int, D_dy.ravel(), plst, type="scalar").ravel()]).T    
        

    def binned_counts(self, plst):
        mu_w, _, _ = np.histogram2d(plst[:, 1], plst[:, 2], bins=(V.xf[1:-1], V.yf[1:-1]), density=False)
        return mu_w.T
    
    def binned_concentration(self, plst):
        mu_w, _, _ = np.histogram2d(plst[:, 1], plst[:, 2], bins=(V.xf[1:-1], V.yf[1:-1]), density=False, weights=plst[:, 0])
        return mu_w.T
    
    def disperse(self, A, B, plst):
        Ai = np.stack([helpers.interpolation(V.Ef, A[:, i], plst).ravel() for i in range(2)], axis=1)
        Bi = np.stack([helpers.interpolation(V.En, B[:, :, i].ravel(), plst) for i in range(4)], axis=2).reshape(-1, 2, 2)
        phi_p = helpers.interpolation(V.En, self.phi, plst).ravel()
        N = np.random.normal(0, 1, size=plst.shape)
        dx = 1/phi_p*Ai[:, 0]*self.dt + np.einsum('ij, ij->i', Bi[:, :, 0], N) * np.sqrt(self.dt)
        dy = 1/phi_p*Ai[:, 1]*self.dt + np.einsum('ij, ij->i', Bi[:, :, 1], N) * np.sqrt(self.dt)
        return np.stack((dx, dy), axis=-1)

    def apply_diffusion_bcs(self, plst): 
        db = np.where((plst[:, 2]>V.H) & ((plst[:, 1]>=V.extent[0]) & (plst[:, 1]<=V.extent[1])))
        nb = ((plst[:, 2]>V.H) & ((plst[:, 1]<V.extent[0]) | (plst[:, 1]>V.extent[1]))) 
        rb = plst[:, 1]>V.L
        lb = plst[:, 1]<0.
        bb = plst[:, 2] < 0.
        plst[nb] = helpers.reflection(plst[nb], np.array(V.H), "x")
        plst[rb] = helpers.reflection(plst[rb], np.array(V.L), "y")
        plst[lb] = helpers.reflection(plst[lb], np.array(0.), "y")
        plst[bb] = helpers.reflection(plst[bb], np.array(0.), "x")
        plst = np.delete(plst, db, axis=0)
        return plst
    
    def apply_dispersion_bcs(self, plst):
        db = np.where((plst[:, 1]>V.L) | (plst[:, 1]<0.))
        nb_top = (plst[:, 2]>V.H) 
        nb_bottom = (plst[:, 2] < 0.) 
        plst[nb_top] = helpers.reflection(plst[nb_top], np.array(V.H), "x")
        plst[nb_bottom] = helpers.reflection(plst[nb_bottom], np.array(0.), "x")
        tot_mass = plst[db][:, 0].sum()
        plst = np.delete(plst, db, axis=0)
        return plst, tot_mass



        
        
            
                    

         
        
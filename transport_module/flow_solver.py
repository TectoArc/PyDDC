import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import bicg, splu
import formulate.V as V

class FlowSolver:
    def __init__(self, field:str=None):
        self.p = np.zeros([V.ny+2, V.nx+2])
        self.vorticity = np.zeros([V.ny, V.nx])
        self.k = field[:, :]
        self.qx = np.zeros([V.ny+2, V.nx+3]); self.qy = np.zeros([V.ny+3, V.nx+2])

        self.kf_n, self.kf_s, self.kf_e, self.kf_w = self.__facet_permeability_interp() # gets k-field facet values

        self.__coefficient_container() # initialize to borrow coefficients from the container when required
        try:
            if isinstance(V.mu, float) | isinstance(V.mu, int): # assemble the stationary pressure coefficient matrix for inversion
                self._AssemblePressureCoefficientMatrix_cv() 
        except:
            pass

    def __coefficient_container(self):
        '''
        precomputes all the terms in assembling coefficient matrices that are invariant for reducing computation time  
        '''

        self.An = 2/(V.dy[1:-1]+V.dy[2:])*(V.dy[1:-1]/V.dy[2:]-V.gn*(V.dy[1:-1]/V.dy[2:] - V.dy[2:]/V.dy[1:-1]))*self.kf_n*V.dx[1:-1].T
        self.As = 2/(V.dy[1:-1]+V.dy[:-2])*(V.dy[1:-1]/V.dy[:-2]-V.gs*(V.dy[1:-1]/V.dy[:-2] - V.dy[:-2]/V.dy[1:-1]))*self.kf_s*V.dx[1:-1].T
        self.Ae = (2/(V.dx[1:-1]+V.dx[2:])*(V.dx[1:-1]/V.dx[2:]-V.ge*(V.dx[1:-1]/V.dx[2:] - V.dx[2:]/V.dx[1:-1]))).T*self.kf_e*V.dy[1:-1]
        self.Aw = (2/(V.dx[1:-1]+V.dx[:-2])*(V.dx[1:-1]/V.dx[:-2]-V.gw*(V.dx[1:-1]/V.dx[:-2] - V.dx[:-2]/V.dx[1:-1]))).T*self.kf_w*V.dy[1:-1]
        self.As[0, :] = 0.0; self.An[-1, :] = 0.0; self.Ae[:, -1] = 0.0; self.Aw[:, 0] = 0.0

        self.Ap_adj_e = 2.0 * (2.0/(V.dx[-2] + V.dx[-1])*(V.dx[-2]/V.dx[-1] - V.ge[-1]*(V.dx[-2]/V.dx[-1] - V.dx[-1]/V.dx[-2]))).T*self.kf_e[:,-1]*V.dy[1:-1].reshape(-1, )
        self.Ap_adj_w = 2.0 * (2.0/(V.dx[0] + V.dx[1])*(V.dx[1]/V.dx[0] - V.gw[0]*(V.dx[1]/V.dx[0] - V.dx[0]/V.dx[1]))).T*self.kf_w[:, 0]*V.dy[1:-1].reshape(-1, )
        
        self.rhs_e = (2.0/(V.dx[-1] + V.dx[-2])*(V.dx[-2]/V.dx[-1] - V.ge[-1]*(V.dx[-2]/V.dx[-1] - V.dx[-1]/V.dx[-2]))).T*2.0*self.kf_e[:, -1]*V.dy[1:-1].reshape(-1, ) #*self.PR
        self.rhs_w = (2.0/(V.dx[0] + V.dx[1])*(V.dx[1]/V.dx[0] - V.gw[0]*(V.dx[1]/V.dx[0] - V.dx[0]/V.dx[1]))).T*2.0*self.kf_w[:, 0]*V.dy[1:-1].reshape(-1, ) # *self.PL
        self.rhs_n = self.kf_n[-1, :]*V.g*V.dx[-2].T
        self.rhs_s = self.kf_s[0, :]*V.g*V.dx[-2].T 

        self.coeff_n = (V.dy[-2]+V.dy[-1])/(2.0 * (V.dy[-2]/V.dy[-1] - V.gn[-1]*(V.dy[-2]/V.dy[-1] - V.dy[-1]/V.dy[-2])))*V.g
        self.coeff_s = (V.dy[0]+V.dy[1])/(2.0 * (V.dy[1]/V.dy[0] - V.gs[0]*(V.dy[1]/V.dy[0] - V.dy[0]/V.dy[1])))*V.g

    def _pressure_DirichletBC(self, r): 
        PL = (V.P+V.pil)*1e6 + (V.H-V.y[1:-1])*r[1:-1, 1]*V.g
        PR = (V.P+V.pir)*1e6 + (V.H-V.y[1:-1])*r[1:-1, -2]*V.g # left and right boundary conditions for pressure 
        return PL, PR

    def _AssemblePressureCoefficientMatrix_cv(self):
        Ap = np.zeros([V.ny, V.nx])
        Ap[:, :] = -(self.An + self.As + self.Ae + self.Aw)
        Ap[:, -1] -= self.Ap_adj_e ; Ap[:, 0] -= self.Ap_adj_w

        d0 = Ap.reshape(V.ny*V.nx)
        de = self.Ae.reshape(V.ny*V.nx)[:-1]
        dw = self.Aw.reshape(V.ny*V.nx)[1:]
        ds = self.As.reshape(V.ny*V.nx)[V.nx:]
        dn = self.An.reshape(V.ny*V.nx)[:-V.nx]

        CMAT = diags([d0, de, dw, dn, ds], [0, 1, -1, V.nx, -V.nx], format='csc')
        self.LU = splu(CMAT)

    def _AssembleCoefficientMatrix_rhs(self, r, PL, PR):
        # this function is to perform the assembly of the rhs for constant viscosity case
        rhs = (self.kf_s*V.gs*r[:-2, 1:-1] + r[1:-1, 1:-1]*(self.kf_s*(1.0-V.gs) - self.kf_n*(1.0-V.gn)) - self.kf_n*V.gn*r[2:, 1:-1])*V.g*V.dx[1:-1].T
        rhs[:, -1] -= self.rhs_e*PR
        rhs[:, 0] -= self.rhs_w*PL
        rhs[-1, :] += self.rhs_n*(V.gn[-1]*r[-1, 1:-1] + (1-V.gn[-1])*r[-2, 1:-1])
        rhs[0, :] -= self.rhs_s*(V.gs[0]*r[0, 1:-1] + (1-V.gs[0])*r[1, 1:-1])
        return rhs

    def _GlobalCoefficientMatrix_vv(self, r, mu, PL, PR):
        mu_n, mu_s, mu_e, mu_w = self.__facet_viscosity_interp(mu)

        Ap = np.zeros([V.ny, V.nx])
        Ap[:, :] = -(self.An/mu_n + self.As/mu_s + self.Ae/mu_e + self.Aw/mu_w)
        Ap[:, -1] -= self.Ap_adj_e/mu_e[:, -1] ; Ap[:, 0] -= self.Ap_adj_w/mu_w[:, 0]
    
        d0 = Ap.reshape(V.ny*V.nx)
        de = (self.Ae/mu_e).reshape(V.ny*V.nx)[:-1]
        dw = (self.Aw/mu_w).reshape(V.ny*V.nx)[1:]
        ds = (self.As/mu_s).reshape(V.ny*V.nx)[V.nx:]
        dn = (self.An/mu_n).reshape(V.ny*V.nx)[:-V.nx]

        CMAT = diags([d0, de, dw, dn, ds], [0, 1, -1, V.nx, -V.nx], format='csc')
    
        rhs = (self.kf_s/mu_s*V.gs*r[:-2, 1:-1] + r[1:-1, 1:-1]*(self.kf_s/mu_s*(1.0-V.gs) - self.kf_n/mu_n*(1.0-V.gn)) - self.kf_n/mu_n*V.gn*r[2:, 1:-1])*V.g*V.dx[1:-1].T

        rhs[:, -1] -= self.rhs_e*PR/mu_e[:, -1]
        rhs[:, 0]  -= self.rhs_w*PL/mu_w[:, 0]
        rhs[-1, :] += self.rhs_n/mu_n[-1, :]*(V.gn[-1]*r[-1, 1:-1] + (1-V.gn[-1])*r[-2, 1:-1])
        rhs[0, :] -= self.rhs_s/mu_s[0, :]*(V.gs[0]*r[0, 1:-1] + (1-V.gs[0])*r[1, 1:-1])

        return CMAT, rhs

    def solve(self, r, mu=None):
        PL, PR = self._pressure_DirichletBC(r)
        if isinstance(mu, np.ndarray):
            CMAT, rhs = self._GlobalCoefficientMatrix_vv(r, mu, PL, PR)
            pt, _ = bicg(CMAT, rhs.ravel(), x0=self.p[1:-1, 1:-1].ravel(), tol=1e-10)
        else:
            rhs = self._AssembleCoefficientMatrix_rhs(r, PL, PR)
            pt = self.LU.solve(rhs.ravel()) # precomputed from the pressire coefficient matrix
        
        self.p[1:-1, 1:-1] = pt.reshape([V.ny, V.nx])
        self.p[1:-1, -1] = 2.0*PR - self.p[1:-1, -2]
        self.p[1:-1, 0] = 2.0*PL -  self.p[1:-1, 1]

        self.p[-1, 1:-1] = self.p[-2, 1:-1] - self.coeff_n * (V.gn[-1]*r[-1, 1:-1] + (1-V.gn[-1])*r[-2, 1:-1])
        self.p[0, 1:-1] = self.p[1, 1:-1] + self.coeff_s * (V.gs[0]*r[0, 1:-1] + (1-V.gs[0])*r[1, 1:-1])
        
        kh = (V.dx[1:] + V.dx[:-1]).T / (V.dx[1:].T/self.k[1:-1, 1:] + V.dx[:-1].T/self.k[1:-1, :-1])
        kv = (V.dy[1:] + V.dy[:-1]) / (V.dy[1:]/self.k[1:, 1:-1] + V.dy[:-1]/self.k[:-1, 1:-1])

        if isinstance(mu, np.ndarray):
            mu_h = V.gx.T*mu[1:-1, 1:] + (1-V.gx.T)*mu[1:-1, :-1]
            mu_v = V.gy*mu[1:, 1:-1] + (1-V.gy)*mu[:-1, 1:-1]
            self.qx[1:-1, 1:-1] = -kh/mu_h*(self.p[1:-1, 1:] - self.p[1:-1, 0:-1])*2.0/(V.dx[:-1]+V.dx[1:]).T
            self.qy[1:-1, 1:-1] = -kv/mu_v*((self.p[1:, 1:-1] - self.p[0:-1, 1:-1])*2.0/(V.dy[:-1]+V.dy[1:]) + (V.gy*r[1:, 1:-1] + (1-V.gy)*r[:-1, 1:-1])*V.g)
        else:
            self.qx[1:-1, 1:-1] = -kh/V.mu*(self.p[1:-1, 1:] - self.p[1:-1, 0:-1])*2.0/(V.dx[:-1]+V.dx[1:]).T
            self.qy[1:-1, 1:-1] = -kv/V.mu*((self.p[1:, 1:-1] - self.p[0:-1, 1:-1])*2.0/(V.dy[:-1]+V.dy[1:]) + (V.gy*r[1:, 1:-1] + (1-V.gy)*r[:-1, 1:-1])*V.g)
        # flux boundary conditions     
        self.qy[-2, :] = 0.0; self.qy[1, :] = 0.0
        self.qx[:, 0] = 2.0*self.qx[:, 1] - self.qx[:, 2]
        self.qx[:, -1] = 2.0*self.qx[:, -2] - self.qx[:, -3]
        self.qx[-1, :] = self.qx[-2, :]; self.qx[0, :] = self.qx[1, :]
        self.qy[0, :] = -self.qy[1, :]; self.qy[-1, :] = -self.qy[-2, :]

        self.Q = np.zeros([V.ny*V.nx, 2])
        self.Q[:, 0] = 0.5*(self.qx[1:-1, 2:-1] + self.qx[1:-1, 1:-2]).ravel()
        self.Q[:, 1] = 0.5*(self.qy[2:-1, 1:-1] + self.qy[1:-2, 1:-1]).ravel() 
        self.vorticity[:, :] = -self.k[1:-1, 1:-1]/mu[1:-1, 1:-1]*(r[1:-1, 2:] - r[1:-1, 1:-1])/(V.dx[2:].T/2 + V.dx[1:-1].T/2)*V.g

    def __facet_permeability_interp(self):
        kf_n = (V.dy[1:-1] + V.dy[2:]) / (V.dy[1:-1]/self.k[1:-1, 1:-1] + V.dy[2:]/self.k[2:, 1:-1]) 
        kf_s = (V.dy[1:-1] + V.dy[:-2]) / (V.dy[1:-1]/self.k[1:-1, 1:-1] + V.dy[:-2]/self.k[:-2, 1:-1])
        kf_e = (V.dx[1:-1] + V.dx[2:]).T / (V.dx[1:-1].T/self.k[1:-1, 1:-1] + V.dx[2:].T/self.k[1:-1, 2:])
        kf_w = (V.dx[1:-1] + V.dx[:-2]).T / (V.dx[1:-1].T/self.k[1:-1, 1:-1] + V.dx[:-2].T/self.k[1:-1, :-2])
        return kf_n, kf_s, kf_e, kf_w
    
    def __facet_viscosity_interp(self, mu):
        mu_n = V.gn*mu[2:, 1:-1] + (1-V.gn)*mu[1:-1, 1:-1]
        mu_s = V.gs*mu[:-2, 1:-1] + (1-V.gs)*mu[1:-1, 1:-1]
        mu_e = V.ge.T*mu[1:-1, 2:] + (1-V.ge.T)*mu[1:-1, 1:-1]
        mu_w = V.gw.T*mu[1:-1, :-2] + (1-V.gw.T)*mu[1:-1, 1:-1]
        return mu_n, mu_s, mu_e, mu_w

    def CellwiseDispersion(self):
        # normal components at the cell centers and shear components at the cell vertices
        eps = 1e-20
        Dxx = np.zeros([V.ny+2, V.nx+2]); Dyy = np.zeros_like(Dxx)
        Dxy = np.zeros([V.ny+1, V.nx+1]); Dyx = np.zeros_like(Dxy)
        
        Dxx[:, :] = (V.al*self.qx[:, 1:]**2 + V.at*self.qy[1:, :]**2) / (np.sqrt(self.qx[:, 1:]**2 + self.qy[1:, :]**2)+eps)
        Dyy[:, :] = (V.at*self.qx[:, 1:]**2 + V.al*self.qy[1:, :]**2) / (np.sqrt(self.qx[:, 1:]**2 + self.qy[1:, :]**2)+eps)
        Dxy[:, :] = (V.al - V.at)*self.qx[:-1, 1:-1]**2*self.qy[1:-1, :-1] / (np.sqrt(self.qx[:-1, 1:-1]**2 + self.qy[1:-1, :-1]**2)+eps)
        Dyx = Dxy[:, :]

        return Dxx, Dxy, Dyx, Dyy

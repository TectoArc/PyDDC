import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import formulate.V as V
import formulate.TH as TH

class CO2BrineViscosity:
    def __init__(self, rho, mco2):
        self.T = V.T + 273.15 # Kelvin
        self.rho = rho
        self.mco2 = mco2

    def h2OViscosity(self):
        # reference physical quantities
        Tr = 647.096 #K
        Rr = 322.0 # kg/m3
        mur = 1e-6 # Pa s

        T_ = self.T/Tr; rho_ = self.rho/Rr; 
        mu0 = 100*np.sqrt(T_) / np.sum(TH.VM.Hi/T_)

        lc_vec = np.array([(1/T_-1.)**i for i in range(0, 5+1)])

        if isinstance(self.rho, np.ndarray) and self.rho.ndim==2:
            rc_vec = np.empty([self.rho.shape[0], self.rho.shape[1], 7])
            for j in range(7):
                rc_vec[:, :, j] = (rho_-1)**j

            mu1 = np.exp(rho_*np.einsum('ijk, k->ij', np.einsum('ijk,lk->ijl', rc_vec, TH.VM.Hij), lc_vec))

        elif isinstance(self.rho, np.ndarray) and self.rho.ndim==1:
            rc_vec = np.empty([len(self.rho), 7])
            for j in range(7):
                rc_vec[:, j] = (rho_-1)**j

            mu1 = np.exp(rho_*np.einsum('ij, j->i', np.einsum('ij,kj->ik', rc_vec, TH.VM.Hij), lc_vec))
        
        else:
            rc_vec = np.array([(rho_-1.)**j for j in range(0, 7)])
            mu1 = np.exp(rho_*(lc_vec.T@(TH.VM.Hij@rc_vec.T)))
        
        mu_ = mu0*mu1 # non-dimensional viscosity
        mu_w = mu_*mur 
        return mu_w

    def ComputeMixtureViscosity(self, X):
        mu_w = self.h2OViscosity()
        nacl = TH.VM.gf[0]; co2 = TH.VM.gf[1] 
        T_vec = np.array([1, self.T, self.T**2])

        Ei = np.array([np.array([nacl[0], nacl[1], nacl[2]]) @ T_vec.T, np.array([co2[0], co2[1], co2[2]]) @ T_vec.T])
        Vi = np.array([np.array([nacl[3], nacl[4], nacl[5]]) @ T_vec.T, np.array([co2[3], co2[4], co2[5]]) @ T_vec.T])

        if X.xNaCl.ndim == 2:
            Xi = np.zeros([X.xNaCl.shape[0], X.xNaCl.shape[1], 2])
            Xi[:, :, 0] = X.xNaCl; Xi[:, :, 1] = X.xCO2 
        elif X.xNaCl.ndim == 1:
            Xi = np.zeros([len(X.xNaCl), 2])
            Xi[:, 0] = X.xNaCl; Xi[:, 1] = X.xCO2 
        else:
            Xi = np.array([X.xNaCl, X.xCO2])
        mur = np.exp(np.sum(Xi*Ei, axis=-1))/(1.0 + np.sum(Xi*Vi, axis=-1)) # relative viscosity
        return mur*mu_w, mu_w
         




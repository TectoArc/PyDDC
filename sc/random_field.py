#!/bin/python3
import gstools as gs
import numpy as np
from scipy.optimize import root
import V

def Kozeny_Carman(x, k, d):
    return x**3 - k*2.5*(1-x)**2*(6/d)**2

class Field:
    @staticmethod 
    def KField(x, y, mean_k, var, corr_length):
        model = gs.Exponential(dim=2, var=var, len_scale=corr_length)
        srf = gs.SRF(model, mean=0.)
        field = srf((x, y), mesh_type='structured')
        field = (field - np.mean(field)) / np.std(field) + np.log(mean_k)
        return np.exp(field).T

    @staticmethod
    def generate_heterogeneity_field(x, y, mean_k, var, corr_length):
        k = Field.KField(x, y, mean_k, var, corr_length)
        phi = Field.PHIField(k)
        return k, phi 

    @staticmethod 
    def PHIField(kf):
        """
        create a random porosity field from the random permeability field based on the Kozeny-Carman relation 
        """
        phi = np.zeros([V.ny, V.nx])
        for i in range(V.ny):
            for j in range(V.nx):
                phi[i, j] = root(Kozeny_Carman, 1.0, args=(kf[i, j], V.d)).x
        return phi
    
    @staticmethod
    def compute_field_variables():
        kf, phif = np.zeros([V.ny+2, V.nx+2]), np.zeros([V.ny+2, V.nx+2])

        if V.lnk_var == 0:
            if V.k_mean is not None: # homogeneous random field
                kf[:, :] = V.k_mean
            else:
                raise Exception("Mean value of permeability must be supplied for generating homogeneous fields")
        else: 
            kf[1:-1, 1:-1] = Field.KField(V.x[1:-1], V.y[1:-1], V.k_mean, V.lnk_var, V.k_corr)
            kf[0, :] = kf[1, :]; kf[-1, :] = kf[-2, :]
            kf[:, -1] = kf[:, -2]; kf[:, 0] = kf[:, 1]  
        
        if V.phi is not None:
            phif[:, :] = V.phi
        else:
            phif[1:-1, 1:-1] = Field.PHIField(kf[1:-1, 1:-1])
            phif[0, :] = phif[1, :]; phif[-1, :] = phif[-2, :]
            phif[:, -1] = phif[:, -2]; phif[:, 0] = phif[:, 1]

        return kf, phif

    @staticmethod
    def compute_field_gradient(f):
        # computes the gradient of any scalar field: 2nd Order accurate
        f_xe, f_ye = Field.FacetFieldInterpolator(f)
        fx = (2/V.dx[1:].T - 2/(V.dx[1:]+V.dx[:-1]).T)*f[:, 1:] + (2/(V.dx[1:]+V.dx[:-1]).T - 2/V.dx[:-1].T)*f[:, :-1] +\
                (2/V.dx[:-1].T - 2/V.dx[1:].T)*f_xe
        fy = (2/V.dy[1:] - 2/(V.dy[1:]+V.dy[:-1]))*f[1:, :] + (2/(V.dy[1:]+V.dy[:-1]) - 2/V.dy[:-1])*f[:-1, :] +\
                (2/V.dy[:-1] - 2/V.dy[1:])*f_ye
        
        return fx, fy
    
    @staticmethod
    def compute_field_divergence(f):
       # computes the divergence of any vector field v: 2nd order accurate
        f_x = (f[1:-1, 1:] - f[1:-1, :-1])/V.dx.T
        f_y = (f[1:, 1:-1] - f[:-1, 1:-1])/V.dy
        return f_x + f_y 
    
    @staticmethod
    def FacetFieldInterpolator(f):
        fny = V.gy*f[1:, :] + (1-V.gy)*f[:-1, :]
        fnx = V.gx.T*f[:, 1:] + (1-V.gx.T)*f[:, :-1]
        return fnx, fny 
    

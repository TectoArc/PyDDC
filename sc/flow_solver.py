import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import bicg, splu
import V
import random_field as field
from particle_tracking import helpers

class FlowSolver:
    def __init__(self, k, phi):
        self.p = np.zeros([V.ny+2, V.nx+2])
        self.k = k; self.phi = phi
        self.qx = np.zeros([V.ny+2, V.nx+3]); self.qy = np.zeros([V.ny+3, V.nx+2])

        self.kf_x, self.kf_y = self.__facet_permeability_interp()
        
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

        self.Ae = self.kf_x[:, 1:]*V.dy[1:-1]/V.dc_x[1:].T
        self.Aw = self.kf_x[:, :-1]*V.dy[1:-1]/V.dc_x[:-1].T
        self.An = self.kf_y[1:, :]*V.dx[1:-1].T/V.dc_y[1:]
        self.As = self.kf_y[:-1, :]*V.dx[1:-1].T/V.dc_y[:-1]

        self.Ae_bdr = 2.0*self.Ae[:, -1]; self.Aw_bdr = 2.0*self.Aw[:, 0]
        self.As[0, :] = 0.0; self.An[-1, :] = 0.0; self.Ae[:, -1] = 0.0; self.Aw[:, 0] = 0.0

        self.B_n = self.kf_y[1:, :]*V.g*V.dx[1:-1].T
        self.B_s = self.kf_y[:-1, :]*V.g*V.dx[1:-1].T
        self.B_e = (self.kf_x[:, -1]*V.dy[1:-1].T/V.dc_x[-1]).ravel()
        self.B_w = (self.kf_x[:, 0]*V.dy[1:-1].T/V.dc_x[0]).ravel()

    def _pressure_DirichletBC(self, r, mu): 
        Delta_P = mu[1:-1, 0]*V.qb*V.L/self.k[1:-1, 0] # k.mean()
        PL = V.P*1e6 + Delta_P + (V.H-V.y[1:-1])*r[1:-1, 1]*V.g
        PR = V.P*1e6 + (V.H-V.y[1:-1])*r[1:-1, -2]*V.g # left and right boundary conditions for pressure 
        return PL, PR

    def _AssembleCoefficientMatrix_rhs(self, rho_y, mu_x, mu_y, PL, PR):

        b =  self.B_s/mu_y[:-1, :]*rho_y[:-1, :] - self.B_n/mu_y[1:, :]*rho_y[1:, :]
        b[:, -1] -= self.B_e/mu_x[:, -1]*2.0*PR
        b[:, 0] -= self.B_w/mu_x[:, 0]*2.0*PL

        b[-1, :] += rho_y[-1, :]*self.B_n[-1, :]/mu_y[-1, :]
        b[0, :] -= rho_y[0, :]*self.B_s[0, :]/mu_y[0, :]

        return b

    def _GlobalCoefficientMatrix_vv(self, rho_y, mu_x, mu_y, PL, PR):

        Ap = np.zeros([V.ny, V.nx])
        Ap[:, :] = -(self.An/mu_y[1:, :] + self.As/mu_y[:-1, :] + self.Ae/mu_x[:, 1:] + self.Aw/mu_x[:, :-1])
        Ap[:, -1] -= self.Ae_bdr/mu_x[:, -1]; Ap[:, 0] -= self.Aw_bdr/mu_x[:, 0]

        d0 = Ap.reshape(V.ny*V.nx)
        de = (self.Ae/mu_x[:, 1:]).reshape(V.ny*V.nx)[:-1]
        dw = (self.Aw/mu_x[:, :-1]).reshape(V.ny*V.nx)[1:]
        ds = (self.As/mu_y[:-1, :]).reshape(V.ny*V.nx)[V.nx:]
        dn = (self.An/mu_y[1:, :]).reshape(V.ny*V.nx)[:-V.nx]

        A = diags([d0, de, dw, dn, ds], [0, 1, -1, V.nx, -V.nx], format='csc')

        b = self._AssembleCoefficientMatrix_rhs(rho_y, mu_x, mu_y, PL, PR)

        return A, b

    def solve(self, r, mu=None):
        PL, PR = self._pressure_DirichletBC(r, mu)
    
        rho_f, mu_f = list(map(helpers.interpolation, [V.En, V.En], [r.ravel(), mu.ravel()], [V.Ef, V.Ef]))
        rho_xy, mu_xy = list(map(self.reshape_arr, [rho_f, mu_f]))
        _, rho_y = rho_xy; mu_x, mu_y = mu_xy


        CMAT, rhs = self._GlobalCoefficientMatrix_vv(rho_y, mu_x, mu_y, PL, PR)
        pt, _ = bicg(CMAT, rhs.ravel(), x0=self.p[1:-1, 1:-1].ravel(), rtol=1e-10)

        
        self.p[1:-1, 1:-1] = pt.reshape([V.ny, V.nx])
        self.p[1:-1, -1] = 2.0*PR - self.p[1:-1, -2]
        self.p[1:-1, 0] = 2.0*PL -  self.p[1:-1, 1]

        self.p[-1, 1:-1] = self.p[-2, 1:-1] - rho_y[-1, :]*V.g*V.dc_y[-1]
        self.p[0, 1:-1] = self.p[1, 1:-1] + rho_y[0, :]*V.g*V.dc_y[0]

        self.qx[1:-1, 1:-1] = -self.kf_x/mu_x*(self.p[1:-1, 1:] - self.p[1:-1, :-1])/V.dc_x.T
        self.qy[1:-1, 1:-1] = -self.kf_y/mu_y*((self.p[1:, 1:-1] - self.p[:-1, 1:-1])/V.dc_y + rho_y*V.g)
        
        #  flux boundary conditions     
        self.qy[-2, :] = 0.0; self.qy[1, :] = 0.0
        self.qx[:, 0] = 2.0*self.qx[:, 1] - self.qx[:, 2]
        self.qx[:, -1] = 2.0*self.qx[:, -2] - self.qx[:, -3]
        self.qx[-1, :] = self.qx[-2, :]; self.qx[0, :] = self.qx[1, :]
        self.qy[0, :] = -self.qy[2, :]; self.qy[-1, :] = -self.qy[-3, :]
        self.qy[:, 0] = self.qy[:, 1]; self.qy[:, -1] = self.qy[:, -2]

        self.Q = np.zeros([(V.ny+2)*(V.nx+2), 2])
        # Local porewater velocity
        self.Qx = 1/self.phi*0.5*(self.qx[:, 1:] + self.qx[:, :-1])
        self.Qy = 1/self.phi*0.5*(self.qy[1:, :] + self.qy[:-1, :])
        self.Q[:, 0] = self.Qx.ravel(); self.Q[:, 1] = self.Qy.ravel() 

    def __facet_permeability_interp(self):
        kf_x = (V.dx[1:]/2 + V.dx[:-1]/2).T/(V.dx[1:].T/self.k[1:-1, 1:] + V.dx[:-1].T/self.k[1:-1, :-1])
        kf_y = (V.dy[1:]/2 + V.dy[:-1]/2)/(V.dy[1:]/self.k[1:, 1:-1] + V.dy[:-1]/self.k[:-1, 1:-1])
        return kf_x, kf_y

    def reshape_arr(self, arr):
        arr = arr.reshape(V.ny+1, V.nx+1)
        arr_x = (arr[1:, :] + arr[:-1, :])/2
        arr_y = (arr[:, 1:] + arr[:, :-1])/2
        return arr_x, arr_y
        
    def CellwiseDispersion(self):
        vx, vy = self.Qx, self.Qy
        Q_norm = np.sqrt(vx**2 + vy**2)
        Dxx = (V.al-V.at) * self.Qx**2/Q_norm + V.at*Q_norm; Dyy = (V.al-V.at) * self.Qy**2/Q_norm + V.at*Q_norm
        Dxy = (V.al-V.at) * self.Qx*self.Qy/Q_norm; Dyx = Dxy
        D = np.zeros([V.ny+2, V.nx+2, 4])
        D[:, :, 0] = Dxx; D[:, :, 1] = Dxy; D[:, :, 2] = Dyx; D[:, :, 3] = Dyy
        return D
    
    def _DivD(self, D):
        Dxx, Dxy, Dyx, Dyy = D[:, :, 0], D[:, :, 1], D[:, :, 2], D[:, :, 3]
        dx_Dxx = (Dxx[:, 1:] - Dxx[:, :-1])/(V.dx[:-1]/2+V.dx[1:]/2).T
        dy_Dxy = (Dxy[1:, :] - Dxy[:-1, :])/(V.dy[1:]/2+V.dy[:-1]/2)
        dx_Dyx = (Dyx[:, 1:] - Dyx[:, :-1])/(V.dx[:-1]/2+V.dx[1:]/2).T
        dy_Dyy = (Dyy[1:, :] - Dyy[:-1, :])/(V.dy[1:]/2+V.dy[:-1]/2)
        DivD_x = dx_Dxx[:-1, :]+dy_Dxy[:, :-1]; DivD_y = dx_Dyx[:-1, :]+dy_Dyy[:, :-1]
        return DivD_x, DivD_y
    
    def _div_DP(self, D, phi):
        phi_x, phi_y = field.Field.compute_field_gradient(phi)
        phi_x = (phi_x[1:, :] + phi_x[:-1, :]) / 2
        phi_y = (phi_y[:, 1:] + phi_y[:, :-1]) / 2

        Di = np.stack([helpers.interpolation(V.En, D[:, :, i].ravel(), V.Ef).reshape(phi_x.shape[0], phi_x.shape[1]) for i in range(4)], axis=2)
        return Di[:, :, 0]*phi_x + Di[:, :, 1]*phi_y, Di[:, :, 2]*phi_x+Di[:, :, 3]*phi_y
    
    def CellwiseDisplacementMatrix(self):
        B = np.zeros([V.ny+2, V.nx+2, 4])
        Q_norm = np.linalg.norm(self.Q, axis=1).reshape(V.ny+2, V.nx+2)
        fac = np.sqrt(2*(V.al*Q_norm))
        B[:, :, 0] = self.Qx/Q_norm*fac; B[:, :, 1] = -self.Qy/Q_norm*fac
        B[:, :, 2] = self.Qy/Q_norm*fac; B[:, :, 3] = -self.Qx/Q_norm*fac
        return B
    
    def DeterministicDrift(self):
        D = self.CellwiseDispersion()
        divD_x, divD_y = self._DivD(D)
        DP_x, DP_y  = self._div_DP(D, self.phi)
        vx = (self.qx[1:, 1:-1] + self.qx[:-1, 1:-1])/2; vy = (self.qy[1:-1, 1:] + self.qy[1:-1, :-1])/2

        Ax = vx + divD_x + DP_x
        Ay = vy + divD_y + DP_y
        A = np.vstack([Ax.ravel(), Ay.ravel()]).T

        return A, D
    
    def _globalCFL(self, D, phi):
        qx = self.Q[:, 0].reshape([V.ny+2, V.nx+2]); qy = self.Q[:, 1].reshape([V.ny+2, V.nx+2])
        vx = qx[1:-1, 1:-1]; vy = qy[1:-1, 1:-1]
        dt_a = 0.5*np.minimum(V.dx[1:-1].T/abs(vx), V.dy[1:-1]/abs(vy))
        dt_d = 0.1 * np.minimum(V.dx[1:-1].T**2/abs(D[1:-1, 1:-1, 0] + V.D*phi), V.dy[1:-1]**2/abs(D[1:-1, 1:-1, -1] + V.D*phi))
        return np.minimum(dt_a.min(), dt_d.min())

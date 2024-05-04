import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import formulate.V as V
import formulate.TH as TH 

class Density:
    def __init__(self, P, T):
        self.T = T + 273.15 # in Kelvin
        self.P = P*10 # in bars 
        self.P0 = 1.01325 # reference pressure (in bars)
        self.R = 83.1447 # cc bar mol^-1 K^-1
        # molecular weights of the different species are required to compute the solution density form apparent molar volumes.
        self.Mw = 18.01528 #g/mol
        self.Ms = 58.44 #g/mol
        self.Mco2 = 44.01 #g/mol

    def _volumetricDHlimitingslope(self):
        rw = self._iapws97()
        D1000 = TH.DM.U[0]*np.exp(TH.DM.U[1]*self.T + TH.DM.U[2]*self.T**2)
        C = TH.DM.U[3] + TH.DM.U[4]/(TH.DM.U[5]+self.T)
        B = TH.DM.U[6] + TH.DM.U[7]/self.T + TH.DM.U[8]*self.T
        
        lnfunc = (B+self.P)/(B+1000)
        D = D1000 + C*np.log(lnfunc) # Dielectric constant of h20 (non-dimensional). 

        N0 = 6.0331415e23 # Avogadro Constant
        e = 1.60217733e-19 # Charge on Electron
        k = 1.3806505e-23 # Boltzman Constant
        e0 = 8.854e-12 # Permittivity of free space

        A_phi = 1/3*(2.0*np.pi*N0*rw)**0.5 * (e**2/(4*np.pi*e0*D*k*self.T))**1.5 # osmotic coefficient
        D_p = C/(B+self.P)
        Av = 6*A_phi*self.R*self.T*(1/D*D_p) # volumetric Debye-Huckel limiting slope
        return Av

    def _iapws97(self): 
        R = 0.461526 # specific gas constant -> kJ kg^-1 K^-1
        pi = self.P/10/16.53; g = 1386./self.T # non-dimensional pressure and temperature 
        # spv: specific volume = 1/density
        if isinstance(self.P, np.ndarray) and self.P.ndim == 2:
            spv = R*self.T/(self.P/10) * pi*(-np.sum(TH.DM.iapws_n*TH.DM.iapws_i*(7.1-pi[:, :, np.newaxis])**(TH.DM.iapws_i-1) * (g-1.222)**(TH.DM.iapws_j), axis=2))
        elif isinstance(self.P, np.ndarray) and self.P.ndim == 1:
            spv = R*self.T/(self.P/10) * pi*(-np.sum(TH.DM.iapws_n*TH.DM.iapws_i*(7.1-pi[:, np.newaxis])**(TH.DM.iapws_i-1) * (g-1.222)**(TH.DM.iapws_j), axis=1))
        else:
            spv = R*self.T/(self.P/10) * pi*(-np.sum(TH.DM.iapws_n*TH.DM.iapws_i*(7.1-pi)**(TH.DM.iapws_i-1) * (g-1.222)**(TH.DM.iapws_j)))
        return 1e3/spv

    def _ApparentMolarVolumeSalt(self):
        '''
        estimate the molar volume of co2-h20-nacl using Rogers & Pitzer (1982);
        '''
        rw = self._iapws97()

        Y = 10
        mr = 1000/Y/self.Mw

        Vmr = TH.DM.pc[0] + TH.DM.pc[1]*self.T + TH.DM.pc[2]*self.T**2 + TH.DM.pc[3]*self.T**3 + \
            (self.P-self.P0)*(TH.DM.pc[4] + TH.DM.pc[5]*self.T + TH.DM.pc[6]*self.T**2) + (self.P-self.P0)**2*(TH.DM.pc[7] + TH.DM.pc[8]*self.T)

        B = TH.DM.pc[9] + TH.DM.pc[10]/(self.T-227) + TH.DM.pc[11]*self.T + TH.DM.pc[12]*self.T**2 + TH.DM.pc[13]/(680-self.T) + \
            (self.P-self.P0)*(TH.DM.pc[14] + TH.DM.pc[15]/(self.T-227) + TH.DM.pc[16]*self.T + TH.DM.pc[17]*self.T**2 +\
            TH.DM.pc[18]/(680-self.T)) + (self.P-self.P0)**2*(TH.DM.pc[19] + TH.DM.pc[20]/(self.T-227) + TH.DM.pc[21]*self.T + TH.DM.pc[22]/(680-self.T))

        C = 0.5 * (TH.DM.pc[23] + TH.DM.pc[24]/(self.T-227) + TH.DM.pc[25]*self.T + TH.DM.pc[26]*self.T**2 + TH.DM.pc[27]/(680-self.T))
        b = 1.2
        I = lambda m: 1/2 * (m*1**2 + m*(-1)**2)
        h = lambda x: np.log(1+b*np.sqrt(x))/(2*b)

        vm = 1; vx = 1; zm = 1; zx = -1 # for NaCl <=> Na^+ + Cl^-
        v = vm+vx
        Av = self._volumetricDHlimitingslope()

        V_sol = V.m*(Vmr/mr + 1e6/rw*(1/V.m - 1/mr) + v*abs(zm*zx)*Av*(h(I(V.m)) - h(I(mr))) -\
                  2.0*vm*vx*self.R*self.T*((V.m-mr)*B + (V.m**2-mr**2)*vm*zm*C))
        return (V_sol - 1e6/rw)/V.m
    
    def _h2OMolarVolume(self):
        iT_vec = np.array([self.T**3, self.T**2, self.T, 1, 1/self.T])
        jT_vec = np.array([self.T**3, self.T**2, 1])
        K = np.array([TH.DM.K0 @ iT_vec.T, TH.DM.K1 @ iT_vec.T, TH.DM.K2 @ jT_vec.T, TH.DM.K3 @ jT_vec.T])

        if isinstance(self.P, np.ndarray) and self.P.ndim == 2:
            P_vec = np.empty([self.P.shape[0], self.P.shape[1], 4])
            P_vec[:,:,0] = 1.0; P_vec[:, :, 1] = self.P; P_vec[:, :, 2] = self.P**2; P_vec[:, :, 3] = self.P**3
            Vw = np.tensordot(K, P_vec, axes=([0], [-1]))

        elif isinstance(self.P, np.ndarray) and self.P.ndim == 1:
            P_vec = np.empty([len(self.P), 4])
            P_vec[:, 0] = 1.0; P_vec[:, 1] = self.P; P_vec[:, 2] = self.P**2; P_vec[:, 3] = self.P**3
            Vw = np.tensordot(K, P_vec, axes=([0], [-1]))

        else:
            P_vec = np.array([1.0, self.P, self.P**2, self.P**3])
            Vw = K @ P_vec.T

        return Vw
    
    def ComputeBrineDensity(self): 
        if V.m != 0:
            rw = self._iapws97()
            sv = self._ApparentMolarVolumeSalt()*V.m + 1e6/rw # solution volume V(m)
            return 1e3*(1000 + V.m*self.Ms)/sv
        else:
            return self._iapws97()

    def ComputeCO2BrineDensity(self, X):
        T_vec = np.array([self.T**2, self.T, 1, 1/self.T, 1/self.T**2])
        A = np.array([TH.DM.Aij[0] @ T_vec.T, TH.DM.Aij[1] @ T_vec.T])
        V_w = self._h2OMolarVolume()
        V_co2 = V_w*(1 + A[0] + A[1]*self.P/10)

        if V.m != 0: # co2-brine density
            V_salt = self._ApparentMolarVolumeSalt()
            Vs = X.xH2O*V_w + X.xCO2*V_co2 + X.xNaCl*V_salt
            RHO = (X.xH2O*self.Mw + X.xNaCl*self.Ms + X.xCO2*self.Mco2)/Vs
        else: # co2-water density
            Vs = X.xH2O*V_w + X.xCO2*V_co2 
            RHO = (X.xH2O*self.Mw + X.xCO2*self.Mco2)/Vs

        return np.round(1e3*RHO, 2) # kg/m3

    
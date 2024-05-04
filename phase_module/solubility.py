import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import formulate.V as V
import formulate.TH as TH

class Solubility:
    def __init__(self, P, T):

        self.T = T+273.15 # Temeprature in Kelvin
        self.P = P*10 # Pressure in bars

        self.P0 = 16.2086 - 12.1147/(1. + np.exp(0.049635*V.T - 2.8034))
            
        self.R = 83.1447 #cc bar mol^-1 K^-1

    def CO2ActivityCoefficient(self):
        lamda = TH.SM.lamdaA*self.T + TH.SM.lamdaB + TH.SM.lamdaC/self.T
        mNa, mCl = V.m, V.m
        return np.exp((lamda + TH.SM.zeta*mCl)*mNa)

    def CO2Solubility(self):
        a_coeff = self.CO2ActivityCoefficient()
            
        X = lambda x: x[0] + x[1]*V.T + x[2]*V.T**2 + x[3]*V.T**3 + x[4]/V.T + x[5]*np.log(V.T)

        c = np.apply_along_axis(X, 1, TH.SM.chbp)   

        dmc = 2.0*c[0]*self.P0 + c[1] + c[2]*\
            (np.sin(np.pi/2*self.P0/(c[4]*self.P0+1)) + np.pi/2.*self.P0/(c[4]*self.P0+1)**2 * np.cos(np.pi/2*self.P0/(c[4]*self.P0+1)))
        
        mco2_P0 = c[0]*self.P0**2 + c[1]*self.P0 + c[2]*self.P0*np.sin(np.pi/2*self.P0/(c[4]*self.P0+1)) + c[5]*np.log(self.P0+c[6]**2) - c[5]*np.log(c[6]**2)
        if self.P > self.P0:
            mco2_h2o = mco2_P0 + dmc*(self.P/10-self.P0) + 1/c[3]*c[5]/(self.P0+c[6]**2)*V.P**c[3]/self.P0**(c[3]-1) - c[5]*self.P0/((self.P0+c[6]**2)*c[3])
            return mco2_h2o/a_coeff
        else:
            return mco2_P0/a_coeff


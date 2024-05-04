import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import formulate.V as V

class ComputeDiffussionCoefficent:
    @staticmethod
    def co2diffusivity(rho):
        Ms = 58.44 #g/mol
        C = V.m*rho/(V.m*Ms+1000)
        D0 = 1e-9
        return (-18.157948*np.exp(-0.05736*C) + 0.068700205361*(V.T+273.15) -0.0003876102395346346*C**0.820561458*(V.T+273.15)**1.46331515077)*D0
    
        
        
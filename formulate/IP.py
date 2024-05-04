import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import formulate.V as V
import formulate.TH as TH
import matplotlib.pyplot as plt

def _MeshRefinement(type):
    # yf and xf referes to the facet coordinates of the mesh
    xf = np.linspace(0, V.L, V.nx+1, endpoint="True")
    yf = np.linspace(0, V.H, V.ny+1, endpoint="True")
    dy = V.H/V.ny
    num_cells = 3
    fac = 100

    if type=="refined":
        yf = np.unique(np.concatenate((yf[:-num_cells], np.arange(yf[-num_cells], V.H, dy/fac))))
        yf = np.append(yf, V.H)

        while True:
            dy = yf[1:] - yf[:-1]
            idx_dy = np.where(dy[:-1]!=dy[1:])[0][0]
            if dy[idx_dy+1]/dy[idx_dy] < 0.5:  
                fac = dy[idx_dy+1]*2
                yt = np.linspace(yf[idx_dy], yf[idx_dy+1], int(1/fac)+2)
                yf = np.unique(np.concatenate((yf, yt)))
            else:
                break

    V.xf = np.zeros(len(xf)+2); V.yf = np.zeros(len(yf)+2)
    V.xf[1:-1] = xf[:]; V.yf[1:-1] = yf[:]
    V.xf[0] = xf[0] - xf[1]; V.xf[-1] = xf[-1] + (xf[-1] - xf[-2])
    V.yf[0] = yf[0] - yf[1]; V.yf[-1] = yf[-1] + (yf[-1] - yf[-2])

    V.y = (V.yf[1:] + V.yf[:-1])/2.0
    V.x = (V.xf[1:] + V.xf[:-1])/2.0

    xxf1, xxf2 = np.meshgrid(V.xf[1:-1], V.y)
    V.Ex = np.vstack([xxf1.ravel(), xxf2.ravel()]).T
    yyf1, yyf2 = np.meshgrid(V.x, V.yf[1:-1])
    V.Ey = np.vstack([yyf1.ravel(), yyf2.ravel()]).T

    V.dx = (V.xf[1:] - V.xf[:-1]).reshape(-1, 1)
    V.dy = (V.yf[1:] - V.yf[:-1]).reshape(-1, 1)
    V.dv = np.outer(V.dy, V.dx)
 
    V.xx, V.yy = np.meshgrid(V.x, V.y)
    V.En = np.vstack([V.xx.ravel(), V.yy.ravel()]).T # Eulerian nodes
    V.En_int = np.vstack([V.xx[1:-1, 1:-1].ravel(), V.yy[1:-1, 1:-1].ravel()]).T
 
    # facet weights to be used for linear interpolation
    V.gy = V.dy[:-1]/(V.dy[1:]+V.dy[:-1])
    V.gx = V.dx[:-1]/(V.dx[1:]+V.dx[:-1])

    V.gn = V.dy[1:-1]/(V.dy[2:]+V.dy[1:-1])
    V.gs = V.dy[1:-1]/(V.dy[:-2]+V.dy[1:-1])
    V.ge = V.dx[1:-1]/(V.dx[2:]+V.dx[1:-1])
    V.gw = V.dx[1:-1]/(V.dx[:-2]+V.dx[1:-1])

    # if save_mesh:
    #     fig, ax = plt.subplots()
    #     ax.hlines(V.yf, V.xf.min(), V.xf.max(), linewidth=0.5, color='k')
    #     ax.vlines(V.xf, V.yf.min(), V.yf.max(), linewidth=0.5, color='k')
    #     ax.set_xlim([V.xf.min(), V.xf.max()])
    #     ax.set_ylim([V.yf.min(), V.yf.max()])
    #     ax.set_ylabel(r"$y$")
    #     ax.set_xlabel(r"$x$")
    #     ax.set_facecolor("whitesmoke")
    #     fig.savefig(type + ".png")

def InitializeThermodynamicParams():
    '''
    TH.DM: Attributes for the Density Model
    1.Duan et al.: Densities of the CO2-H2O and CO2-H2O-NaCl systems up to 647 K and 100 MPa, Energy & Fuels, 22, 2008.
    2.Hu et al.: PVTx properties of the CO2-H2O and CO2-H2O-NaCl systems below 647 K: Assessment of experimental data and thermodynamic models, Chemical Geology, 238, 2007.
    3.Rogers and Pitzer: Volumetric properties of aqueous sodium chloride solutions, Journal of Physical and Chemical Reference Data, 11, 1982.
    4.Wagner and Kretzschmar: IAPWS industrial formulation 1997 for the thermo-dynamic properties of water and steam, International steam tables: properties of water and steam based on
        the industrial formulation IAPWS-IF97, 2008.
    5.Bradley and Pitzer: Dielectric properties of water and Debye-Hueckel parameters to 350. degree. C and 1 kbar, Journal of Physical Chemistry, 83, 1979.  
    '''
    #5.
    TH.DM.U = np.array([3.4279e2, -5.0866e-3, 9.4690e-7, -2.0525, 3.1159e3, -1.8289e2, -8.0325e3, 4.2142e6, 2.1417]) 
    #3.
    TH.DM.pc = np.array([1.0249125e3, 2.7796679e-1, -3.0203919e-4, 1.4977178e-6, -7.2002329e-2, 3.1453130e-4, -5.9795994e-7, -6.6596010e-6,
                         3.0407621e-8, 5.3699517e-5, 2.2020163e-3, -2.6538013e-7, 8.6255554e-10, -2.6829310e-2, -1.1173488e-7, -2.6249802e-7, 
                         3.4926500e-10, -8.3571924e-13, 3.0669940e-5, 1.9767979e-11, -1.9144105e-10, 3.1387857e-14, -9.6461948e-9, 2.2902837e-5, 
                         -4.3314252e-4, -9.0550901e-8, 8.6926600e-11, 5.1904777e-4])
    
    #1.
    TH.DM.K0 = np.array([3.27225e-7, -4.20950e-4, 2.32594e-1, -4.16920e1, 5.71292e3]) 
    TH.DM.K1 = np.array([-2.32306e-10, 2.91138e-7, -1.49662e-4, 3.59860e-2, -3.55071])
    TH.DM.K2 = np.array([2.57241e-14, -1.24336e-11, 5.42707e-7])
    TH.DM.K3 = np.array([-4.42028e-18, 2.10007e-15, -8.11491e-11])
    
    TH.DM.Aij = np.array([[0.38384020e-3, -0.55953850, 0.30429268e3, -0.72044305e5, 0.63003388e7],
                          [-0.57709332e-5, 0.82764653e-2, -0.43813556e1, 0.10144907e4, -0.86777045e5]])
    #4.
    TH.DM.iapws_i = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 
                            29, 30, 31, 32])

    TH.DM.iapws_j = np.array([-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4, 0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, 
                              -39, -40, -41])
    TH.DM.iapws_n = np.array([0.14633, -0.84548, -3.7564, 3.3855, -0.95792, 0.15772, -1.6616e-2, 8.1215e-4, 2.8319e-4, -6.0706e-4, -1.8990e-2, 
                              -3.25297e-2, -2.1842e-2, -5.2838e-5, -4.7184e-4, -3.00e-4, 4.7661e-5, -4.4142e-6, -7.2695e-16, -3.1679e-5, -2.8270e-6, 
                              -8.5205e-10, -2.2425e-6, -6.5171e-7, -0.14342e-12, -4.0517e-7, -1.2734e-9, -1.7425e-10, -6.8762e-19, 1.4478e-20, 
                              2.6336e-23, -1.1928e-23, 1.8228e-24, -9.3537e-26])
    
    '''
    TH.SM: Attributes of the Solubility Model
    1. Sun, Xiaohui and Wang, Zhiyuan and Li, Hangyu and He, Haikang and Sun, Baojiang.: A simple model for the prediction of mutual solubility in CO2-brine system at geological conditions, Desalination, 504, 2021.
    '''
    TH.SM.lamdaA = -2.3447473e-3 
    TH.SM.lamdaB = 1.5231928 
    TH.SM.lamdaC = -3.0944008e2
    TH.SM.zeta = 5.7587599e-3

    TH.SM.chbp = np.array([[1.04e-7, 1.50e-8, -8.48e-11, 1.53e-13, 2.21e-7, -2.35e-7], 
                           [-0.8934504, 8.64e-3, -3.2540648e-5, -1.4988959e-8, 4.57e-2, 4.01e-5], 
                           [0.5729271, -3.3513558e-3, -8.4827891e-6, 1.52e-7, -5.7929902e-2, 3.895e-5], 
                           [1.012116, -1.5057825e-3, 1.66e-5, -1.0011081e-7, -3.9395094e-2, -1.6943612e-4], 
                           [0.6804949, 4.08e-3, 1.81e-5, -4.3441358e-8, 1.90e-2, -1.7145340e-4], 
                           [2.130724, -1.0764878e-3, 5.35e-5, -2.1669502e-7, 1.007592, 1.04e-3], 
                           [0.9707671, 2.78e-2, -8.2421283e-5, 1.23e-7, 0.8249869, 1.05E-03]]) 
    
    '''
    TH.VM: Attributes of the Viscosity Model
    1.Huber et al.: New International formulation for the viscosity of H2O. J. Phys. Chem. Ref. Data 38, 101-125. 2009. 
    2.Sun, Rui and Niu, Zhigang and Lai, Shaocong.: Modeling dynamic viscosities of multi-component aqueous electrolyte solutions containing Li+, Na+, K+, Mg2+, Ca2+, Cl-, SO42- and dissolved CO2 under conditions
        of CO2 sequestration, Applied Geochemistry, 142, 2022.
    '''
    #2.
    TH.VM.gf = np.array([[33.551, -0.08043, 6.700e-5, 50.179, -0.1811, 1.708e-4], 
                         [4.7054, 0.06556, -1.553e-4, -61.411, 0.3973, -5.736e-4]])
    #1.
    TH.VM.Hi = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
    
    TH.VM.Hij = np.array([[5.20094e-1, 2.22531e-1, -2.81378e-1, 1.61913e-1, -3.25372e-2, 0., 0.], 
                          [8.50895e-2, 9.99115e-1, -9.06851e-1, 2.57399e-1, 0., 0., 0.], 
                          [-1.08374, 1.88797, -7.72479e-1, 0., 0., 0., 0.,], 
                          [-2.89555e-1, 1.26613, -4.89837e-1, 0., 6.98452e-2, 0., -4.35673e-3], 
                          [0., 0., -2.57040e-1, 0., 0., 8.72102e-3, 0.], 
                          [0., 1.20573e-1, 0., 0., 0., 0., -5.93264e-4]])

def ModelInitialization(file):
    '''
    Initializes the model and stores global info in V.py
    file : ".json" file provided by the user
    '''
    
    with open(file, "r") as f:
        param = json.load(f)

    V.L = param["domain_params"]["Length"]
    V.H = param["domain_params"]["Height"]
    V.nx = param["domain_params"]["NumCellsX"]
    V.ny = param["domain_params"]["NumCellsY"]
    
    if param["domain_params"]["refine_levels"] and param["domain_params"]["res_levels"]:
        mesh_type = "refined"
    else:
        mesh_type = "default"
    _MeshRefinement(type=mesh_type)

    V.P = param["physical_params"]["Pressure"]# MPa
    V.T = param["physical_params"]["Temperature"]# Celcius
    V.m = param["phase_params"]["NaCl"] 

    V.rw = param["phase_params"]["BrineDensity"]
    V.rs = param["phase_params"]["CO2SaturatedBrineDensity"]
    V.mu_co2br = param["phase_params"]["CO2SaturatedBrineViscosity"]
    V.mu = param["phase_params"]["BrineViscosity"]
    V.D = param["phase_params"]["DiffusionCoefficient"]
    V.c_sat = param["phase_params"]["CO2SaturatedConcentration"]

    V.g = 9.8 #m/s

    V.k_mean = param["kfield_params"]["Mean"]
    V.lnk_var = param["kfield_params"]["lnVariance"]
    V.k_corr = param["kfield_params"]["CorrelationLength"]
    
    V.al = param["physical_params"]["SubgridLongitudinalDispersion"]
    V.at = param["physical_params"]["SubgridTransverseDispersion"]   
    V.ST = param["physical_params"]["SimulationTime"]
    V.dt = param["physical_params"]["TimeIncrement"]

    V.pir = param["boundary_params"]["PressureIncrementRight"]
    V.pil = param["boundary_params"]["PressureIncrementLeft"]

    V.extent = param["reservoir_params"]["horizontal_extent"]
    V.mpp = param["reservoir_params"]["molesperparticle"]
    V.dirichlet_dofs = np.arange(int(V.extent[0]/V.dx[0]+1), int(V.extent[1]/V.dx[0])+1) # concentration boundary conditions: c=c_sat
    
    InitializeThermodynamicParams()

if __name__ == "__main__":
    from phase_module import density
    with open("inputs.json", "r") as f:
        param = json.load(f)

    ModelInitialization("inputs.json")

    
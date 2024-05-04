import numpy as np
from phase_module import density, solubility, viscosity, diffusivity
import formulate.IP as IP
import formulate.V as V
import transport_module.flow_solver as fs
import transport_module.particletracking as pt
import transport_module.field as field       
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import h5py
import matplotlib

class SpeciesInfo: 
    @staticmethod
    def info(mco2):
        # negligible dissociation of Co2 <=> H+ + HCO3-; vm = 1*m_co2 for Co2
        #\sigma v_i*m_i = v_na*m_na + v_cl*m_cl = 2*m_nacl
        SpeciesInfo.mco2 = mco2
        SpeciesInfo.xNaCl = V.m/(55.508 + 2.0*V.m + mco2)
        SpeciesInfo.xCO2 = mco2/(55.508 + 2.0*V.m + mco2)
        SpeciesInfo.xH2O = 1 - (SpeciesInfo.xNaCl + SpeciesInfo.xCO2)
        return SpeciesInfo

class Simulate:
    def __init__(self, filename, kf=None, phif=None, UsePhaseModule=True, datafile:str=None):    
        IP.ModelInitialization(filename)
        self.interp_DirichletBC = lambda f: V.gn[-1]*f[-2, V.dirichlet_dofs] + (1-V.gn[-1])*f[-1, V.dirichlet_dofs]

        self.attr = {"c": np.zeros([V.ny+2, V.nx+2]), 
                     "rho": np.zeros([V.ny+2, V.nx+2]), 
                     "rw": np.zeros([V.ny+2, V.nx+2]), 
                     "mu": np.zeros([V.ny+2, V.nx+2]), 
                     "D": np.zeros([V.ny+2, V.nx+2])}
            
        
        if V.mu_co2br:
            self.mu_func = lambda c: np.exp(np.log(V.mu_co2br/V.mu_br)*c)
                # see literature and set the exponential funcion for mu
        if V.rs and V.rw:
            self.rho_func = lambda c: V.rw + (V.rs-V.rw)*c/V.c_sat # use the standard interpolation for density

        self.kf = np.zeros([V.ny+2, V.nx+2])
        self.phif = np.zeros([V.ny+2, V.nx+2])

        if kf is not None:
            if isinstance(kf, np.ndarray):
                assert kf.shape == (V.ny, V.nx)
            self.kf[1:-1, 1:-1] = kf
        else:
            self.kf[1:-1, 1:-1] = field.Field.KField(V.x[1:-1], V.y[1:-1], V.k_mean, V.lnk_var, V.k_corr)
        self.kf[0, :] = self.kf[1, :]; self.kf[-1, :] = self.kf[-2, :]
        self.kf[:, -1] = self.kf[:, -2]; self.kf[:, 0] = self.kf[:, 1]

        if phif is not None:
            if isinstance(phif, np.ndarray):
                assert phif.shape == (V.ny, V.nx) 
            self.phif[1:-1, 1:-1] = phif
        else:
            self.phif[1:-1, 1:-1] = field.Field.PHIField(self.kf[1:-1, 1:-1])
        self.phif[0, :] = self.phif[1, :]; self.phif[-1, :] = self.phif[-2, :]
        self.phif[:, -1] = self.phif[:, -2]; self.phif[:, 0] = self.phif[:, 1]

        self.df = h5py.File(datafile, 'a') # initialize the binary file to write data into

        self.si = SpeciesInfo.info
        self.upm = UsePhaseModule

    def _apply_DirichletBC(self, f, bval):
        f[-1, V.dirichlet_dofs] = 2.0*bval - f[-2, V.dirichlet_dofs]
        f[0, :] = f[1, :]
        return f
    
    def _apply_attribute_bcs(self, p):
        if isinstance(p, np.ndarray):
            p = self.interp_DirichletBC(p)
            p_avg = np.average(p)

        c_bc = solubility.Solubility(P=p_avg, T=V.T).CO2Solubility() 
        rho_bc = density.Density(P=p, T=V.T).ComputeCO2BrineDensity(self.si(c_bc))
        mu_bc, _ = viscosity.CO2BrineViscosity(rho_bc, c_bc).ComputeMixtureViscosity(self.si(c_bc))
        c_bc *= rho_bc # to change concentration unit from moles/kg to moles/m3
        self.attr["c"], self.attr["rho"], self.attr["mu"] = list(map(self._apply_DirichletBC, [self.attr["c"], self.attr["rho"], self.attr["mu"]], 
                                             [c_bc, rho_bc, mu_bc]))
                
    def get_phase_attributes(self, p):
        self.attr["rw"][1:-1, 1:-1] = density.Density(P=p[1:-1, 1:-1], T=V.T).ComputeBrineDensity()
        self.attr["rho"][1:-1, 1:-1] = density.Density(P=p[1:-1, 1:-1], T=V.T).ComputeCO2BrineDensity(self.si(self.attr["c"][1:-1, 1:-1]))
        self.attr["mu"][1:-1, 1:-1], _ = viscosity.CO2BrineViscosity(
                                            self.attr["rho"][1:-1, 1:-1], self.attr["c"][1:-1, 1:-1]).ComputeMixtureViscosity(
                                                                                                                    self.si(self.attr["c"][1:-1, 1:-1]))
        self.attr["c"][1:-1, 1:-1] *= self.attr["rho"][1:-1, 1:-1] # moles/kg -> moles/m3

        self._apply_attribute_bcs(p) 

        self.attr["D"][1:-1, 1:-1] = diffusivity.ComputeDiffussionCoefficent().co2diffusivity(self.attr["rw"][1:-1, 1:-1]) # returns cellwise diffusion coefficient
        self.attr["D"][-1, :] = self.attr["D"][-2, :]; self.attr["D"][0, :] = self.attr["D"][1, :]
        self.attr["D"][:, -1] = self.attr["D"][:, -2]; self.attr["D"][:, 0] = self.attr["D"][:, 1]

    def _configure_init_condition(self):
        if self.upm:
            self.get_phase_attributes(np.ones([V.ny+2, V.nx+2])*V.P)
        else:
            self.attr["rho"][1:-1, 1:-1] = self.rho_func(self.attr["c"][1:-1, 1:-1])
            if V.mu_co2br:
                self.attr["mu"][1:-1, 1:-1] = self.mu_func(self.attr["c"][1:-1, 1:-1])
                self.attr["c"], self.attr["rho"], self.attr["mu"] = list(map(self._apply_DirichletBC, 
                                                                                [self.attr["c"], self.attr["rho"], self.attr["mu"]], 
                                                                                [V.c_sat, V.rs, V.mu_co2br]))
            else:
                self.attr["mu"][1:-1, 1:-1] = V.mu
                self.attr["c"], self.attr["rho"], self.attr["mu"] = list(map(self._apply_DirichletBC, 
                                                                                [self.attr["c"], self.attr["rho"], self.attr["mu"]], 
                                                                                [V.c_sat, V.rs, V.mu]))
            self.attr["D"][:, :] = V.D

        F = fs.FlowSolver(self.kf)
        F.solve(self.attr["rho"], self.attr["mu"])
        print("Initial condition configured")
        return F 

    def ParticleTracker(self, steps=None, realization=1, intervals=1, params=None):
        F = self._configure_init_condition()
        self.Ln = np.empty([0, 2]) # Ln: Lagrangian nodes denoting the particle locations

        if steps is None:
            steps = int(V.ST*3.154e7/V.dt)
        
        #NP = np.empty([steps, 2]) # total particle count (both inside and outside simulation domain)
        period = int(steps/intervals)
        dg = self.write_data_obj(realization, period)

        for i in tqdm(range(steps)):
            PT = pt.RWPT(self.attr["c"][:, :])
            slst = PT.particle_reservoir()
            slst += PT.diffuse(self.attr["D"], slst, "source")
            if len(self.Ln)!= 0:
                self.Ln += PT.diffuse(self.attr["D"], self.Ln, "domain")
                self.Ln = PT.apply_collision_bcs(self.Ln, "diffusion")
            self.Ln = np.concatenate((slst[slst[:, 1] <= V.H], self.Ln))
 
            Dxx, Dxy, Dyx, Dyy = F.CellwiseDispersion()
            self.Ln += PT.disperse(Dxx, Dxy, Dyx, Dyy, self.phif, F.qx, F.qy, self.Ln)
            NP = len(self.Ln)
            print(NP)
            self.Ln = PT.apply_collision_bcs(self.Ln, "dispersion")
            

            PT.binned_concentration(self.Ln); PT.c[1:-1, 1:-1] /= self.phif[1:-1, 1:-1]

            if self.upm:
                self.attr["c"][1:-1, 1:-1] = PT.c[1:-1, 1:-1]/self.attr["rho"][1:-1, 1:-1] # convert conc from moles/m3 -> moles/kg 
                xco2 = self.si(self.attr["c"][:, :]).xCO2
                self.get_phase_attributes(F.p/1e6)

            elif V.rs and V.rw: 
                self.attr["c"][1:-1, 1:-1] = PT.c[1:-1, 1:-1]
                self.attr["rho"][1:-1, 1:-1] = self.rho_func(self.attr["c"][1:-1, 1:-1])
                self.attr["D"][:, :] = V.D
                if V.mu_co2br:
                    self.attr["mu"][1:-1, 1:-1] = self.mu_func(self.attr["c"][1:-1, 1:-1])
                    self.attr["c"], self.attr["rho"], self.attr["mu"] = list(map(self._apply_DirichletBC, 
                                                                                [self.attr["c"], self.attr["rho"], self.attr["mu"]], 
                                                                                [V.c_sat, V.rs, V.mu_co2br]))
                else:
                    self.attr["mu"][1:-1, 1:-1] = V.mu
                    self.attr["c"], self.attr["rho"], self.attr["mu"] = list(map(self._apply_DirichletBC, 
                                                                                [self.attr["c"], self.attr["rho"], self.attr["mu"]], 
                                                                                [V.c_sat, V.rs, V.mu]))
            if params is not None:
                self.attr.update(params)
            F.solve(self.attr["rho"], self.attr["mu"])

            # write attributes to disk
            if i%intervals == 0:
                dg["rho"][:, :, i] = self.attr["rho"][:, :]
                if self.upm:
                    dg["c"][:, :, i] = xco2[:, :]
                else:
                    dg["c"][:, :, i] = self.attr["c"][:, :]/V.c_sat
                dg["v"][:, :, i] = F.Q[:, :]
                dg["p"][:, :, i] = F.p[:, :]
                dg["V"][:, :, i] = F.vorticity[:, :]
                dg["ppos_x"][i] = self.Ln[:, 0]
                dg["ppos_y"][i] = self.Ln[:, 1]
                dg["particle_count"][i] = [NP]
    
    def write_data_obj(self, idx, period):
        dg = self.df.create_group("realization" + "{}".format(idx)) # data group
        dg.create_dataset("v", shape=(V.En_int.shape[0], V.En_int.shape[1], period))
        dg.create_dataset("c", shape=(V.ny+2, V.nx+2, period))
        dg.create_dataset("rho", shape=(V.ny+2, V.nx+2, period))
        dg.create_dataset("p", shape=(V.ny+2, V.nx+2, period))
        dg.create_dataset("V", shape=(V.ny, V.nx, period))
        dg.create_dataset("ppos_x", shape=(period, ), dtype=h5py.vlen_dtype(np.dtype('float32')))
        dg.create_dataset("ppos_y", shape=(period, ), dtype=h5py.vlen_dtype(np.dtype('float32')))
        dg.create_dataset("particle_count", shape=(period, ), dtype=h5py.vlen_dtype(np.dtype('int32')))
        dg.create_dataset("nodal_coordinates", data=V.En_int)
        dg.create_dataset("permeability field", data=self.kf[1:-1, 1:-1])
        dg.create_dataset("porosity field", data=self.phif[1:-1, 1:-1])
        return dg

    







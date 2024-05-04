## PyDDC: A Lagrangian based numerical framework for simulating density driven convection of CO2--brine miscible flows in saturated porous media at geological storage conditions.
A reservoir simulator for solving 2D density-driven convection of CO2--brine mixture in fully saturated aquifers at geologic storage conditions. It uses a finite volume approach to solve the Darcy flux and combines it with a random walk particle tracking approach to simulate the sclar transport. Heterogeneity can be introduced by user specified random field meta-parameters --- mean permeability, log normal permeabiltiy variance and anisotropy based on difference in axial correlation lengths. Constains phase module for estimating the single phase attributes namely concentration, density, viscosity and diffusion coefficient of the mixture at ambient reservoir pressrue and temperature.

## Package Dependencies
The user must have Python-3.11.0 or above installed. Following are the lsit of packages with their minimum version of requirement which are needed to be installed before installing PyDDC. 
  1. scipy - 1.10.0
  2. numpy - 1.24.1
  3. gstools - 1.4.1
  4. h5py - 3.7.0
  5. tqdm - 4.66.2

## Functionalities
The coupling of the non-linear flow andtransport problem is handled by 2 interacting modules:


1. **formulate**: Contains Python files _IP.py_, _V.py_ and _TH.py_. ```IP.ModelInitialization(file)``` reads data from inputs.json (file) and creates attribute repository file V.py to store global variables required for simulation. It contains function ```_MeshRefinement(type=mesh_type)``` to define the 2D computational domain. ```TH.py``` holds data information to configure the phase equilibrium model of the CO2--brine mixture and compute phase attributes during simulation if the inbuilt thermodynamic model is used.


2. **phase_module**: Contains python files **density.py**, **solubility.py**, **diffusivity.py** and **viscosity.py**.
  - _density.py_: Estimates the density of the brine and the CO2--brine mixture at amnbient reservoir pressure and temperature.
                 The density class can be initialized by creating the class object ```d=Density(P,T)```. The density of the water can be computed by
                 calling ```d._iapws97()```. ```d.ComputeCO2BrineDensity(X)``` computes the density of the mixture or pure brine based on the mole
                 fraction of CO2 (X) in brine. Internally it calls the functions ```d._h20MolarVolume()``` and ```d.ApparentMolarVolumeSalt()``` to
                 compute the molar volumes of the different species required to compute the density.


  - _solubility.py_: Estimates the CO2 solubility in brine at any pressure and temperature. It is used to obtain the Dirichlet
                  boundary condition value for the CO2 concentration. The solubility class takes sclar values of pressure and temperature as
                  arguments and can be initialized by creating as ```s=Solubility(P,T)```. ```s.CO2Solubility()``` computes the concentration of CO2 in
                  brine based on teh CO2 activity coefficeint defined in ```s.CO2ActivityCoefficeint()```.


  - _viscosity.py_: Contains class ```CO2BrineViscosity(rho,mco2)``` responsible for estimating the viscosity of CO2--brine mixture or
                  pure brine based on density (rho) and CO2 solubility (mco2). Electrolyte free pure water viscosity is estimated in
                  ```h2OViscosity()``` and the mixture/brine viscosity is obtained by calling ```ComputeMixtureViscosity(X)``` where X is the mole-
                  fraction of CO2.


  - _diffusivity.py_: Computes the CO2 diffusivity in brine, defined in ```ComputeDiffussionCoefficent.co2diffusivity(rho)```, based on
                    salt molarity and brine density.




3. **transport_module**: Contains Python files _field.py_, _flow_solver.py_ and _particletracking.py_ responsible for simulating
the overall flow and transport behaviour on a predefiend homogeneous or heterogeneous random field.


  - _field.py_:  Computes the random permeability as defined in ```Field.KField(x, y, mean_k, var, corr_length)``` based on domain xy
                coordinates, mean permebaility variance and axial correlation lengths. The porosity field is computed from ```Field.PHIField(kf)```
                based on the permeability field, kf.


  - _flow_solver.py_: Obtains the pressure field and Darcy flux and is defiend inside the main class ```FlowSolver(field)``` where
                    field is the random permeability field. ```solve(r, mu)``` method is invoked to solve the global linear system of equations to
                    obtain the pressure based on density (r) and viscosity (mu). The Darcy flux is obtained directly from this pressure field by
                    computing fluxes at the cell interfaces using the standard linear interpolation method. ```CellwiseDispersion()``` is used to compute
                    the components of local dispersion tensor cellwise.


  - _particletracking.py_: Solves the random walk particle tracking (RWPT) method based on teh generalized stochastic differential
                          equation and is defined inside the class ```RWPT(c)``` which takes in the concentration field of CO2--brine mixture as its argument.
                          sCO2 source is represented as a dense particle cloud overlying the computation domain and is defined by the function
                          ```particle_reservoir()```. The RWPT algorithm is defined inside the ```disperse(Dxx, Dxy, Dyx, Dyy, phi, vx, vy, plst)``` which takes the
                          individual components in the disperison tensor, porosity, velocity components and particle configuration as arguments. boundary
                          conditions are enforced by ```apply_collision_bcs(plst, type)``` where plst are the vector of particle locations and type is the
                          physical process which is subjected to a particle type of boundary condition, namely "diffusion" or "dispersion". The binned
                          concentration is obtained from the particle positions by invoking the ```binned_concentration(plst)``` function.




_simulation.py_: Establishes the overall flow of control by integrating the previously defined modules in an efficient way. The high level control is provided by the ```Simulate(filename, kf, phif, UsePhaseModule, datafile)``` class inside which all the functionalities are defined. filename is the input JSON file containing the list of pre-defiend physical parameters and datafile is the name of the binary HDF5 file used to store simulation results. The user can directly supply permebaility (kf) and porosity (phif) fields if necessary and those field values would be used instead. Similarly the user can also decide whether or not to use the inbuilt phase module through the argument UsePhaseModule which is a boolean variable. If UsePhaseModule is set to false, the Simulate class reads the CO2 saturated concentration, diffusion coefficient and end-member density and viscosity from the global attribute repository _V.py_, defines interpolation functions for density and viscosity. The simulation process is initiated by calling the function ```ParticleTracker(steps=None, realization=1, intervals=1, params=None)``` where steps denote the user specified number of steps for the simulation to run, realization is the simulation result for a single permebaility-porosity field realization, intervals is the alternate period in years where the data has to be stored and params is a dictionary containing phase attributes which are computed by the user if those quantities are not determined from the inbuilt phase module.

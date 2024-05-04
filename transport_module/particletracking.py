import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import formulate.V as V
import transport_module.flow_solver as fs
import transport_module.field as field
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator


class helpers:
    @staticmethod
    def binning(plst, mpp):
        d, _ = np.histogramdd(plst, bins=(V.xf[1:-1], V.yf[1:-1]), density=False)
        d *= mpp/V.dv[1:-1, 1:-1].T
        return d[:, :].T 

    @staticmethod
    def interpolation(pts, vals, ppos, region="domain"):
        if region == "source":
            interp = NearestNDInterpolator(pts, vals.ravel())
        if region == "domain":
            interp = LinearNDInterpolator(pts, vals.ravel())
        return interp(ppos).reshape(-1, 1)
    
class RWPT:
    def __init__(self, c):
        self.c = c
        self.N = np.ceil(abs(self.c[-1, V.dirichlet_dofs]*V.dv[-1, V.dirichlet_dofs]/V.mpp))

    def binned_concentration(self, plst):
        self.c[1:-1, 1:-1] = helpers.binning(plst, V.mpp)

    def particle_reservoir(self):
        source = np.empty([0, 2]) 
        res_lower = V.dirichlet_dofs[0]-1
        for i in range(len(self.N)):
            particle_cloud = np.random.uniform(
                [V.xf[res_lower+i], V.H], [V.xf[res_lower+i+1], V.H+V.dy[-1][0]], 
                (int(self.N[i]), 2)
                )
            source = np.concatenate((particle_cloud, source))
        return source
    
    def diffuse(self, D, plst, region="source"):
        if region == "source": #perform constant interpolation for source particles
            interp = NearestNDInterpolator(V.En, D.ravel())
            D_p = interp(plst).reshape(-1, 1)
        elif region == "domain":
            interp = LinearNDInterpolator(V.En, D.ravel())
            D_p = interp(plst).reshape(-1, 1)
        return np.random.normal(0., 2.0*D_p*V.dt, size=plst.shape) 

    def _DivD(self, *args):
        # returns vector components in x and y 
        Dxx, Dxy, Dyx, Dyy = args[0], args[1], args[2], args[3] 

        dx_Dxx = np.zeros([V.ny, V.nx+1]); dy_Dxy = np.zeros_like(dx_Dxx)
        dy_Dyy = np.zeros([V.ny+1, V.nx]); dx_Dyx = np.zeros_like(dy_Dyy)

        dx_Dxx[:, :] = (Dxx[1:-1, 1:] - Dxx[1:-1, :-1])/(V.dx[:-1]/2 + V.dx[1:]/2).T
        dy_Dxy[:, :] = (Dxy[1:, :] - Dxy[:-1, :])/V.dy[1:-1]
        dx_Dyx[:, :] = (Dyx[:, 1:] - Dyx[:, :-1])/V.dx[1:-1].T
        dy_Dyy[:, :] = (Dyy[1:, 1:-1] - Dyy[:-1, 1:-1])/(V.dy[:-1]/2 + V.dy[1:]/2)

        divD_x = dx_Dxx + dy_Dxy; divD_y = dx_Dyx + dy_Dyy
        return divD_x, divD_y

    def _vec_tensor_ip(self, Dxx, Dxy, Dyx, Dyy, phi):
        # returns vector components in x and y
        phi_x, phi_y = field.Field.compute_field_gradient(phi)
        return Dxx[1:-1, :-1]*phi_x[1:-1, :] + Dxy[1:, ]*phi_y[1:, :-1], Dyx[:, 1:]*phi_x[:-1, 1:] + Dyy[:-1, 1:-1]*phi_y[:, 1:-1]
        
    def disperse(self, Dxx, Dxy, Dyx, Dyy, phi, vx, vy, plst):
        divD_x, divD_y = self._DivD(Dxx, Dxy, Dyx, Dyy)
        DP_x, DP_y = self._vec_tensor_ip(Dxx, Dxy, Dyx, Dyy, phi)
        # gather the 2 component vectors on the x and y direction 
        Ax = np.zeros([V.ny+2, V.nx+1]); Ay = np.zeros([V.ny+1, V.nx+2])
        Ax[1:-1, :] = (vx[1:-1, 1:-1] + phi[1:-1, :-1]*divD_x + DP_x)
        Ay[:, 1:-1] = (vy[1:-1, 1:-1] + phi[:-1, 1:-1]*divD_y + DP_y)
        # boundary conditions for linear interpolation 
        Ax[-1, :] = Ax[-2, :]; Ax[0, :] = Ax[1, :]
        Ay[:, -1] = Ay[:, -2]; Ay[:, 0] = Ay[:, 1]
        A = np.zeros_like(plst)
        Ax_p = helpers.interpolation(V.Ex, Ax, plst, "domain")
        Ay_p = helpers.interpolation(V.Ey, Ay, plst, "domain") 
        A[:, 0] = Ax_p.ravel(); A[:, 1] = Ay_p.ravel()

        vp = np.zeros_like(plst)
        vx_p = helpers.interpolation(V.Ex, vx[:, 1:-1], plst)
        vy_p = helpers.interpolation(V.Ey, vy[1:-1, :], plst)
        vp[:, 0] = vx_p.ravel(); vp[:, 1] = vy_p.ravel()
        v_norm = np.linalg.norm(vp, axis=1)

        # displacement matrix: BB^T = 2D
        B = np.zeros_like(plst)
        B[:, 0] = (vp[:, 0] * np.sqrt(2.0*V.al*v_norm) - vp[:, 1] * np.sqrt(2.0*V.at*v_norm)) / v_norm
        B[:, 1] = (vp[:, 1] * np.sqrt(2.0*V.al*v_norm) - vp[:, 0] * np.sqrt(2.0*V.at*v_norm)) / v_norm
        
        phi_p = helpers.interpolation(V.En, phi, plst)
        dX = 1/phi_p*A*V.dt + B*np.random.normal(size=(plst.shape[0], 2))*np.sqrt(V.dt)
        return dX
    
    def apply_collision_bcs(self, plst, type:str=None):
        ry = np.array([[-1, 0], [0, 1]]) # define the reflection matrices for 2 dimensions
        rx = np.array([[1, 0], [0, -1]])

        box = (plst[:, 0] <= V.extent[1]) & (plst[:, 0] >=V.extent[0]) & (plst[:, 1] <=V.H) & (plst[:, 1] >= 0.0)
        bbox = plst[box]
        obox = plst[~box]

        if len(obox)!=0 and type == "diffusion":
            top_nb = (obox[:, 1]>V.H) & (obox[:, 0]<V.extent[0]) & (obox[:, 0]>V.extent[1])
            bottom = obox[:, 1]<0.0
            refl_top = (np.array([0, V.H]).reshape(-1, 1) + (rx @ (obox[top_nb].T+np.array([0, -V.H]).reshape(-1, 1)))).T
            refl_bottom = (rx @ obox[bottom].T).T

            right =  obox[(obox[:, 0]>V.L) & ~top_nb & ~bottom]
            left = obox[(obox[:, 0]<0.0) & ~top_nb & ~bottom]

            if len(right) != 0 | len(left) != 0:
                refl_right = (np.array([V.L, 0]).reshape(-1, 1) +(ry @ (right.T+np.array([-V.L, 0]).reshape(-1, 1)))).T
                refl_left = (ry @ left.T).T

                refl_top_right = refl_top[:, 0]>V.L; refl_top_left = refl_top[:, 0]<0.0
                refl_bottom_right = refl_bottom[:, 0]>10; refl_bottom_left = refl_bottom[:, 0]<0
                # changes right and left reflection conditions only if particles are found to beyond the extent of the domain
                right = np.concatenate((right, refl_top[refl_top_right], refl_bottom[refl_bottom_right]))
                left = np.concatenate((left, refl_top[refl_top_left], refl_bottom[refl_bottom_left]))

                refl_right = (np.array([V.L, 0]).reshape(-1, 1) + (ry @ (right.T+np.array([-V.L, 0]).reshape(-1, 1)))).T
                refl_left = (ry @ left.T).T
                return np.concatenate((bbox, refl_top[~refl_top_right & ~refl_top_left], refl_bottom[~refl_bottom_right & ~refl_bottom_left], refl_right, refl_left))
            else:
                return np.concatenate((bbox, refl_top, refl_bottom))
        
        elif len(obox)!=0 and type == "dispersion":
            top = obox[:, 1]>V.H
            bottom = obox[:, 1]<0.0
            refl_top = (np.array([0, V.H]).reshape(-1, 1) + (rx @ (obox[top].T+np.array([0, -V.H]).reshape(-1, 1)))).T
            refl_bottom = (rx @ obox[bottom].T).T
            return np.concatenate((bbox, refl_top, refl_bottom))
        
        elif type is None:
            raise Exception("Must specify physics type to implement specific boundary")
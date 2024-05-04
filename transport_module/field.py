import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gstools as gs
import formulate.V as V
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def KC(x, k):
    d = 111.75e-6 # diameter of grain in mts
    S = 6/d
    return x**3 - k*2.5*(1-x)**2*S**2

class Field:
    @staticmethod 
    def KField(x, y, mean_k, var, corr_length):
        model = gs.Exponential(dim=2, var=var, len_scale=corr_length)
        srf = gs.SRF(model, mean=0.)
        field = srf((x, y), mesh_type='structured')
        field = (field - np.mean(field)) / np.std(field) + np.log(mean_k)
        return np.exp(field).T

    @staticmethod
    def generate_heterogeneity_field(x, y, mean_k, var, corr_length, savekf=False, savephif=False):
         
        k = Field.KField(x, y, mean_k, var, corr_length)
        phi = Field.PHIField(k)
        
        if savekf:
            fig, ax = plt.subplots()
            phi_c = ax.contourf(V.xx, V.yy, phi)
            fig.colorbar(phi_c, ax=ax)
            ax.set_aspect("equal")
            plt.savefig("PHI Field.png")

        if savephif:      
            fig, ax = plt.subplots()
            kc = ax.contourf(V.xx, V.yy, k)
            fig.colorbar(kc, ax=ax)
            ax.set_aspect("equal")
            plt.savefig("K Field.png")
        
        return k, phi
    
    @staticmethod 
    def PHIField(kf):
        """
        create a random porosity field from the random permeability field based on the Kozeny-Carman relation 
        """
        phi = np.zeros([V.ny, V.nx])
        for i in range(V.ny):
            for j in range(V.nx):
                phi[i, j] = root(KC, 1.0, kf[i, j]).x

        return phi
    
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
         
    if __name__ == "__main__":
        import formulate.IP as IP
        import formulate.V as V
        import matplotlib.pyplot as plt
        from field import Field as rf
        import seaborn as sns

        ip = IP.ModelInitialization("inputs.json", UsePhaseModule=True)
        
        kf = rf.KField(V.x, V.y, 1e-13, 0.5, [100, 10])
        phi = np.zeros([V.ny+2, V.nx+2])
        phi[1:-1, 1:-1] = PHIField(kf)
        phi[0, :] = phi[1, :]; phi[-1, :] = phi[-2, :]; phi[:, -1] = phi[:, -2]; phi[:, 0] = phi[:, 1]

        px, py = rf.compute_field_gradient(phi)
        print(px.shape, py.shape)

        fig1, ax = plt.subplots(2, 2, sharey="row")
        kimg = ax[0, 0].imshow(np.log(kf), origin="lower", aspect="auto", cmap='inferno', vmin=np.log(kf).min(), vmax=np.log(kf).max(), extent=(0, 1000, 0, 100))
        cb = fig1.colorbar(kimg, ax=ax[0, 0], location="top", shrink=0.8, ticks=np.linspace(np.log(kf).min(), np.log(kf).max(), 4))
        cb.set_label(r"ln$k$")
        u = ax[0, 1].imshow(phi, origin="lower", aspect="auto", cmap='inferno', vmin=phi.min(), vmax=phi.max(), extent=(0, 1000, 0, 100))
        cb = fig1.colorbar(u, ax=ax[0, 1], location="top", shrink=0.8, ticks=np.linspace(phi.min(), phi.max(), 4))
        cb.set_label(r"$\phi$")
        ax[0, 0].set_ylabel(r"$y$")
        ax[0, 0].set_xlabel(r"$x$")
        ax[0, 1].set_xlabel(r"$x$")

        ax[1, 0].hist(np.log(kf).ravel(), color="orange", bins=5000)
        ax[1, 0].set_xlabel(r"$lnk$")
        ax[1, 0].set_ylabel(r"counts")
        ax[1, 0].axvline(np.mean(np.log(kf.ravel())), color='crimson', ls=':', label=r'mean ln$k$')
        ax[1, 0].legend()
        ax[1, 1].hist(phi.ravel(), color="blue", bins=5000)
        ax[1, 1].set_xlabel(r"$\phi$")
        ax[1, 1].axvline(np.mean(phi).ravel(), color='crimson', ls=':', label=r'mean $\phi$')
        ax[1, 1].legend()
        fig1.tight_layout()
        fig1.savefig("random k field.png", dpi=800)

        
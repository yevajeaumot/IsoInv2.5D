# needs env cart
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from scipy.optimize import least_squares
from scipy.interpolate import griddata
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib import cm, ticker
import time
import math
import os
import resource


class FlowLine(object):

    def __init__(self,label):

        self.label = label

    # set default parameters
    def default_params(self):

        # Default values for parameters
        self.ic = {}
        self.imax = 100
        self.ninv = 20
        self.nobs = 50
        self.delta = 0.08
        self.age_surf = -50
        self.x_right = 370.
        self.x_step = 1.
        self.thickness_ie = False
        self.accu_relative = 1.
        self.accu_present = True
        self.R_exp = 1.
        self.temp_fact_linear = False
        self.traj_step = 10
        self.fig_age_max = 1000000
        self.fig_age_spacing = 10000
        self.fig_age_spacing_labels = 100000
        self.fig_age_iso = [100, 200, 300, 400]
        self.beta = 0.015
        self.create_figs = True
        self.fig_format = 'pdf'
        self.comp_flowline = None
        self.comp_isochrones = None
        self.output_ic = True
        self.output_fl = True
        self.counter = 0
        self.error = []
        self.jac = False
        self.interp_total = 0
        self.mat_total = 0
        self.age_total = 0
        self.calc_jac = True

    # load parameters from yml file
    def load_parameters(self):
        # ---------------------------------------------------------
        # Read parameters.yml file (imax, delta, ...)
        # ---------------------------------------------------------
        # parameters for all radar lines (file is mandatory - will overwrite default params)
        data = yaml.load(open(self.label+'/parameters.yml').read(),
                         Loader=yaml.FullLoader)
        if data != None:
            self.__dict__.update(data)

    # load age scale and flow line data from .txt files
    def load_data(self):

        # -----------------------------------------------------------------------------
        # Load data from temporal_factor.txt and relative_density.txt
        # -----------------------------------------------------------------------------
        self.age_R, self.R = np.loadtxt(self.label+'temporal_factor.txt', unpack=True)
        self.age_R = np.append(self.age_R, self.age_R[-1]+1)
        self.age_R = np.append(self.age_R, 10000000)
        self.R = np.append(self.R, 1)
        self.R = np.append(self.R, 1)
        self.R = self.R**self.R_exp
        if self.temp_fact_linear:
            self.R = (self.R[1:] + self.R[:-1])/2
            self.R = np.append(self.R, self.R[-1])

        # ------------------------------
        # Computation of steady_age_R
        # -------------------------------
        self.steady_age_R = np.concatenate((np.array([self.age_R[0]]),
                                       (self.age_R[1:] - self.age_R[:-1]) * self.R[:-1]))
        self.steady_age_R = np.cumsum(self.steady_age_R)

        self.D_depth, self.D_D = np.loadtxt(self.label+'relative_density.txt', unpack=True)

        # -----------------------------------------------------
        # Load files for Geographic data, arrays creations
        # -----------------------------------------------------

        # Steady accumulation
        self.x_a, self.a_measure = np.loadtxt(self.label+'accumulation.txt', unpack=True)
        self.a_measure = self.a_measure * self.accu_relative
        if self.accu_present:
            self.a_measure = self.a_measure / self.R[0]
            #self.a_measure = self.a_measure / R[0]

        # Sliding rate
        self.x_s, self.s_measure = np.loadtxt(self.label+'sliding.txt', unpack=True)

        # Lliboutry parameter
        self.x_p, self.p_measure = np.loadtxt(self.label+'p_Lliboutry.txt', unpack=True)

        # Surface
        self.x_Su, self.Su_measure = np.loadtxt(self.label+'surface.txt', unpack=True)

        # Thickness
        self.x_H, self.H_measure = np.loadtxt(self.label+'thickness.txt', unpack=True)

        # Tube width
        self.x_Y, self.Y_measure = np.loadtxt(self.label+'tube_width.txt', unpack=True)
        
        # Melting
        self.x_m, self.m_measure = np.loadtxt(self.label+'melting.txt', unpack=True)


    # load isochrone data
    def load_obs_data(self):

        # -----------------------------------------------------------
        # Reading of comparison files
        # -----------------------------------------------------------

        if self.comp_flowline is not None:
            self.cp_fl_x, self.cp_fl_ux_surf = np.loadtxt(self.label+self.comp_flowline, unpack=True)

        if self.comp_isochrones is not None:
            readarray = np.loadtxt(self.label+self.comp_isochrones, unpack=True)
            self.cp_iso_x = readarray[0, :]
            self.cp_iso_depth = readarray[1:, :]
            self.cp_iso_nb = self.cp_iso_depth.shape[0]

    		# load ages of isochrones
            xlen = self.ninv+1          # ??? perhaps do this elsewhere ninv
            self.iso_obs_age, self.iso_obs_age_sigma = np.transpose(np.loadtxt(self.label+'/ages.txt',delimiter='\t'))
            self.iso_obs_steadyage = np.interp(self.iso_obs_age-self.age_surf, self.age_R, self.steady_age_R)
            self.iso_obs_steadyage_sigma = np.interp(self.iso_obs_age_sigma, self.age_R, self.steady_age_R)


        for name in self.ic:
            if self.ic[name]['constraint'] is not None:
                self.ic[name]['obs_depth'], self.ic[name]['obs_age'], self.ic[name]['obs_age_sigma'] = np.loadtxt(self.label+self.ic[name]['constraint'], unpack=True)

    # set up grids for all variables
    def initial_setup(self):

        self.x_fld = np.arange(0, self.x_right+self.x_step, self.x_step)
        self.Y_fld = np.interp(self.x_fld, self.x_Y, self.Y_measure)   # doesn't change after this
        self.m_fld = np.interp(self.x_fld, self.x_m, self.m_measure)
        # updated with each iteration (below) ninv for all below
        self.x_inv = np.linspace(0,self.x_right, self.ninv+1)      # dictates number in array to minimise
        self.x_obs = np.linspace(np.min(self.cp_iso_x),self.x_right, self.nobs+1)
        self.a_inv = np.interp(self.x_inv, self.x_a, self.a_measure)
        self.H_inv = np.interp(self.x_inv, self.x_H, self.H_measure)
        self.p_inv = np.interp(self.x_inv, self.x_p, self.p_measure)
        self.m_inv = np.interp(self.x_inv, self.x_m, self.m_measure)
        
        self.Delta_inv = np.zeros_like(self.H_inv)

        self.p_prime_inv = np.log(self.p_inv+1)
        # generate arrays for comparison
        self.iso_obs_steadyage = np.array(len(self.x_obs)*[self.iso_obs_steadyage]).T
        self.iso_obs_steadyage_sigma = np.array(len(self.x_obs)*[self.iso_obs_steadyage_sigma]).T
        self.iso_obs_age = np.array(len(self.x_obs)*[self.iso_obs_age]).T
        self.iso_obs_age_sigma = np.array(len(self.x_obs)*[self.iso_obs_age_sigma]).T

        # for residual calculation
        self.iso_obs_depth = np.array([np.interp(self.x_obs, self.cp_iso_x, single_iso) for single_iso in self.cp_iso_depth])
        self.H_obs = np.interp(self.x_obs, self.x_H, self.H_measure)
        

        # ----------------------------------------------------------
        # Mesh generation (pi,theta)
        # ----------------------------------------------------------
        # ??? will be the same every time
        self.pi = np.linspace(-self.imax * self.delta, 0,  self.imax + 1)
        # Rmk: We could make a column vector here and for OMEGA
        self.theta = np.linspace(0, - self.imax * self.delta,  self.imax + 1)

        self.Qm = np.zeros(len(self.pi))
        # ----------------------------------------------------------
        # OMEGA
        # ----------------------------------------------------------
        self.OMEGA = np.exp(self.theta)

        # -----------------------------------------------------
        # depth vs ie-depth conversion with density data
        # -----------------------------------------------------
        self.D_depth_ie = np.cumsum(np.concatenate((np.array([0]),
                                 self.D_D[:-1] * (self.D_depth[1:]-self.D_depth[:-1]))))

        # ----------------------------------------------------------
        # DELTA H
        # ----------------------------------------------------------
        self.DELTA_H = self.D_depth[-1] - self.D_depth_ie[-1]
        print('DELTA_H:', self.DELTA_H)

        # # -------------------------------------------------------
        # # Matrix theta
        # # -------------------------------------------------------
        # self.mat_theta = self.theta.reshape(self.imax+1, 1)*np.ones((1, self.imax+1))
        # # theta_min_mesh is the grid min for each vertical profile, to plot the mesh.
        # self.theta_min_mesh = np.nanmin(self.mat_theta, axis=0)
        
       
        # -------------------------------------------------------
        # Matrix OMEGA: mat_OMEGA
        # -------------------------------------------------------
        #self.mat_OMEGA = np.array((self.imax+1)*[self.OMEGA]).T
        #self.mat_OMEGA = np.where(grid, self.OMEGA.reshape(self.imax+1, 1), np.nan)
        # -------------------------------------------------------
        # Matrix pi: mat_pi
        # -------------------------------------------------------
        #self.mat_pi = np.array((self.imax+1)*[self.pi]).T
        
        # age density array
        #self.age_density = np.ones_like(self.mat_OMEGA)*np.nan


        self.x_reg = np.linspace(0,self.x_right, self.njac+1)      # dictates number in array to minimise
        self.depth_reg = np.linspace(0., np.max(self.H_measure), self.njac+1)

        self.jac_mat_x = np.array((self.njac+1)*[self.x_reg])                   # not used if we keep simple interpolation
        self.jac_mat_depth = np.array((self.njac+1)*[self.depth_reg]).T

        self.resi_sd = np.full(self.nobs+1,np.nan)
        self.niso = np.full(self.nobs+1,np.nan)

    # interpolate variables onto grid for forward model calculation
    def interpolation(self):

        # --------------------
        # Interpolation
        # --------------------
        self.a_fld = np.interp(self.x_fld, self.x_inv, self.a_try)

        # Computation of total flux Q
        # Formula checked 2023/04/27 by F. Parrenin
        dQdx = (self.x_fld[1:]-self.x_fld[:-1]) * 1000 * \
            (self.a_fld[:-1] * self.Y_fld[:-1] +
             0.5 * ((self.a_fld[1:]-self.a_fld[:-1]) * self.Y_fld[:-1] + (self.Y_fld[1:]-self.Y_fld[:-1])
                    * self.a_fld[:-1]) +
             1./3 * (self.a_fld[1:]-self.a_fld[:-1]) * (self.Y_fld[1:]-self.Y_fld[:-1]))
        dQdx = np.insert(dQdx, 0, 0)
        Q_fld = np.cumsum(dQdx)

        self.m_fld = np.interp(self.x_fld, self.x_inv, self.m_try)

        # Computation of basal melting flux Qm
        dQmdx = (self.x_fld[1:]-self.x_fld[:-1]) * 1000 * \
            (self.m_fld[:-1] * self.Y_fld[:-1] +
             0.5 * ((self.m_fld[1:]-self.m_fld[:-1]) * self.Y_fld[:-1] + (self.Y_fld[1:]-self.Y_fld[:-1])
                    * self.m_fld[:-1]) +
             1./3 * (self.m_fld[1:]-self.m_fld[:-1]) * (self.Y_fld[1:]-self.Y_fld[:-1]))
        dQmdx = np.insert(dQmdx, 0, 0)
        Qm_fld = np.cumsum(dQmdx)
        
        # ----------------------------------------------------------
        # Total flux Q(m^3/yr)
        # ----------------------------------------------------------
        self.Q = Q_fld[-1] * np.exp(self.pi)  # Q_ref = Q_fld[-1]
        # ----------------------------------------------------------
        # interpolation of flow line data files for x, Qm, ...
        # ----------------------------------------------------------
        # We need to interpolate x in Q, but then we can interpolate in x.
        # We could also interpolate everything in Q.
        self.x = np.interp(self.Q, Q_fld, self.x_fld)
        self.Qm = np.interp(self.Q, Q_fld, Qm_fld)
        self.a = np.interp(self.x, self.x_inv, self.a_try)
        self.H = np.interp(self.x, self.x_inv, self.H_try)
        self.p = np.interp(self.x, self.x_inv, self.p_try)
        self.Y = np.interp(self.x, self.x_Y, self.Y_measure)
        self.S = np.interp(self.x, self.x_Su, self.Su_measure)
        self.s = np.interp(self.x, self.x_s, self.s_measure)
        # get depths of isochrones in current depth matrix
        self.iso_obs_depth_imax = np.array([np.interp(self.x, self.cp_iso_x, single_iso) for single_iso in self.cp_iso_depth])

        # ------------------------------------------------------
        # Computation of theta_min and theta_max
        # ------------------------------------------------------
        # We just use theta_max for the mesh plot
        self.theta_max = np.zeros(self.imax+1)
        #self.theta_min = -self.imax*self.delta * np.ones((self.imax+1,))
        self.theta_min = np.where(self.Qm > 0,
                     np.maximum(np.log(self.Qm.clip(min=10**-100)/self.Q),
                                -self.imax*self.delta * np.ones((self.imax+1,))),
                     -self.imax*self.delta * np.ones((self.imax+1,)))


        # --------------------------------------------------
        # Computation of H, S_ie and B
        # --------------------------------------------------
        if self.thickness_ie:
            self.H_ie = self.H
            self.H = self.H_ie + self.DELTA_H
        else:
            self.H_ie = self.H - self.DELTA_H
        self.B = self.S - self.H
        self.S_ie = self.S - self.DELTA_H
        
        # --------------------------------------------------
        # Melting
        # --------------------------------------------------

        # m is just used for the boundary conditions plot.
        self.m = np.interp(self.x, self.x_fld, self.m_fld)


    # staircase interpolation for accumulation rate
    def interp_stair_aver(self, x_out, x_in, y_in):
        """Return a staircase interpolation of a (x_in,y_in) series
        at x_out abscissas with averaging."""
        x_mod = x_in+0
        y_mod = y_in+0
        if x_out[0] < x_in[0]:
            x_mod = np.concatenate((np.array([x_out[0]]), x_mod))
            y_mod = np.concatenate((np.array([y_in[0]]), y_mod))
        if x_out[-1] > x_in[-1]:
            x_mod = np.concatenate((x_mod, np.array([x_out[-1]])))
            y_mod = np.concatenate((y_mod, np.array([y_in[-1]])))
        y_int = np.cumsum(np.concatenate((np.array([0]),
                                          y_mod[:-1]*(x_mod[1:]-x_mod[:-1]))))

        y_out = (np.interp(x_out[1:], x_mod, y_int) -
                 np.interp(x_out[:-1], x_mod, y_int)) / (x_out[1:]-x_out[:-1])
        return y_out

    # calculate matrices for current run
    def matrices(self):
        
        # -------------------------------------------------------
        # GRID
        # -------------------------------------------------------

        self.grid = np.ones((self.imax + 1, self.imax + 1), dtype=bool)

        self.grid[:, 0] = self.theta >= self.theta_min[0]

        for j in range(1, self.imax+1):
            self.grid[1:, j] = np.logical_and(self.theta[1:] >= self.theta_min[j-1], self.grid[0:-1, j-1])
        # print('After defining grid boolean ',
        #       round(time.perf_counter()-self.START_TIME, 4), 's.')

        # -------------------------------------------------------
        # Matrix theta
        # -------------------------------------------------------

        self.mat_theta = self.theta.reshape(self.imax+1, 1)*np.ones((1, self.imax+1))
        self.mat_theta = np.where(self.grid, self.mat_theta, np.nan)
        # theta_min_mesh is the grid min for each vertical profile, to plot the mesh.
        self.theta_min_mesh = np.nanmin(self.mat_theta, axis=0)

        # -------------------------------------------------------
        # Matrice omega : mat_omega
        # -------------------------------------------------------

        self.mat_omega = np.zeros((self.imax+1, self.imax+1))

        self.mat_omega = np.where(self.grid,
                             (np.dot(self.OMEGA.reshape(self.imax+1, 1),
                                     self.Q.reshape(1, self.imax+1))-self.Qm)/(self.Q-self.Qm),
                             np.nan)
       

        # ------------------------------------------------------
        # Computation of omega=fct(zeta)
        # ------------------------------------------------------
        # vertical coordinate zeta
        zeta = np.linspace(1, 0, 1001).reshape(1001, 1)
        # Lliboutry model for the horizontal flux shape function
        self.omega = zeta * self.s + (1-self.s) * (1 - (self.p+2)/(self.p+1) * (1-zeta) +
                                    1/(self.p+1) * np.power(1-zeta, self.p+2))

        # -------------------------------------------------------
        # Matrix mat_z_ie
        # -------------------------------------------------------
        self.mat_z_ie = np.zeros((self.imax+1, self.imax+1))

        for j in range(0, self.imax+1):
            # self.inter = np.interp(-self.mat_OMEGA[:, j], -self.omega[:, j].flatten(),
            #                   zeta.flatten())
            self.inter = np.interp(-self.mat_omega[:, j], -self.omega[:, j].flatten(),
                                  zeta.flatten())
            #self.mat_z_ie[:, j] = self.B[j]+self.inter*self.H_ie[j]
            self.mat_z_ie[:, j] = np.where(self.grid[:, j], self.B[j]+self.inter*self.H_ie[j],
                              np.nan)

        # z_ie_min is the grid min for each vertical profile, used to plot the mesh
        self.z_ie_min_mesh = np.nanmin(self.mat_z_ie, axis=0)
        
        # -------------------------------------------------------
        # Matrix OMEGA: mat_OMEGA
        # -------------------------------------------------------

        self.mat_OMEGA = np.where(self.grid, self.OMEGA.reshape(self.imax+1, 1), np.nan)

        # -------------------------------------------------------
        # Matrix pi: mat_pi
        # -------------------------------------------------------

        self.mat_pi = np.where(self.grid, self.pi, np.nan)

        # -------------------------------------------------------
        # Matrix x: mat_x
        # -------------------------------------------------------
        #self.mat_x = np.array((self.imax+1)*[self.x])
        self.mat_x = np.where(self.grid, self.x, np.nan)
        # -------------------------------------------------------
        # Matrix depth_ie: mat_depth_ie
        # -------------------------------------------------------
        #self.mat_depth_ie = self.S_ie - self.mat_z_ie
        self.mat_depth_ie = np.where(self.grid, self.S_ie - self.mat_z_ie, np.nan)

        self.mat_depth_ie[0, :] = 0
        # ----------------------------------------------------------
        #  Computation of depth matrix: mat_depth
        # ----------------------------------------------------------
        self.mat_depth = np.interp(self.mat_depth_ie, np.append(self.D_depth_ie,
                                                      self.D_depth_ie[-1]+10000.),
                              np.append(self.D_depth, self.D_depth[-1]+10000.))
        # ----------------------------------------------------------
        #  Computation of z matrix: mat_z
        # ----------------------------------------------------------
        self.mat_z = self.S - self.mat_depth
        # -------------------------------------------------------
        # Matrix of stream function q: mat_q
        # -------------------------------------------------------
        #self.mat_q = self.Q * self.mat_OMEGA
        self.mat_q = np.where(self.grid, self.Q * self.mat_OMEGA, np.nan)

        # -------------------------------------------------------
        # Matrix a0: mat_a0
        # -------------------------------------------------------
        # a0 is not defined when trajectories reach the dome area, so we set to a[0].
        #self.mat_a0 = toeplitz(self.a[0]*np.ones(self.imax+1), self.a)
        self.mat_a0 = np.where(self.grid, toeplitz(self.a[0]*np.ones(self.imax+1), self.a),
                  np.nan)

        # -------------------------------------------------------
        # Matrix x0: mat_x0
        # -------------------------------------------------------
        self.mat_x0 = np.zeros((self.imax+1, self.imax+1))
        # x0 is not defined when trajectories reach the dome area, so we set to x[0].
        #self.mat_x0 = toeplitz(self.x[0]*np.ones(self.imax+1), self.x)
        self.mat_x0 = np.where(self.grid, toeplitz(self.x[0]*np.ones(self.imax+1), self.x),
                  np.nan)
        
        # age density array
        self.age_density = np.ones_like(self.mat_OMEGA)*np.nan


    # calculate basal melt rate
    # def melt_calc(self):
    #     self.m = np.empty(len(self.x))
    #     self.m[:] = np.nan
    #     H_obs = np.interp(self.x, self.x_inv, self.H_inv)
    #     self.x_m = np.copy(self.x)
    #     # nan where stagnant ice
    #     self.x_m[H_obs>self.mat_depth[-1]] = np.nan
    #     # Qm is q at depth Hobs
    #     Qm = np.array([np.interp(H_obs[i], self.mat_depth[:,i], self.mat_q[:,i]) for i in range(len(H_obs))])
    #     H_obs[H_obs>self.mat_depth[-1]] = np.nan
    #     Qm[H_obs>self.mat_depth[-1]] = np.nan
    #     self.m = np.zeros(len(Qm))

    #     # mask where melting is non zero
    #     Qm_use = Qm[~np.isnan(Qm)]
    #     self.x_m = self.x_m[~np.isnan(Qm)]
    #     Y = self.Y[~np.isnan(Qm)]

    #     nonnan=0       # count non nans in Qm and x_m
    #     for point in range(1,len(Qm)):
    #         # calculate melt rate m
    #         if not math.isnan(Qm[point]):
    #             self.m[point] = (Qm_use[nonnan] - Qm_use[nonnan-1]) / (Y[nonnan]* (self.x_m[nonnan] - self.x_m[nonnan-1]))
    #             nonnan = nonnan+1
    #         else:
    #             self.m[point] = np.nan

    #         if nonnan==1:
    #             self.m[point]=np.nan

    #     # interpolate onto optimisation grid
    #     self.m_inv = np.interp(self.x_inv, self.x, self.m)

    # determine thickness of refrozen/accreted ice
    # def refrozen(self):
    #     # # calc refrozen ice once at the end
    #     self.H_interp = np.interp(self.x, self.x_inv, self.H_inv)
    #     self.ref_dep = np.empty_like(self.H_interp)
    #     self.ref_dep[:] = np.nan
    #     min_q=0
    #     #
    #     for i in range(len(self.mat_x[:])):
    #         end_q = len(self.mat_depth[:,i][self.mat_depth[:,i]<self.H_interp[i]])-1
    #         if min_q<self.mat_q[end_q,i]:
    #             min_q = self.mat_q[end_q,i]
    #         self.ref_dep[i] = self.mat_depth[:,i][self.mat_q[:,i]>min_q][-1]
    #         self.mat_q[:,i][self.mat_q[:,i]<min_q] = np.nan

    #     self.H_ref = np.interp(self.x_inv, self.x, self.ref_dep)

    # calculate age matrix
    def age_calc(self):

        # -------------------------------------------------------
        # Matrix STEADY-AGE:
        # -------------------------------------------------------
        interp_before = time.perf_counter()

        self.mat_steady_age = np.zeros((self.imax+1, self.imax+1))

        # Dome boundary condition
        self.mat_steady_age[1:, 0] = self.delta / self.a[0] * np.cumsum((self.mat_z_ie[:-1, 0] -
                                                          self.mat_z_ie[1:, 0]) /
                                                         (self.OMEGA[:-1] - self.OMEGA[1:]))

        dzdOMEGA = (self.mat_z_ie[1:, :] - self.mat_z_ie[:-1, :]) /\
                   (self.OMEGA[1:] - self.OMEGA[:-1]).reshape(self.imax, 1)
        # Calculation line by line, F. Parrenin, 2023/04/28
        # Rmq : It is possible to calculate column by column or diagonal by diagonal
        for i in range(1, self.imax+1):
            self.mat_steady_age[i, 1:] = self.mat_steady_age[i-1, :-1] + self.delta * (
                dzdOMEGA[i-1, :-1]/self.a[:-1] +
                0.5 * (dzdOMEGA[i-1, 1:] - dzdOMEGA[i-1, :-1])/self.a[:-1] +
                0.5 * dzdOMEGA[i-1, :-1] * (1/self.a[1:] - 1/self.a[:-1]) +
                1./3 * (dzdOMEGA[i-1, 1:] - dzdOMEGA[i-1, :-1]) * (1/self.a[1:] - 1/self.a[:-1]))
        # The grid for the age can be slightly different if the nb of nodes increases
        self.grid_age = ~np.isnan(self.mat_steady_age)
        interp_before = time.perf_counter()

        # -------------------------------------------------------
        # Matrix of thinning function: mat_tau
        # -------------------------------------------------------
        #self.mat_tau = (self.mat_z_ie[:-1, :] - self.mat_z_ie[1:, :]) / (self.mat_steady_age[1:, :] - self.mat_steady_age[:-1, :]) / (self.mat_a0[:-1, :] + self.mat_a0[1:, :]) * 2
        self.mat_tau = np.where(self.grid[1:, :], (self.mat_z_ie[:-1, :] - self.mat_z_ie[1:, :])
                   / (self.mat_steady_age[1:, :] - self.mat_steady_age[:-1, :])
                   / (self.mat_a0[:-1, :] + self.mat_a0[1:, :]) * 2, np.nan)

        
        # --------------------------------------------------------------------
        # Matrix of thinning function with analytical formula: mat_tau_anal
        # --------------------------------------------------------------------
        tau_reduc_pitheta = np.zeros((self.imax+1, self.imax+1))

        # Calculation line by line, F. Parrenin, 2023/05/03
        # Based on the analytical formula from Parrenin (HDR, 2013)
        for i in range(1, self.imax+1):
            tau_reduc_pitheta[i, 1:] = tau_reduc_pitheta[i-1, :-1] + \
                dzdOMEGA[i-1, 1:] / self.a[1:] - dzdOMEGA[i-1, :-1] / self.a[:-1]
        self.mat_tau_anal = np.ones((self.imax+1, self.imax+1))
        self.mat_tau_anal[1:-1, :] = 1/(self.mat_tau_anal[1:-1, :] - tau_reduc_pitheta[1:-1, :]
                                   * self.a / (dzdOMEGA[:-1, :] + dzdOMEGA[1:, :]) * 2)
        self.mat_tau_anal = self.mat_tau_anal * self.a / self.mat_a0 * self.OMEGA.reshape(self.imax+1, 1)
        self.mat_tau_anal[-1, :] = np.nan

        # ---------------------------------
        # Surface velocity
        # ---------------------------------
        self.ux_surf = self.Q/self.Y/dzdOMEGA[0, :]

        # ----------------------------------------------------------
        #  Computation age matrix: mat_age
        # ----------------------------------------------------------
        # Rmq if age_R[0]>age_surf, there is a top layer of age age_R[0]
        self.mat_age = np.interp(self.mat_steady_age+self.age_surf, self.steady_age_R, self.age_R)
        # # list comprehension, slightly faster than for loop

        # print('before iso age')
        obs_age = np.array([np.interp(self.x_obs, self.x, self.mat_age[i]) for i in range(len(self.mat_x[0])) ])
        obs_depth = np.array([np.interp(self.x_obs, self.x, self.mat_depth[i]) for i in range(len(self.mat_x[0])) ])
        self.iso_mod_age = np.array([np.interp(self.iso_obs_depth[:,i],obs_depth[:,i], obs_age[:,i]) for i in range(len(self.x_obs))]).T

        for name in self.ic:
            if self.ic[name]['constraint'] is not None:
                ic_age = np.array([np.interp(self.ic[name]['x'], self.x, self.mat_age[:,i]) for i in range(len(self.mat_x[:,0])) ]).T
                ic_depth = np.array([np.interp(self.ic[name]['x'], self.x, self.mat_depth[:,i]) for i in range(len(self.mat_x[:,0])) ]).T
                self.ic[name]['mod_age'] = np.interp(self.ic[name]['obs_depth'], ic_depth, ic_age)

        # only do these calculations if caclculating the jacobian
        if self.jac == True:

            self.jac_mat_age = np.array([np.interp(self.depth_reg, self.mat_depth[:,i], self.mat_age[:,i]) for i in range(len(self.x)) ]).T
            self.jac_mat_age = np.array([np.interp(self.x_reg, self.x, self.jac_mat_age[i]) for i in range(len(self.depth_reg)) ])

            for j in range(self.imax):
                self.age_density[1:-2,j] = 1/self.a[j]*(1/ self.mat_tau[1:-1, j] + 1/self.mat_tau[:-2, j])/2
            
            self.jac_mat_age = np.nan_to_num(self.jac_mat_age, nan=0.0)
            #self.melt_calc()
            return np.concatenate((self.H_try, self.m_try, self.jac_mat_age.flatten()))

    # get jacobian for given index
    def jacobian(self):
        self.jac = True         # makes age calc return jacobian values
        self.variables = self.variables.flatten()
        epsilon = np.sqrt(np.diag(self.hess))/10000000000.
        model0 = self.age_calc()
        # model0 should be a_try, p_try, H_try ie the optimised values
        jacob = np.empty((np.size(self.variables), np.size(model0)))

        for i in np.arange(len(self.variables)):
            self.variables[i] = self.variables[i]+epsilon[i]
            self.residuals(self.variables)
            model1 = self.age_calc()
            self.variables[i] = self.variables[i]-2*epsilon[i]
            self.residuals(self.variables)
            model2 = self.age_calc()
            jacob[i, :] = (model1-model2)/2./epsilon[i]
            self.variables[i] = self.variables[i]+epsilon[i]
        self.residuals(self.variables)
         
        return jacob

    # get uncertainties for various parameters
    def sigma(self):
        mat_before = time.perf_counter()
        jacob = self.jacobian()
        self.mat_total += time.perf_counter() - mat_before
        index = 0

        age_before = time.perf_counter()
        self.sigma_a = np.sqrt(np.diag(self.hess[index:index+self.ninv+1, index:index+self.ninv+1]))*self.a_try
        # index = index+self.ninv+1

        self.sigma_p = np.sqrt(np.diag(self.hess[index:index+self.ninv+1, index:index+self.ninv+1]))*self.p_try
        # index = index+self.ninv+1

        self.sigma_Delta = np.sqrt(np.diag(self.hess[index:index+self.ninv+1, index:index+self.ninv+1]))
        # index = index+self.ninv+1

        c_model = np.dot(np.transpose(jacob[:, index:index+self.ninv+1]),
                         np.dot(self.hess, jacob[:, index:index+self.ninv+1]))
        self.sigma_H = np.sqrt(np.diag(c_model))
        index = index+self.ninv+1

        c_model = np.dot(np.transpose(jacob[:, index:index+self.ninv+1]),
                         np.dot(self.hess, jacob[:, index:index+self.ninv+1]))
        self.sigma_m = np.sqrt(np.diag(c_model))
        index = index+self.ninv+1

        # for grid of age uncertainties
        c_model = np.dot(np.transpose(jacob[:, index:index+np.size(self.jac_mat_age)]),
                         np.dot(self.hess, jacob[:, index:index+np.size(self.jac_mat_age)]))
        self.sigma_age = np.sqrt(np.diag(c_model)).reshape((self.njac+1, self.njac+1))   #*self.jac_mat_age
        self.age_total += time.perf_counter() - age_before

        return

    # Residuals function
    def residuals(self, vari):

        var = np.array(vari).reshape((3,-1))

        # variables (var) given are ln(a), p' and ln(H)
        # seperate variables to appropriate array
        self.a_try = np.exp(var[0])
        self.p_prime_try = var[1]                       # variable not needed
        self.p_try = np.exp(var[1])-1					# transform to p in xz coords
        #self.H_try = np.exp(var[2])					    # H0 = inital observed ice thickness
        # self.H_try = np.copy(self.H_inv)
        # self.m_try = np.zeros_like(self.a_try)

        self.Delta_try = var[2]
        self.H_try = np.where(self.Delta_try < 0, 
                          self.H_inv * (1 + self.Delta_try), 
                          self.H_inv)
    
        self.m_try = np.where(self.Delta_try >= 0, 
                          self.a_try * self.Delta_try, 
                          0.0)
        
        # interpolate on to new x coords
        self.interpolation()

        # calc new matrices
        self.matrices()

        # run forward model
        self.age_calc()

        # calculate residuals at nobs points
        resi = (self.iso_obs_age-self.iso_mod_age)/self.iso_obs_age_sigma

        # resi = (np.log(self.iso_obs_age)-np.log(self.iso_mod_age))/self.iso_obs_age_sigma *self.iso_obs_age
        self.resi = resi[~np.isnan(resi)]

        resa = ((np.log(self.a_try)-np.log(self.a_prior))/\
               self.a_sigma).flatten()

        resp = ((self.p_prime_try-np.log(self.p_prior+1))/\
               self.p_prime_sigma).flatten()
            
        resd = ((self.Delta_try- self.Delta_inv)/ self.Delta_sigma).flatten()
        

        # resh = ((np.log(self.H_try)-np.log(self.H_inv))/\
        #                      self.H_sigma).flatten()

        #resi = np.concatenate((self.resi, resa, resp, resh))
        resi = np.concatenate((self.resi, resa, resp, resd))
        

        resi = resi[np.where(~np.isnan(resi))]


        for name in self.ic:
            if self.ic[name]['constraint'] is not None:
                resic = (self.ic[name]['obs_age']-self.ic[name]['mod_age'])/self.ic[name]['obs_age_sigma']
                resi = np.concatenate((resi,resic))



        return resi.flatten()

    # optimisation function
    def optimise(self):

        print('Starting optimisation')

        inf_bound = np.ones(len(self.a_inv))*np.inf
        zero_bound = np.zeros(len(self.a_inv))
        #min_Hbound_old = np.log(np.interp(self.x_inv,self.cp_iso_x, self.cp_iso_depth[-1]))
        epsilon = 1e-6
        
        # radar_max_depth = np.array([np.nanmax(col) if np.any(~np.isnan(col)) else 0.0 
        #     for col in self.cp_iso_depth.T])

        #max_iso = np.interp(self.x_inv, self.cp_iso_x, self.cp_iso_depth[-1]) c'est la bonne ligne 
       
        # raw_max_iso = np.zeros(len(self.cp_iso_x))
        
        raw_max_iso = np.array([np.nanmax(self.cp_iso_depth[:, i]) if np.any(~np.isnan(self.cp_iso_depth[:, i])) 
                                else 0.0 for i in range(len(self.cp_iso_x))])
        max_iso = np.interp(self.x_inv, self.cp_iso_x, raw_max_iso)
        
        #max_iso = np.array([np.nanmax(self.cp_iso_depth[:, x]) if np.any(~np.isnan(self.cp_iso_depth[:, x])) 
                        #else 0.0 for x in range(len(self.x_inv))])
        #min_iso = np.array([np.nanmax(self.cp_iso_depth[:,x]) for x in range(len(self.cp_iso_x))])
        delta_min = (max_iso / self.H_inv) - 1.0 + epsilon
        delta_max = np.ones_like(self.H_inv) - epsilon
        #min_Hbound = np.log(np.interp(self.x_inv,self.cp_iso_x, min_iso))
        # print(min_Hbound)
        # min_Hbound = np.log(np.interp(self.x_inv,self.cp_iso_x, self.cp_iso_depth[-4]))   #for discontinuous
        #all_bounds = (np.array([-inf_bound,zero_bound,min_Hbound]).flatten() , np.array([inf_bound,inf_bound,inf_bound]).flatten())
        bounds = (np.array([-inf_bound, zero_bound, delta_min]).flatten() , np.array([inf_bound, inf_bound, delta_max]).flatten())
        
        #H_init = np.nan_to_num(self.H_inv, nan=3000)
        #self.variables = np.array([np.log(self.a_inv), self.p_prime_inv, np.log(H_init)]).flatten()
        self.variables = np.array([np.log(self.a_inv), self.p_prime_inv, self.Delta_inv]).flatten()

        # do least square fit to get variables and hessian matrix
        # leastsq_fit1D = least_squares(self.residuals, self.variables, bounds=([-np.inf, -np.inf, m.log(max_iso_depth)], [np.inf, np.inf, np.inf]), args=(j,), method='trf')
        #leastsq_fit = least_squares(self.residuals, self.variables, bounds = all_bounds, method='trf', verbose=2)
        leastsq_fit = least_squares(self.residuals, self.variables, bounds = bounds, method='trf', verbose=2)

        self.variables = leastsq_fit.x
        
        self.variables = np.array(self.variables).reshape((3,-1))
        if self.calc_jac:
            self.hess = np.linalg.inv(np.dot(np.transpose(leastsq_fit.jac), leastsq_fit.jac))
            if np.size(self.hess) != 1:
                self.sigma()
        else:
            self.sigma_a = np.zeros_like(self.x_inv)
            self.sigma_Delta = np.zeros_like(self.x_inv)
            self.sigma_m = np.zeros_like(self.x_inv)
            self.sigma_p = np.zeros_like(self.x_inv)
            self.sigma_age = np.zeros_like(self.x_inv)
                #self.melt_calc()
        
       
        #self.refrozen()

    # extract data at specified ice core site
    def ice_core(self):
        # ----------------------------------------------------------
        # Post-processing: transfering of the modeling results
        # on the 1D grid of the drill sites
        # ----------------------------------------------------------

        print('Before calculating for the ice cores',
              round(time.perf_counter()-self.START_TIME, 4), 's.')

        for name in self.ic:

            # ---------------------------------------------------
            # Depth_ic and ie_depth_ic at drill site
            # ---------------------------------------------------
            self.ic[name]['depth'] = np.arange(0., self.ic[name]['max_depth'] + 0.0001,
                                          self.ic[name]['step_depth'])
            ie_depth_ic = np.interp(self.ic[name]['depth'], self.D_depth, self.D_depth_ie)

            # -------------------------------------------------------
            # Calculation of the surrounding nodes for the ice core
            # -------------------------------------------------------

            if self.ic[name]['x'] > self.x_right:
                print(name, "is downstream of the domain.")
                self.error.append(name)
                continue
            elif self.ic[name]['x'] < self.x[0]:
                print(name, "is upstream of the domain.")
                self.error.append(name)
                continue
            elif self.ic[name]['x'] == self.x_right:
                ggrid = self.grid_age[:, self.imax]
                ddepth_ie = self.mat_depth_ie[:, self.imax][ggrid]
                OOMEGA = self.mat_OMEGA[:, self.imax][ggrid]
                aa0 = self.mat_a0[:, self.imax][ggrid]
                ssteady_age = self.mat_steady_age[:, self.imax][ggrid]
                xx0 = self.mat_x0[:, self.imax][ggrid]
                #ttau = self.mat_tau_anal[:, self.imax][ggrid]
                ttheta = self.theta[ggrid]
                self.ic[name]['S'] = self.S[self.imax]
            else:
                ii = np.argmax(self.x[self.x <= self.ic[name]['x']])
                self.inter = (self.x[ii+1]-self.ic[name]['x']) / (self.x[ii+1] - self.x[ii])
                ggrid = np.logical_and(self.grid_age[:, ii], self.grid_age[::, ii+1])
                ddepth_ie = self.inter * self.mat_depth_ie[:, ii][ggrid] +\
                    (1-self.inter) * self.mat_depth_ie[:, ii+1][ggrid]
                OOMEGA = self.inter * self.mat_OMEGA[:, ii][ggrid] +\
                    (1-self.inter) * self.mat_OMEGA[:, ii+1][ggrid]
                aa0 = self.inter * self.mat_a0[:, ii][ggrid] +\
                    (1-self.inter) * self.mat_a0[:, ii+1][ggrid]
                ssteady_age = self.inter * self.mat_steady_age[:, ii][ggrid] +\
                    (1-self.inter) * self.mat_steady_age[:, ii+1][ggrid]
                xx0 = self.inter * self.mat_x0[:, ii][ggrid] + (1-self.inter) * self.mat_x0[:, ii+1][ggrid]
                #ttau = self.inter * self.mat_tau_anal[:, ii][ggrid] +\
                #    (1-self.inter) * self.mat_tau_anal[:, ii+1][ggrid]
                ttheta = self.theta[ggrid]
                self.ic[name]['S'] = self.inter * self.S[ii] + (1-self.inter) * self.S[ii+1]
                self.ic[name]['PI'] = self.pi[ii] + self.inter*(self.pi[ii+1]-self.pi[ii])

            # ----------------------------------------------------------
            #  Computation of theta for the ice core: theta_ic
            # ----------------------------------------------------------
            if self.mat_depth_ie[self.imax, ii] < ie_depth_ic[-1]:
                print("The mesh does not extend down to the bottom of the",name, "core.")
                self.ic[name]['depth'][self.ic[name]['depth']>self.mat_depth_ie[self.imax, self.imax]] = np.nan

            self.ic[name]['theta'] = np.log(np.interp(ie_depth_ic, ddepth_ie, OOMEGA))

            # ----------------------------------------------------------
            #  Computation steady a0 and x0 for the ice core
            # ----------------------------------------------------------

            # Be careful, xp must be in increasing order for np.interp
            steady_a0_ic = np.interp(-self.ic[name]['theta'], -ttheta, aa0)
            self.ic[name]['x0'] = np.interp(-self.ic[name]['theta'], -ttheta, xx0)

            # ----------------------------------------------------------
            #  Computation of steady_age icecore
            # ----------------------------------------------------------

            # Cubic spline with derivative constraint at surface
            # We had a point close to the surface to impose the derivative of the age

            new_ttheta = np.insert(ttheta, 1, -1/1000000)
            chi_0 = np.insert(ssteady_age, 1,
                              1/(steady_a0_ic[0])*(ie_depth_ic[1] - ie_depth_ic[0]) /
                              (self.ic[name]['theta'][0] - self.ic[name]['theta'][1]) / 1000000)

            steady_age_ic = interp1d(-new_ttheta, chi_0, assume_sorted=True,
                                     kind='cubic')(-self.ic[name]['theta'])
            steady_age_sigma_ic = interp1d(-new_ttheta, chi_0, assume_sorted=True,
                                     kind='cubic')(-self.ic[name]['theta'])

            # ----------------------------------------------------------
            #  Computation of age for the ice core
            # ----------------------------------------------------------

            self.ic[name]['age'] = np.interp(steady_age_ic+self.age_surf, self.steady_age_R, self.age_R)
            print('Bottom age for the', name, 'ice core:', self.ic[name]['age'][-1])

            # ----------------------------------------------------------
            #  a0_ic
            # ----------------------------------------------------------
            # Here, steady_a0_ic is at the node, while a0_ic is for an interval
            self.ic[name]['a0'] = (steady_a0_ic[1:]+steady_a0_ic[:-1])/2 *\
                self.interp_stair_aver(steady_age_ic, self.steady_age_R, self.R)

            # ----------------------------------------------------------
            #  Computation of tau_ic for the ice core
            # ----------------------------------------------------------

            aa = (steady_a0_ic[1:]+steady_a0_ic[:-1]) / 2
            self.ic[name]['tau'] = (ie_depth_ic[1:] - ie_depth_ic[:-1]) / aa / \
                (steady_age_ic[1:] - steady_age_ic[:-1])

            # ----------------------------------------------------------
            #  Computation of age density and max age for the ice core
            # ----------------------------------------------------------
            self.ic[name]['dens'] = 1/self.ic[name]['a0'][:-2] * (1/self.ic[name]['tau'][1:-1] + 1/self.ic[name]['tau'][:-2]/2)
            self.ic[name]['dens'] = np.append(self.ic[name]['dens'],[np.nan,np.nan,np.nan])
            self.ic[name]['max_age'] = np.interp(self.dens_lim, self.ic[name]['dens'], self.ic[name]['age'] )
            # self.ic[name]['max_sigma_age'] = np.interp(self.dens_lim, self.ic[name]['dens'], self.ic[name]['sigma_age'] )
            #self.ic[name]['max_depth'] = np.interp(self.dens_lim, self.ic[name]['dens'], self.ic[name]['depth'] )
            mask = ~np.isnan(self.ic[name]['dens'])
            self.ic[name]['max_depth'] = np.interp(self.dens_lim, self.ic[name]['dens'][mask], self.ic[name]['depth'][mask])
            print(name, 'max_age',self.ic[name]['max_age'], '(yrs)\n' ,'max_age_depth', self.ic[name]['max_depth'])
            # ----------------------------------------------------------
            # Output for the ice cores
            # ----------------------------------------------------------
            self.ic[name]['a0'] = np.append(self.ic[name]['a0'], np.nan)
            self.ic[name]['tau'] = np.append(self.ic[name]['tau'], np.nan)
            self.ic[name]['stag'] = np.interp(self.ic[name]['x'], self.x_inv, self.stagnant)
            self.ic[name]['melting'] = np.interp(self.ic[name]['x'], self.x_inv, self.m_try)
            self.ic[name]['p'] = np.interp(self.ic[name]['x'], self.x_inv, self.p_try)
            self.ic[name]['steady_accu'] = np.interp(self.ic[name]['x'], self.x_inv, self.a_try)
            self.ic[name]['H_obs'] = np.interp(self.ic[name]['x'], self.x_inv, self.H_inv)
            #self.ic[name]['H_ref'] = np.interp(self.ic[name]['x'], self.x_inv, self.H_ref)
            self.ic[name]['H_stag'] = np.interp(self.ic[name]['x'], self.x_inv, self.H_try)
            self.ic[name]['vel_v'] = self.ic[name]['tau'] * self.ic[name]['steady_accu']


            header = name + ' Maximum age (yrs): '+str(self.ic[name]['max_age']) \
                            + '\nDepth of maxiumum age (m): '+ str(self.ic[name]['max_depth'])  \
                            +'\nObserved bedrock depth (m): ' +str(self.ic[name]['H_obs']) \
                            +'\nStagnant ice depth (m): ' +str(self.ic[name]['H_stag']) \
                            +'\nBasal melt rate (m/yr): ' + str(self.ic[name]['melting']) \
                            +'\nSteady Accumulation (m/yr): '+ str(self.ic[name]['steady_accu']) \
                            +'\np: '+ str(self.ic[name]['p']) +'\n'
                            #+'\nTotal basal layer thickness (stagnant+accreted)(m): ' +str(self.ic[name]['H_obs']-self.ic[name]['H_ref']) \
                            #+'\nAccreted ice depth (m): ' +str(self.ic[name]['H_ref']) \
            if self.output_ic:
                output = np.vstack((self.ic[name]['depth'], self.ic[name]['age'],
                                    self.ic[name]['tau'],
                                    self.ic[name]['a0'],
                                    self.ic[name]['x0'],
                                    steady_a0_ic, self.ic[name]['dens'],
                                    self.ic[name]['vel_v']))    #, self.ic[name]['dens']
                np.savetxt(self.label+name+'_ice_core_output.txt', np.transpose(output),
                           delimiter='\t', header=header+"depth\tage\tthinning\taccu\tx_origin\taccu_steady\tage_density\tvertical_vel")

            # -----------------------------------------------------------
            # Reading of comparison files
            # -----------------------------------------------------------
            if self.ic[name]['comp'] is not None:
                self.ic[name]['cp_depth'], self.ic[name]['cp_age'], self.ic[name]['cp_x'],\
                    self.ic[name]['cp_tau'] = np.loadtxt(self.label+self.ic[name]['comp'],
                                                    unpack=True)

        for error in self.error:
            self.ic.pop(error)

        # ----------------------------------------------------------
        # Output quantities along the flow line
        # ----------------------------------------------------------
        if self.output_fl:
            output = np.vstack((self.x, self.Q, self.a, self.ux_surf))
            np.savetxt(self.label+'flow_line_output.txt', np.transpose(output),
                       delimiter='\t',header='x(km)\ttotal_flux\taccu(m/yr)\tsurf_velocity(m/yr)')

    # save data to .csv files
    def save_data(self):

        # save inverted parameters and all other related values
        if self.inversion:
            # output = np.vstack((self.x_inv, self.a_try, self.sigma_a, self.p_try, self.sigma_p, self.H_try, self.sigma_h, self.m_inv, self.sigma_m, self.H_ref, self.H_inv)).T
            # np.savetxt(self.label+'inverted_results.txt', output, delimiter='\t', header='x(m),a, sigma_a,p, sigma_p,H, sigma_H, m,sigma_m, refrozen, H_obs')
            
            output = np.vstack((self.x_inv, self.a_try, self.sigma_a, self.p_try, self.sigma_p, self.H_try, self.m_inv, self.sigma_m, self.H_inv)).T
            np.savetxt(self.label+'inverted_results.txt', output, delimiter='\t', header='x(m),a, sigma_a,p, sigma_p,H, m,sigma_m, H_obs')

            self.stagnant = self.H_inv - self.H_try
            self.stagnant[self.stagnant<0] = np.nan
            output = np.vstack((self.x_inv, self.H_inv, self.H_try, self.stagnant, self.m_inv)).T
            np.savetxt(self.label+'stagnant.txt', output, delimiter='\t', header='x(m),Hobs, Hinverted, stagnant (m), melt rate')
    
            #parameters save
            header_base = '#x_inv(km)'
            
            output = np.vstack((self.x_inv, self.a_try))
            with open(self.label + 'accumulation_opt.txt', 'w') as f:
                f.write(header_base + '\taccu_opt(ice-m/yr)\n')
                np.savetxt(f, np.transpose(output), delimiter='\t')
    
            output = np.vstack((self.x_inv, self.p_try))
            with open(self.label + 'p_Lliboutry_opt.txt', 'w') as f:
                f.write(header_base + '\tp_opt\n')
                np.savetxt(f, np.transpose(output), delimiter='\t')
        
            output = np.vstack((self.x_inv, self.Delta_try))
            with open(self.label + 'Delta_opt.txt', 'w') as f:
                f.write(header_base + '\tDelta_opt\n')
                np.savetxt(f, np.transpose(output), delimiter='\t')
                
            
        # for checking the distance between x nodes
        self.for_dist = self.x[1:] - self.x[:-1]
        output = np.vstack((self.x[:-1], self.for_dist))
        np.savetxt(self.label+'x_mesh.txt', np.transpose(output),
                   delimiter='\t',header='x(km)\tmesh_width(m)')
        
      

    print('Optimised parameters saved to accumulation_opt.txt, '
          'p_Lliboutry_opt.txt, Delta_opt.txt')

    # plot figures showing results
    def plot_figs(self):

        # -----------
        # FIGURES
        # -----------
        # Note: We don't plot refrozen ice since the mesh does not always extend to it.
        if self.create_figs:

            print('Before creating figures.',
                  round(time.perf_counter()-self.START_TIME, 4), 's.')

            for name in self.ic:

                self.ic[name]['XX'] = self.ic[name]['x'] * np.ones(2)
                self.ic[name]['ZZ'] = np.array([self.ic[name]['S'],
                                           self.ic[name]['S']-self.ic[name]['max_depth']])
                self.ic[name]['DD'] = np.array([0, self.ic[name]['max_depth']])
                self.ic[name]['PP'] = self.ic[name]['PI']*np.ones(2)
                self.ic[name]['TT'] = np.array([0, self.ic[name]['theta'][-1]])
            color_core = 'r'
            lw_core = 2
            ls_core = 'dashed'

            # ----------------------------------------------------------
            # Display of (pi,theta) mesh
            # ----------------------------------------------------------

            fig, ax = plt.subplots(figsize=(12, 6))
            plt.vlines(self.pi, self.theta_min_mesh, self.theta_max, color='k', linewidths=0.1)
            for i in range(0, self.imax+1):
                plt.plot(self.pi, self.mat_theta[i, :], color='k', linewidth=0.1)
            plt.xlabel(r'$\pi$', fontsize=18)
            plt.ylabel(r'$\theta$', fontsize=18)
            for name in self.ic:
                plt.plot(self.ic[name]['PP'], self.ic[name]['TT'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['PP'][0], 0.03), ha='center',
                             va='bottom', color=color_core)
            plt.savefig(self.label+'mesh_pi_theta.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Display of (x, z) mesh
            # ----------------------------------------------------------

            fig, ax = plt.subplots(figsize=(15, 5))
            plt.plot(self.x[self.x>self.x_left], self.S[self.x>self.x_left], label='Surface', color='0')
            plt.plot(self.x[self.x>self.x_left], self.B[self.x>self.x_left], label='Bedrock', color='0')
            # The vertical grid step can increase near the bed.
            # This is due do iso-omega layers being thicker near the bed.
            for i in range(0, self.imax+1):
                plt.scatter(self.x[self.x>self.x_left], self.mat_z[i,self.mat_x[0]>self.x_left],  ls='-', color='k', linewidth=0.1)
            plt.vlines(self.x[self.x>self.x_left], self.z_ie_min_mesh[self.x>self.x_left], self.S[self.x>self.x_left], color='k', linewidths=0.1)
            plt.xlabel(r'$x$ (km)', fontsize=18)
            plt.ylabel(r'$z$ (m)', fontsize=18)
            for name in self.ic:
                plt.plot(self.ic[name]['XX'], self.ic[name]['ZZ'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['x'], self.ic[name]['S']+50), ha='center',
                             va='bottom', color=color_core)
            plt.savefig(self.label+'mesh_x_z.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # -------------------------------------------------------------------------
            # Boundary conditions of the flow in x
            # -------------------------------------------------------------------------

            fig, ax = plt.subplots()
            fig.set_size_inches(15, 5)
            fig.subplots_adjust(right=0.8)
            ax.set_xlabel('x (km)', fontsize=18)
            axic = ax.secondary_xaxis("top")
            axic.set_xticks(ticks=[self.ic[name]['x'] for name in self.ic], labels=self.ic.keys())
            axic.tick_params(colors='r')

            ax.set_ylabel('Y (relative unit)')
            ax.plot(self.x, self.Y, color='k')
            ax.spines.right.set_visible(False)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)

            ax1 = ax.twinx()
            ax1.spines['right'].set_position(('axes', 1.))
            ax1.spines['right'].set_color('g')
            ax1.plot(self.x, self.a, color='g')
            ax1.set_ylabel('a (m/yr)', color='g')
            ax1.tick_params(axis='y', colors='g')

            ax2 = ax.twinx()
            ax2.spines['right'].set_position(('axes', 1.09))
            ax2.spines['right'].set_color('b')
            ax2.plot(self.x, self.m, color='b')
            ax2.set_ylabel('m (m/yr)', color='b')
            ax2.tick_params(axis='y', colors='b')
            plt.savefig(self.label+'boundary_conditions_x.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Calculated quantities along the flow line
            # ----------------------------------------------------------

            fig, ax = plt.subplots()
            fig.set_size_inches(15, 5)
            fig.subplots_adjust(right=0.8)
            ax.set_xlabel('x (km)', fontsize=18)
            axic = ax.secondary_xaxis("top")
            axic.set_xticks(ticks=[self.ic[name]['x'] for name in self.ic], labels=self.ic.keys())
            axic.tick_params(colors='r')

            ax.set_ylabel('Q (relative unit)')
            ax.plot(self.x, self.Q, color='k')
            ax.spines.right.set_visible(False)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)

            ax1 = ax.twinx()
            ax1.spines['right'].set_position(('axes', 1.))
            ax1.spines['right'].set_color('g')
            ax1.plot(self.x, self.a*self.R[0], color='g')
            ax1.set_ylabel('a (m/yr)', color='g')
            ax1.tick_params(axis='y', colors='g')

            ax2 = ax.twinx()
            ax2.spines['right'].set_position(('axes', 1.09))
            ax2.spines['right'].set_color('r')
            ax2.plot(self.x, self.ux_surf*self.R[0], color='r')
            if self.comp_flowline is not None:
                ax2.plot(self.cp_fl_x, self.cp_fl_ux_surf, color='r', linestyle='dashed')
            ax2.set_ylabel('surface velocity (m/yr)', color='r')
            ax2.tick_params(axis='y', colors='r')
            plt.savefig(self.label+'surface_velocity_x.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Display of iso-omega lines in (x, z)
            # ----------------------------------------------------------

            fig, ax = plt.subplots(figsize=(12, 6))
            plt.plot(self.x, self.S, label='Surface', color='0')
            plt.plot(self.x, self.B, label='Bedrock', color='0')
            levels = np.arange(0, 1.01, 0.01)
            levels_cb = np.arange(0, 11, 1)/10.
            # There is no node on the bedrock, so the color does not go down there.
            cp = plt.contourf(self.mat_x, self.mat_z, self.mat_OMEGA,
                              levels=levels,
                              cmap='jet')
            cp2 = plt.contour(self.mat_x, self.mat_z, self.mat_OMEGA,
                              levels=levels_cb,
                              colors='k')
            # cp = plt.contourf(self.mat_x, self.mat_z, self.mat_omega,
            #                   levels=levels,
            #                   cmap='jet')
            # cp2 = plt.contour(self.mat_x, self.mat_z, self.mat_omega,
            #                   levels=levels_cb,
            #                   colors='k')
            
            cb = plt.colorbar(cp)
            cb.set_ticks(levels_cb)
            cb.set_ticklabels(levels_cb)
            cb.add_lines(cp2)
            cb.set_label(r'$\omega$')
            ax.set_xlabel(r'$x$ (km)', fontsize=19)
            ax.set_ylabel(r'$z$ (m)', fontsize=19)
            for name in self.ic:
                plt.plot(self.ic[name]['XX'], self.ic[name]['ZZ'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['x'], self.ic[name]['S']+50), ha='center',
                             va='bottom', color=color_core)
            plt.savefig(self.label+'iso-omega_lines.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Display of age and isochrones in (x, z)
            # ----------------------------------------------------------

            fig, ax = plt.subplots(figsize=(12, 6))

            plt.plot(self.x, self.S, label='Surface', color='0')
            plt.plot(self.x, self.B, label='Bedrock', color='0')

            levels = np.arange(0, self.fig_age_max, self.fig_age_spacing)
            levels_cb = np.arange(0, self.fig_age_max, self.fig_age_spacing_labels)
            levels_iso = np.array(self.fig_age_iso)
            cp = plt.contourf(self.mat_x, self.mat_z, self.mat_age/1000.,
                              levels=levels,
                              cmap='jet')
            cp2 = plt.contour(self.mat_x, self.mat_z, self.mat_age/1000.,
                              levels=levels_iso,
                              colors='k')
            # Corner trajectory
            level0 = np.array([self.Q[0]])
            plt.contour(self.mat_x, self.mat_z, self.mat_q, colors='k', linestyles='dashed',
                        levels=level0, linewidths=1)
            cb = plt.colorbar(cp)
            cb.set_ticks(levels_cb)
            cb.set_ticklabels(levels_cb)
            cb.add_lines(cp2)
            ax.clabel(cp2)
            cb.set_label('Modeled age (kyr)')
            ax.set_xlabel(r'$x$ (km)')
            ax.set_ylabel(r'$z$ (m)')
            ax.grid()
            if self.comp_isochrones is not None:
                for i in range(self.cp_iso_nb):
                    plt.plot(self.cp_iso_x, np.interp(self.cp_iso_x, self.x, self.S) - self.cp_iso_depth[i, :],
                             color='k', linestyle='dashed')
            for name in self.ic:
                plt.plot(self.ic[name]['XX'], self.ic[name]['ZZ'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['x'], self.ic[name]['S']+50), ha='center',
                             va='bottom', color=color_core)
            plt.savefig(self.label+'age_x_z.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Display of age and isochrones in (x, depth)
            # ----------------------------------------------------------

            self.H = np.interp(self.x, self.x_inv, self.H_try)

            fig, ax = plt.subplots(figsize=(12, 6))
            # fig, ax = plt.subplots(figsize=(5, 6))

            plt.plot(self.x[self.x>self.x_left], np.zeros_like(self.S)[self.x>self.x_left], label='Surface', color='0')

            levels = np.arange(0, self.fig_age_max, self.fig_age_spacing)
            levels_cb = np.arange(0, self.fig_age_max, self.fig_age_spacing_labels)
            levels_iso = np.array(self.fig_age_iso)
            # modelled ages
            cp = plt.contourf(self.mat_x[:,self.mat_x[0]>self.x_left], self.mat_depth[:,self.mat_x[0]>self.x_left], self.mat_age[:,self.mat_x[0]>self.x_left]/1000.,
                              levels=levels,
                              cmap='jet')

            cp2 = ax.contour(self.mat_x[:,self.mat_x[0]>self.x_left], self.mat_depth[:,self.mat_x[0]>self.x_left], self.mat_age[:,self.mat_x[0]>self.x_left]/1000.,
                              levels=levels_iso,
                              colors='k',zorder=2)

            if self.inversion:
                #plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], color='darkblue', zorder=1)    #refrozen ice
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left], label='Inverted bedrock', color='darkviolet', zorder=2)    #inverted Bedrock
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], label='Observed bedrock', color='0', linewidth=2)    # observed Bedrock

                #plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    #where=self.H_try[self.x_inv>self.x_left]>self.H_ref[self.x_inv>self.x_left], color='#8B8BC6', label='refrozen ice',interpolate=True,zorder=-1, edgecolor=None)
                plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    where=self.H_try[self.x_inv>self.x_left]<self.H_inv[self.x_inv>self.x_left], color='0.7',label='stagnant ice',interpolate=True, edgecolor=None)
                plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    where=self.H_try[self.x_inv>self.x_left]>self.H_inv[self.x_inv>self.x_left], color='white', label='bedrock',interpolate=True, edgecolor=None)
            else:
                plt.plot(self.x[self.x>self.x_left], self.H[self.x>self.x_left], label='Bedrock', color='0')
            cb = plt.colorbar(cp)
            cb.set_ticks(levels_cb)
            cb.set_ticklabels(levels_cb)
            # cb.add_lines(cp2)
            ax.clabel(cp2)
            cb.set_label('Modeled age (kyr)')
            ax.set_xlabel(r'Distance (km)')
            ax.set_ylabel(r'Depth (m)')


            ax.invert_yaxis()

            ax.grid()
            if self.comp_isochrones is not None:
                for i in range(self.cp_iso_nb):
                    plt.plot(self.cp_iso_x, self.cp_iso_depth[i, :],
                             color='white', zorder=1)
            for name in self.ic:
                plt.plot(self.ic[name]['XX'], self.ic[name]['DD'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['x'], -50), ha='center', va='bottom',
                             color=color_core)


            ax.grid(color='gray', alpha=0.3)
            # for DC
            x1,x2,y1,y2 = plt.axis()
            # plt.axis((4.,x2,y1,y2))

            plt.savefig(self.label+'age_x_depth.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Display of age sigma and isochrones in (x, depth)
            # ----------------------------------------------------------
            if self.calc_jac:
                if self.inversion:
                    fig, ax = plt.subplots(figsize=(12, 6))

                    plt.plot(self.x[self.x>self.x_left], np.zeros_like(self.S)[self.x>self.x_left], label='Surface', color='0')

                    #plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], color='darkblue', zorder=1)    #refrozen ice
                    plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left], label='Inverted bedrock', color='darkviolet', zorder=2)    #inverted Bedrock
                    plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], label='Observed bedrock', color='0', linewidth=2)    # observed Bedrock

                    #plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                        #where=self.H_try[self.x_inv>self.x_left]>self.H_ref[self.x_inv>self.x_left], color='#8B8BC6', label='refrozen ice',interpolate=True,zorder=-1)
                    plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                        where=self.H_try[self.x_inv>self.x_left]<self.H_inv[self.x_inv>self.x_left], color='0.7',label='stagnant ice',interpolate=True)
                    plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                        where=self.H_try[self.x_inv>self.x_left]>self.H_inv[self.x_inv>self.x_left], color='white', label='bedrock',interpolate=True, edgecolor=None)

                    levels = np.arange(0, self.fig_age_max, self.fig_age_spacing)
                    levels_cb = np.arange(0, self.fig_age_max, self.fig_age_spacing_labels)
                    levels_iso = np.array(self.fig_age_iso)
                    levels_sigma_log = np.arange(2, 6, 0.1)
                    levels_sigma = np.power(10, levels_sigma_log)

                    # modelled isochrones
                    cp2 = plt.contour(self.mat_x[:,self.mat_x[0]>self.x_left], self.mat_depth[:,self.mat_x[0]>self.x_left], self.mat_age[:,self.mat_x[0]>self.x_left]/1000.,
                                      levels=levels_iso,
                                      colors='k',zorder=2)
                    # Corner trajectory
                    level0 = np.array([self.Q[0]])
                    sigma = plt.contourf(self.jac_mat_x[:,self.jac_mat_x[0]>self.x_left],
                                self.jac_mat_depth[:,self.jac_mat_x[0]>self.x_left],
                                self.sigma_age[:,self.jac_mat_x[0]>self.x_left],
                                levels=levels_sigma,
                                cmap='viridis',zorder=-1.5,
                                norm=LogNorm())
                    # plt.fill_between(self.x_inv, self.H_inv, np.max(self.jac_mat_depth),
                    #     color='white', label='bedrock',zorder=1)
                    # plt.fill_between(self.x_inv, self.H_try, np.max(self.jac_mat_depth),
                    #     where=self.H_try<self.H_inv, color='white', label='bedrock',zorder=1)

                    # set colorbar ticks
                    cb = plt.colorbar(sigma)
                    levels_sigma_labels = np.array([])
                    for i in np.arange(2, 6, 1):
                        levels_sigma_labels = np.concatenate((levels_sigma_labels,
                                                        np.array([10**i/1000, '', '', '', '', '', '', '', '', ''])))
                    levels_ticks = np.concatenate((np.arange(100, 1100, 100),
                                                   np.arange(1000, 11000, 1000),
                                                   np.arange(10000, 110000, 10000),
                                                   np.arange(100000, 1100000, 100000)))

                    cb.set_ticks(levels_ticks[:-5])
                    cb.set_ticklabels(levels_sigma_labels[:-5])

                    cb.set_label('Modeled age sigma (kyr)')
                    ax.set_xlabel(r'Distance (km)')
                    ax.set_ylabel(r'Depth (m)')
                    ax.invert_yaxis()


                    if self.comp_isochrones is not None:
                        for i in range(self.cp_iso_nb):
                            plt.plot(self.cp_iso_x, self.cp_iso_depth[i, :],
                                     color='white', zorder=1)
                    for name in self.ic:
                        plt.plot(self.ic[name]['XX'], self.ic[name]['DD'], linewidth=lw_core,
                                 color=color_core, linestyle=ls_core)
                        plt.annotate(name, (self.ic[name]['x'], -50), ha='center', va='bottom',
                                     color=color_core)

                    ax.grid(color='gray', alpha=0.3)
                    x1,x2,y1,y2 = plt.axis()

                    plt.savefig(self.label+'age_sigma.'+self.fig_format,
                                format=self.fig_format, bbox_inches='tight')

            # ---------------------------------------------------------------------
            # Display of age and isochrones in (pi,theta)
            # ---------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(12, 6))

            plt.plot(self.pi, self.theta_max, label='Surface', color='0')
            plt.plot(self.pi, self.theta_min_mesh, label='Bedrock', color=None)

            levels = np.arange(0, self.fig_age_max, self.fig_age_spacing)
            levels_cb = np.arange(0, self.fig_age_max, self.fig_age_spacing_labels)
            cp = plt.contourf(self.mat_pi, self.mat_theta, self.mat_age/1000., levels=levels,
                              cmap='jet')
            print(self.mat_age)
            cp2 = plt.contour(self.mat_pi, self.mat_theta, self.mat_age/1000.,
                              levels=levels_iso, colors='k')
            # # # Corner trajectory
            # # level0 = np.array([self.Q[0]])
            # # plt.contour(self.mat_pi, self.mat_theta, self.mat_q, colors='k', linestyles='dashed',
            # #             levels=level0, linewidths=1)
            # cb = plt.colorbar(cp)
            # cb.set_ticks(levels_cb)
            # cb.set_ticklabels(levels_cb)
            # cb.add_lines(cp2)
            # ax.clabel(cp2)
            # cb.set_label('Modeled age (kyr)')
            ax.set_xlabel(r'$\pi$', fontsize=19)
            ax.set_ylabel(r'$\theta$', fontsize=19)
            ax.grid()
            for name in self.ic:
                plt.plot(self.ic[name]['PP'], self.ic[name]['TT'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['PP'][0], 0.03), ha='center',
                             va='bottom', color=color_core)
            plt.savefig(datadir+'age_pi_theta.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Display of thinning function - analytical formula
            # ----------------------------------------------------------

        # FIXME: Check why the z-axis is different from the previous figure.

            fig, ax = plt.subplots(figsize=(12, 6))

            plt.plot(self.x, self.S, label='Surface', color='0')
            plt.plot(self.x, self.B, label='Bedrock', color='0')
            levels = np.arange(0, 1.21, 0.01)
            levels_cb = np.arange(0, 13, 1)/10.
            cp = plt.contourf(self.mat_x, self.mat_z, self.mat_tau_anal,
                              levels=levels,
                              cmap='jet')
            cp2 = plt.contour(self.mat_x, self.mat_z, self.mat_tau_anal,
                              levels=levels_cb,
                              colors='k')
            # Corner trajectory
            level0 = np.array([self.Q[0]])
            plt.contour(self.mat_x, self.mat_z, self.mat_q, colors='k', linestyles='dashed',
                        levels=level0, linewidths=1)
            cb = plt.colorbar(cp)
            cb.set_ticks(levels_cb)
            cb.set_ticklabels(levels_cb)
            cb.add_lines(cp2)
            cb.set_label('Thinning function (no unit)')
            ax.set_xlabel(r'$x$ (km)', fontsize=19)
            ax.set_ylabel(r'$z$ (m)', fontsize=19)
            ax.grid()
            for name in self.ic:
                plt.plot(self.ic[name]['XX'], self.ic[name]['ZZ'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['x'], self.ic[name]['S']+50), ha='center',
                             va='bottom', color=color_core)
            plt.savefig(self.label+'thinning_analytical_x_z.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Display of stream lines
            # ----------------------------------------------------------

            fig, ax = plt.subplots(figsize=(12,6))
            # Rmq: We use plt.contour instead of plotting the individual lines, since
            # it is simpler and slightly faster.
            # Rmq2: We don't exactly go down to the bedrock here but this is normal.
            # Trajectories that come from the surface and traj that come from the dome.
            levels = np.concatenate((self.Q[-1:0:-self.traj_step],
                                     self.mat_q[0::self.traj_step, 0]))
            levels = np.flip(levels[~np.isnan(levels)])
            color = 'k'
            lw = 0.5
            plt.contour(self.mat_x[:,self.mat_x[0]>self.x_left], self.mat_depth[:,self.mat_x[0]>self.x_left], self.mat_q[:,self.mat_x[0]>self.x_left], colors=color,
                        levels=levels, linewidths=lw,zorder=3)

            # # Corner trajectory
            # level0 = np.array([self.Q[0]])
            # plt.contour(self.mat_x, self.mat_depth, self.mat_q, colors='k', linestyles='dashed',
            #             levels=level0, linewidths=1, zorder=2)
            # Color contour plot.
            cp = plt.contourf(self.mat_x[:,self.mat_x[0]>self.x_left], self.mat_depth[:,self.mat_x[0]>self.x_left], self.mat_q[:,self.mat_x[0]>self.x_left], levels=levels,
                              locator=ticker.LogLocator())
            # plt.plot(self.x, self.B, label='Bedrock', color='0')
            cb = plt.colorbar(cp)
            cb.set_label('Total ice flux')
            # cb.set_ticks(np.array([np.min(self.plmat_q[:,self.mat_x[0]>self.x_left]), np.max(self.mat_q[:,self.mat_x[0]>self.x_left])]))
            cb.set_ticklabels([])
            if self.inversion:

                plt.plot(self.x[self.x>self.x_left], np.zeros_like(self.S)[self.x>self.x_left], label='Surface', color='0')
                #plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], color='darkblue', zorder=1)    #refrozen ice
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left], label='Inverted bedrock', color='darkviolet', zorder=2)    #inverted Bedrock
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], label='Observed bedrock', color='0', linewidth=2)    # observed Bedrock
                # Fake plots for the legend
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], label="Trajectories", color=color, linewidth=lw)
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], label="Corner trajectory", color='k', linewidth=1,
                         linestyle='dashed')

                #plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    #where=self.H_try[self.x_inv>self.x_left]>self.H_ref[self.x_inv>self.x_left], color='#8B8BC6', label='refrozen ice',interpolate=True,zorder=-1, edgecolor=None)
                plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    where=self.H_try[self.x_inv>self.x_left]<self.H_inv[self.x_inv>self.x_left], color='0.7',label='stagnant ice',interpolate=True)
                plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    where=self.H_try[self.x_inv>self.x_left]>self.H_inv[self.x_inv>self.x_left], color='white', interpolate=True, edgecolor=None)  # bedrock
            else:
                plt.plot(self.x[self.x>self.x_left], self.H[self.x>self.x_left], label='Bedrock', color='0')
                plt.plot(self.x, self.S, label='Surface', color='0')
                # Fake plots for the legend
                plt.plot(self.x, -self.B, label="Trajectories", color=color, linewidth=lw)
                plt.plot(self.x, -self.B, label="Corner trajectory", color='k', linewidth=1,
                         linestyle='dashed')

            for name in self.ic:
                plt.plot(self.ic[name]['XX'], self.ic[name]['DD'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['x'], -50), ha='center', va='bottom',
                             color=color_core)
            x1,x2,y1,y2 = plt.axis()

            plt.gca().invert_yaxis()

            # plt.legend(loc='lower left')
            plt.legend(bbox_to_anchor=(1.4, 0.5))
            plt.xlabel(r'Distance (km)')
            plt.ylabel('Depth (m)')
            ax.grid(color='gray', alpha=0.3)
            plt.savefig(self.label+'stream_lines.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')


            # ---------------------------------------------------------------------
            # R(t) - Age
            # ---------------------------------------------------------------------

            fig, ax = plt.subplots()
            plt.stairs(self.R[:-2], self.age_R[:-1]/1000, baseline=None)
            plt.xlabel('time (kyr)', fontsize=15)
            plt.ylabel(r'$R(t)$', fontsize=15)
            plt.savefig(self.label+'R_temporal_factor.'+self.fig_format,
                        format=self.fig_format, bbox_inches='tight')

            # ---------------------------------------------------------------------
            # inverted parameters
            # ---------------------------------------------------------------------
            if self.inversion:
                fig, ax = plt.subplots(4, figsize=(7,10))
                ax[0].plot(self.x_inv, self.a_try, color='C0')
                ax[1].plot(self.x_inv, self.p_try, color='C1')
                ax[2].plot(self.x_inv, self.Delta_try, color='C2')
                ax[3].plot(self.x_inv, self.m_try, color='C3')
                ax3b = ax[3].twinx()
                ax3b.plot(self.x_inv, self.stagnant, color='grey')
                
                ax[0].fill_between(self.x_inv, self.a_try+self.sigma_a, self.a_try-self.sigma_a, color='C0', alpha=0.3, edgecolor=None)
                ax[1].fill_between(self.x_inv, self.p_try+self.sigma_p, self.p_try-self.sigma_p, color='C1', alpha=0.3, edgecolor=None)
                ax[2].fill_between(self.x_inv, self.Delta_try+self.sigma_Delta, self.Delta_try-self.sigma_Delta, color='C2', alpha=0.3, edgecolor=None)
                ax[3].fill_between(self.x_inv, self.m_try+self.sigma_m, self.m_try-self.sigma_m, color='C3', alpha=0.3, edgecolor=None)
                #ax3b.fill_between(self.x_inv, self.stagnant, 0, color='grey', alpha=0.3, edgecolor=None)
                # ax[2].set_ylim(np.min(self.H_try)-300, np.max(self.H_try)+300)
                # ax[2].invert_yaxis()
                ax[0].set_ylabel('a (m/yr)', color='C0')
                ax[1].set_ylabel('p', color='C1')
                ax[2].set_ylabel('Delta', color='C2')
                ax[3].set_ylabel('m (m/yr)', color='C3')
                ax[3].set_xlabel('Distance (km)')
                ax3b.set_ylabel('stagnant ice thickness (m)', color='grey')

                plt.setp(ax, xlim=(np.min(self.x), np.max(self.x)))
                plt.savefig(self.label+'inverted_params.'+self.fig_format,
                            format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # Graphs vs depth for the ice core
            # ----------------------------------------------------------
            for name in self.ic:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_ylabel('depth (m)')
                ax.invert_yaxis()
                ax.plot(self.ic[name]['x0'], self.ic[name]['depth'], color='r')
                if self.ic[name]['comp'] is not None and ~np.isnan(self.ic[name]['cp_x']).all():
                    ax.plot(self.ic[name]['cp_x'], self.ic[name]['cp_depth'], color='r',
                            linestyle='dashed')
                ax.set_xlabel(r'$x$ origin (km)', color='r')
                ax.spines['bottom'].set_color('r')
                ax.tick_params(axis='x', colors='r')

                ax2 = ax.twiny()
                ax2.spines.bottom.set_visible(False)
                ax2.plot(self.ic[name]['age']/1000, self.ic[name]['depth'], color='b')
                if self.ic[name]['comp'] is not None and \
                        ~np.isnan(self.ic[name]['cp_age']).all():
                    ax2.plot(self.ic[name]['cp_age'], self.ic[name]['cp_depth'], color='b',
                             linestyle='dashed')
                ax2.set_xlabel('age (kyr)', color='b')
                ax2.spines['top'].set_color('b')
                ax2.tick_params(axis='x', colors='b')
                
                iso_depth_at_ic = np.array([np.interp(self.ic[name]['x'], self.x_obs, self.iso_obs_depth[i])
                                            for i in range(self.cp_iso_nb)])
                iso_age_at_ic   = self.iso_obs_age[:, 0] / 1000        
                iso_sigma_at_ic = self.iso_obs_age_sigma[:, 0] / 1000  

                ax2.errorbar(iso_age_at_ic, iso_depth_at_ic, xerr=iso_sigma_at_ic, fmt=".",label='depth isos', color = 'black')

                ax3 = ax.twiny()
                ax3.spines['top'].set_position(('axes', 1.1))
                ax3.spines.bottom.set_visible(False)
                ax3.plot(self.ic[name]['tau'], self.ic[name]['depth'], color='g')
                if self.ic[name]['comp'] is not None and \
                        ~np.isnan(self.ic[name]['cp_tau']).all():
                    ax3.plot(self.ic[name]['cp_tau'], self.ic[name]['cp_depth'], color='g',
                             linestyle='dashed')
                ax3.set_xlabel('thinning function (no unit)', color='g')
                ax3.spines['top'].set_color('g')
                ax3.tick_params(axis='x', colors='g')

                plt.savefig(self.label+name+'_ice_core_vs_depth.'+self.fig_format,
                            format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------
            # Graphs vs age for the ice core
            # ----------------------------------------------

            for name in self.ic:
                fig, ax = plt.subplots(figsize=(15, 7))
                ax.set_xlabel('age (kyr)')
                ax.set_ylabel('layer thickness (m/yr)')

                ax.stairs(self.ic[name]['a0'][:-1], self.ic[name]['age']/1000, baseline=None,
                          label='accumulation')
                ax.stairs(self.ic[name]['tau'][:-1] * self.ic[name]['a0'][:-1], self.ic[name]['age']/1000,
                          baseline=None, label='layer thickness')
                ax.legend()

                plt.savefig(self.label+name+'_ice_core_vs_age.'+self.fig_format,
                            format=self.fig_format, bbox_inches='tight')

            # ----------------------------------------------------------
            # AgeMisfit
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 6))

            plt.plot(self.x[self.x>self.x_left], np.zeros_like(self.S)[self.x>self.x_left], label='Surface', color='0')

            if self.inversion:
                #plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], color='darkblue', zorder=1)    #refrozen ice
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left], label='Inverted Bedrock', color='darkviolet', zorder=3)    #inverted Bedrock
                plt.plot(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], label='Observed Bedrock', color='0', zorder=3, linewidth=2)    # observed Bedrock

                #plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_ref[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    #where=self.H_try[self.x_inv>self.x_left]>self.H_ref[self.x_inv>self.x_left], color='#8B8BC6', label='refrozen ice',interpolate=True,zorder=-1)
                plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], self.H_try[self.x_inv>self.x_left],
                    where=self.H_try[self.x_inv>self.x_left]<self.H_inv[self.x_inv>self.x_left], color='0.7',label='stagnant ice',interpolate=True)
                plt.fill_between(self.x_inv[self.x_inv>self.x_left], self.H_inv[self.x_inv>self.x_left], np.max(self.H_try),
                    where=self.H_try[self.x_inv>self.x_left]>self.H_inv[self.x_inv>self.x_left], color='white', interpolate=True, edgecolor=None)
            else:
                plt.plot(self.x[self.x>self.x_left], self.H[self.x>self.x_left], label='Bedrock', color='0')
            # plt.plot(self.x, resi.T)
            resi = self.iso_mod_age-self.iso_obs_age
            for i in range(len(resi)):
                plt.scatter(self.x_obs[self.x_obs>self.x_left], self.iso_obs_depth[i][self.x_obs>self.x_left], c=resi[i][self.x_obs>self.x_left], s=3,
                    edgecolor=None, norm=Normalize(vmin=-self.mis_lim, vmax=self.mis_lim),cmap='coolwarm')
            cb = plt.colorbar(label='age misfit (kyr)')
            ticks = np.arange(-self.mis_lim, self.mis_lim+1000, 1000)
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticks/1000)

            # x1,x2,y1,y2 = plt.axis()
            # plt.axis((x1,x2,0,y2))

            for name in self.ic:
                plt.plot(self.ic[name]['XX'], self.ic[name]['DD'], linewidth=lw_core,
                         color=color_core, linestyle=ls_core)
                plt.annotate(name, (self.ic[name]['x'], -50), ha='center', va='bottom',
                             color=color_core)

            ax.grid(color='gray', alpha=0.3)

            plt.gca().invert_yaxis()
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (m)')
            plt.savefig(self.label+'age_misfit.pdf')
            plt.close()

            # ----------------------------------------------------------
            # x axis
            # ----------------------------------------------------------

            plt.figure()

            obs_dist = np.mean(self.x_obs[1:] - self.x_obs[:-1])
            inv_dist = np.mean(self.x_inv[1:] - self.x_inv[:-1])

            plt.plot(self.x[:-1], self.for_dist, label='forward')
            plt.axhline(obs_dist, label='nobs', color='C1')
            plt.axhline(inv_dist, label='ninv', color='C2')

            plt.xlabel('Distance (km)')
            plt.ylabel('Distance between nodes (km)')

            plt.legend()
            plt.savefig(self.label+'meshs.pdf')

    # overall run function
    def run_model(self):

        # Registration of start time
        self.START_TIME = time.perf_counter()
        # load and interpolate data
        self.default_params()
        self.load_parameters()
        self.load_data()
        self.load_obs_data()
        self.initial_setup()
        
        if self.start == 'restart':
            print('Restart mode: loading optimised parameters from *_opt.txt')
            # load and re-interpolate onto current x_inv 
            x_opt, a_opt     = np.loadtxt(self.label + 'accumulation_opt.txt',  unpack=True)
            x_opt, p_opt     = np.loadtxt(self.label + 'p_Lliboutry_opt.txt',   unpack=True)
            x_opt, Delta_opt = np.loadtxt(self.label + 'Delta_opt.txt',         unpack=True)
    
            self.a_inv       = np.interp(self.x_inv, x_opt, a_opt)
            self.p_inv       = np.interp(self.x_inv, x_opt, p_opt)
            self.p_prime_inv = np.log(self.p_inv + 1)          
            self.Delta_inv   = np.interp(self.x_inv, x_opt, Delta_opt)
            print(f'  a_opt      : min={self.a_inv.min():.4f}  max={self.a_inv.max():.4f}')
            print(f'  p_opt      : min={self.p_inv.min():.4f}  max={self.p_inv.max():.4f}')
            print(f'  Delta_opt  : min={self.Delta_inv.min():.4f}  max={self.Delta_inv.max():.4f}')
        else:
            print('Import mode: starting from imported parameter files')
            
        # run interpolation once
        self.a_try = self.a_inv
        self.m_try = self.m_inv
        self.p_try = self.p_prime_inv
        self.H_try = self.H_inv
        self.Delta_try = self.Delta_inv
        self.interpolation()
        # run inversion
        if self.inversion == False:
            #self.variables = np.array([np.log(self.a_inv), self.p_prime_inv, np.log(self.H_inv)]).flatten()
            self.variables = np.array([np.log(self.a_inv), self.p_prime_inv, self.Delta_inv]).flatten()
            
            self.residuals(self.variables)
        else:
            self.optimise()
            print('minimise complete')

        self.save_data()
        # post processing
        self.ice_core()
        self.plot_figs()

        # Program execution time
        MESSAGE = 'Program execution time: '+str(time.perf_counter()-self.START_TIME)+' s.'
        print(MESSAGE)


        if os.name != 'nt':
           mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
           print('Max memory usage: '+str(mem)+' kbytes')

# Setting experiment directory
datadir = sys.argv[1]
if datadir[-1] != '/':
    datadir = datadir+'/'
print('Parameters directory is: ', datadir)
flow = FlowLine(datadir)
flow.run_model()

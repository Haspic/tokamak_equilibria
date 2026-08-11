

import os
import pickle

import numpy as np

import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from freegsnke import build_machine, equilibrium_update, GSstaticsolver, nonlinear_solve
from freegsnke.jtor_update import ConstrainPaxisIp
from freegsnke.inverse import Inverse_optimizer


class Tokamak:
    """
    Centralised object to more easily handle the creation of freegnske's Tokamak object
    (addition of coils, initialization from files, etc..)
    """

    def __init__(self, dimension: tuple):
        """
        Some parameter initialization

        :param dimension: Plotting dimensions
        """

        self.eta_copper = 1.55e-8  # resistivity in Ohm*m
        self.resistivity_wall = 5.5e-7

        self.width, self.height = dimension

        self.active_coils = {}
        self.passive_coils = []
        self.limiter = []
        self.wall = []

    def add_active_coil(self,
                        coil_name: str,
                        coil_part: str,
                        center: tuple,
                        dR: float,
                        dZ: float,
                        individual_windings_size: tuple = (0, 0),
                        geometry: tuple = (1,1),
                        polarity: int = 1,
                        multiplier: int = 1):
        """
        Registers a new coil and stores its parameters for later initialization.

        :param coil_name: Coil name
        :param coil_part: Coil subname (for multiple coils linked to same power supply)
        :param center: Center place of given coil (R, Z)
        :param dR: R-axis coil radius
        :param dZ: Z-axis coil radius
        :param individual_windings_size: (R, Z)-axis size/length (value 0 will automatically scale said axis to fill the gaps)
        :param geometry: Number of windings in each (R, Z)-axis | Ex: (4,4) for a 4x4 = 16 windings coil.
        :param polarity: "Circuit wiring, enabling coil pairs to be linked in series or anti-series"
        :param multiplier: "Current multiplier, used for splitting current among coils"
        :return: None

        ( See: https://docs.freegsnke.com/notebooks/example00%20-%20build_tokamak_machine.html )
        """

        r, z = center
        nr, nz = geometry

        radius_r, radius_z = individual_windings_size
        # If given axis windings size is 0, set each windings length (in said axis) to fit and fill the gaps
        if individual_windings_size[0] == 0:
            radius_r = dR / geometry[0]
        if individual_windings_size[1] == 0:
            radius_z = dZ / geometry[1]

        dR_mes = dR / nr
        dZ_mes = dZ / nz

        # Create coordinates for the center points of all individual windings for given geometry
        if nr % 2 == 1:
            r_coords = np.linspace(r + dR_mes/2*(1 - nr), r - dR_mes/2*(1 - nr), nr)
        else:
            r_coords = np.linspace(r - dR_mes/2*(nr - 1), r + dR_mes/2*(nr - 1), nr)

        if nz % 2 == 1:
            z_coords = np.linspace(z + dZ_mes/2*(1 - nz), z - dZ_mes/2*(1 - nz), nz)
        else:
            z_coords = np.linspace(z - dZ_mes/2*(nr - 1), z + dZ_mes/2*(nr - 1), nz)

        r_coords, z_coords = np.meshgrid(r_coords, z_coords)

        # Create parameters dictionaries to later on pass over to freegnske's tokamak object during initialization
        if coil_name not in list(self.active_coils.keys()):
            self.active_coils[coil_name] = {}
        self.active_coils[coil_name][coil_part] = {
            "R": np.concatenate(r_coords),
            "Z": np.concatenate(z_coords),
            "dR": radius_r,
            "dZ": radius_z,
            "resistivity": self.eta_copper,
            "polarity": polarity,
            "multiplier": multiplier,
        }

    def add_passive_coil(self, start_coord: tuple, end_coord: tuple, thickness: float):
        """
        Registers a new passive coil/structure and stores its parameters for later initialization.

        Functions by giving it a start point, an end point and a structure thickness. Will automatically
        create the contour points to be passed along to freegsnke's tokamak object.

        :param start_coord: Starting coordinate (R, Z)
        :param end_coord: Ending coordinate (R, Z)
        :param thickness: Thickness of passive coil/structure
        :return: None

        ( See: https://docs.freegsnke.com/notebooks/example00%20-%20build_tokamak_machine.html )
        """

        s_r, s_z = start_coord
        e_r, e_z = end_coord
        t = thickness

        if (s_r < e_r) and (s_z < e_z):
            R = [s_r-t, e_r-t, e_r+t, e_r+t, s_r+t, s_r-t]
            Z = [s_z+t, e_z+t, e_z+t, e_z-t, s_z-t, s_z-t]
        elif (s_r < e_r) and (s_z > e_z):
            R = [s_r-t, s_r+t, e_r+t, e_r+t, e_r-t, s_r-t]
            Z = [s_z+t, s_z+t, e_z+t, e_z-t, e_z-t, s_z-t]
        elif (s_r > e_r) and (s_z > e_z):
            R = [s_r-t, s_r+t, s_r+t, e_r+t, e_r-t, e_r-t]
            Z = [s_z+t, s_z+t, s_z-t, e_z-t, e_z-t, e_z+t]
        else: # (s_r > e_r) and (s_z < e_z):
            R = [s_r+t, s_r+t, s_r-t, e_r-t, e_r-t, e_r+t]
            Z = [s_z+t, s_z-t, s_z-t, e_z-t, e_z+t, e_z+t]

        self.passive_coils.append({
            "R": R,
            "Z": Z,
            "resistivity": self.resistivity_wall
        })

    def add_limiter(self, r_coords: list, z_coords: list):
        """
        Registers a limiter and stores its parameters for later initialization.

        :param r_coords: List of R-axis coordinates
        :param z_coords: List of corresponding Z-axis coordinates
        :return: None

        ( See: https://docs.freegsnke.com/notebooks/example00%20-%20build_tokamak_machine.html )
        """

        for r, z in zip(r_coords, z_coords):
            self.limiter.append({"R": r, "Z": z})

    def add_wall(self, r_coords, z_coords):
        """
        Registers a wall and stores its parameters for later initialization.

        :param r_coords: List of R-axis coordinates
        :param z_coords: List of corresponding Z-axis coordinates
        :return: None

        ( See: https://docs.freegsnke.com/notebooks/example00%20-%20build_tokamak_machine.html )
        """

        for r, z in zip(r_coords, z_coords):
            self.wall.append({"R": r, "Z": z})

    def save_tokamak_as_files(self, folder_path: str = "./tokamak/"):
        """
        Save created tokamak object as files for sharing / importing with freegsnke

        :param folder_path: Folder path '.../folder_name/'
        :return: None
        """

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(folder_path + "active_coils.pickle", "wb") as f:
            pickle.dump(self.active_coils, f)

        with open(folder_path + "passive_coils.pickle", "wb") as f:
            pickle.dump(self.passive_coils, f)

        with open(folder_path + "limiter.pickle", "wb") as f:
            pickle.dump(self.limiter, f)

        with open(folder_path + "wall.pickle", "wb") as f:
            pickle.dump(self.wall, f)

    def initialize_object_from_files(self, folder_path: str = "./tokamak/"):
        """
        Initialize tokamak object from files

        :param folder_path: Forder path '.../folder_name/'
        :return: None
        """

        self.tokamak = build_machine.tokamak(
            active_coils_path=folder_path + "active_coils.pickle",
            passive_coils_path=folder_path + "passive_coils.pickle",
            limiter_path=folder_path + "limiter.pickle",
            wall_path=folder_path + "wall.pickle",
        )

    def initialize_object(self):
        """
        Create/initialize freegnske's tokamak object from defined/imported tokamak parameters

        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :return: None
        """

        self.tokamak = build_machine.tokamak(
            active_coils_data=self.active_coils,
            passive_coils_data=self.passive_coils,
            limiter_data=self.limiter,
            wall_data=self.wall,
        )

    def instantiate_equilibrium(self,
                                Rmin: float,
                                Rmax: float,
                                Zmin: float,
                                Zmax: float,
                                nx: int,
                                ny: int):
        """
        Initialize freegnske's equilibrium object

        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :param Rmin: "Minimum major radius [m]."
        :param Rmax: "Maximum major radius [m]."
        :param Zmin: "Minimum height [m]."
        :param Zmax: "Maximum height [m]."
        :param nx: "Number of radial grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...)."
        :param ny: "Number of vertical grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...)."
        :return: None
        """

        self.eq = equilibrium_update.Equilibrium(
            tokamak=self.tokamak,
            Rmin=Rmin, Rmax=Rmax,
            Zmin=Zmin, Zmax=Zmax,
            nx=nx,  # (2**n + 1)
            ny=ny  # (2**n + 1)
        )

    def set_coil_current(self, coil_name, current, control: bool=False):
        """
        Apply a current to a given coil. Can set this current to be fixed or not during solve.

        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :param coil_name: Coil name
        :param current: Current amount [A]
        :param control: Whever or not the coil should be allowed to change current (False = wont change)
        :return: None
        """

        self.eq.tokamak.set_coil_current(coil_name, current)
        self.eq.tokamak[coil_name].control = control

    def instantiate_PaxisIp_profile(self,
                                    paxis: float,
                                    Ip: float,
                                    fvac: float,
                                    alpha_m: float,
                                    alpha_n: float):
        """
        Instantiate PaxisIp profile object.

        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :param paxis: "Pressure on the magnetic axis [Pa]."
        :param Ip: "Total plasma current [A]." (within last closed flux surface)
        :param fvac: "Vacuum toroidal field strength (f = R*B_tor) [T]."
        :param alpha_m: "Shape/peakedness parameter (non-negative)."
        :param alpha_n: "Shape/peakedness parameter (non-negative)."
        :return: None
        """

        self.profiles = ConstrainPaxisIp(
            eq=self.eq,  # equilibrium object
            paxis=paxis,  # pressure on the magnetic axis
            Ip=Ip,  # plasma current
            fvac=fvac,  # fvac = R B_{tor}
            alpha_m=alpha_m,  # profile function parameter
            alpha_n=alpha_n  # profile function parameter
        )

    def load_static_nonlinear_solve(self):
        """
        Initialize non-linear solver.

        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :return:
        """
        self.GSStaticSolver = GSstaticsolver.NKGSsolver(
            eq=self.eq,
        )

    def set_constraints(self,
                        null_points: None or tuple or np.array = None,
                        isoflux_set: None or tuple or np.array = None):
        """
        Initialize profile constraints.

        :param null_points: Coordinates of null points constraints
        :param isoflux_set: Coordinates of isoflux points (same flux) constraints
        :return: None
        """

        self.constraints = Inverse_optimizer(
            null_points=null_points,
            isoflux_set=isoflux_set
        )

    def forward_solve(self,
                      target_relative_tolerance: float,
                      verbose: bool):
        """
        Do a static forward solve (sets constraints = None).

        ( See : https://docs.freegsnke.com/notebooks/example02%20-%20static_forward_solve_mastu )

        :param target_relative_tolerance: "maximum relative error on the plasma flux function allowed for convergence"
        :param verbose: Whether freegsnke's solve process should be printed or not.
        :return: None
        """

        self.GSStaticSolver.solve(eq=self.eq,
                                  profiles=self.profiles,
                                  constrain=None, #self.constraints,
                                  target_relative_tolerance=target_relative_tolerance,
                                  verbose=verbose
                                  )

    def inverse_solve(self,
                      target_relative_tolerance,
                      target_relative_psit_update,
                      verbose,
                      l2_reg):
        """
        Do a static inverse solve.

        ( See : https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :param target_relative_tolerance: "maximum relative error on the plasma flux function allowed for convergence"
        :param target_relative_psit_update: "ensures that the relative update to the plasma flux is lower than this target value"
        :param verbose: Whether freegsnke's solve process should be printed or not.
        :param l2_reg: l2_reg: "defines the Tikonov regularisation used by the optmiser"
        :return: None
        """

        self.GSStaticSolver.solve(
                             eq=self.eq,
                             profiles=self.profiles,
                             constrain=self.constraints,
                             target_relative_tolerance=target_relative_tolerance,
                             target_relative_psit_update=target_relative_psit_update,
                             verbose=verbose,
                             l2_reg=l2_reg,
                             )

    def plot_machine(self):
        """ Plot the tokamak machine's structure """

        fig1, ax1 = plt.subplots(1, 1, figsize=(4, 5), dpi=80)
        plt.tight_layout()

        self.tokamak.plot(axis=ax1, show=False)
        ax1.plot(self.tokamak.limiter.R, self.tokamak.limiter.Z, color='k', linewidth=1.2, linestyle="--")
        ax1.plot(self.tokamak.wall.R, self.tokamak.wall.Z, color='k', linewidth=1.2, linestyle="-")

        ax1.grid(alpha=0.5)
        ax1.set_aspect('equal')
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(-self.height/2, self.height/2)
        ax1.set_xlabel(r'Major radius, $R$ [m]')
        ax1.set_ylabel(r'Height, $Z$ [m]')

        plt.show()

    def plot_solver(self):
        """ Plot the tokamak's structure and the solved profile """
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8), dpi=80)

        ax1.grid(zorder=0, alpha=0.75)
        ax1.set_aspect('equal')
        self.eq.tokamak.plot(axis=ax1, show=False)  # plots the active coils and passive structures
        ax1.fill(self.tokamak.wall.R, self.tokamak.wall.Z, color='k', linewidth=1.2, facecolor='w', zorder=0)  # plots the limiter
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(-self.height/2, self.height/2)

        ax2.grid(zorder=0, alpha=0.75)
        ax2.set_aspect('equal')
        self.eq.tokamak.plot(axis=ax2, show=False)  # plots the active coils and passive structures
        ax2.fill(self.tokamak.wall.R, self.tokamak.wall.Z, color='k', linewidth=1.2, facecolor='w', zorder=0)  # plots the limiter
        self.eq.plot(axis=ax2, show=False)  # plots the equilibrium
        ax2.set_xlim(0, self.width)
        ax2.set_ylim(-self.height/2, self.height/2)

        """
        Part of plot that shows constraints, seems not to work with newer version of freegnske (0.13.1), 
        works with version 0.12.0
        """

        # ax3.grid(zorder=0, alpha=0.75)
        # ax3.set_aspect('equal')
        # self.eq.tokamak.plot(axis=ax3, show=False)  # plots the active coils and passive structures
        # ax3.fill(self.tokamak.wall.R, self.tokamak.wall.Z, color='k', linewidth=1.2, facecolor='w', zorder=0)  # plots the limiter
        # self.eq.plot(axis=ax3, show=False)  # plots the equilibrium
        # ax3.set_xlim(0, self.width)
        # ax3.set_ylim(-self.height/2, self.height/2)
        # self.constraints.plot(axis=ax3, show=False)  # plots the contraints

        plt.tight_layout()
        plt.show()

    """ ----- ----- PROFILE STABILITY ----- ----- """


    # ----- unfinished ----- #


    def nonlinear_solve(self, plasma_resistivity, min_dIy_dI, threshold_dIy_dI, max_mode_frequency, **kwargs):

        self.nonlinear_solver = nonlinear_solve.nl_solver(
            eq=self.eq,
            profiles=self.profiles,
            GSStaticSolver=self.GSStaticSolver,
            plasma_resistivity=plasma_resistivity,
            min_dIy_dI=min_dIy_dI, # Minimum coupling threshold for excluding vessel modes (must be a number in [0,1]).
            threshold_dIy_dI=threshold_dIy_dI, # Relative coupling threshold for including vessel modes (must be a number in [0,1]).
            max_mode_frequency=max_mode_frequency, # Threshold frequency for retaining vessel modes.
            **kwargs
        )

    def plot_unstability_poloidal_flux(self):
        """ Works only if there exist unstabilities ? """

        # accessing the growth rates (via the timescales)
        timescales = self.nonlinear_solver.linearised_sol.all_timescales  # all eigenvalues: timescales
        growth_rates = 1 / timescales  # growth rates are simply 1/timescales
        modes = self.nonlinear_solver.linearised_sol.all_modes  # all eigenvectors (columns in same order as e'values)
        # mode number (choose which one you want to visualize)

        # extracting the unstable mode
        mask = (timescales > 0)
        print(mask)
        print(np.where(mask))

        idx = np.where(mask)[0][0]  # index of unstable mode
        unstable_timescales = timescales[mask]
        unstable_modes = np.squeeze(modes[:, mask])

        i = idx  # default is unstable mode
        mode_currents = np.real(modes[:, i])

        # the associated instability timescale and growth rate
        print(f"Mode {i} ---> {'stable' if np.real(timescales[i]) < 0 else 'unstable'}")
        print(f"Growth rate = {np.real(growth_rates[i]):.2e} [1/s]")
        print(f"Timescale = {np.real(timescales[i]):.2e} [s]")

        # multiply each metal current (from the eigenvector) with its corresponding Greens matrix and sum
        # (don't forget to omit the plasma current mode, i.e. the final element)
        flux = np.sum(mode_currents[0:-1, np.newaxis, np.newaxis] * self.nonlinear_solver.vessel_modes_greens, axis=0)

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 8), dpi=60)
        ax.grid(True, which='both')
        ax.set_aspect('equal')
        ax.set_xlim(0, self.width)
        ax.set_ylim(-self.height/2, self.height/2)
        ax.set_title(f"Poloidal flux (mode {i})")
        ax.set_xlabel(r'Major radius, $R$ [m]')
        ax.set_ylabel(r'Height, $Z$ [m]')
        plt.tight_layout()

        self.eq.tokamak.plot(axis=ax, show=False)
        ax.plot(self.eq.tokamak.wall.R, self.eq.tokamak.wall.Z, color='k', linewidth=1.2, linestyle="-")
        im = ax.contour(self.eq.R, self.eq.Z, flux, levels=50)
        cbar = plt.colorbar(im, ax=ax, fraction=0.09)

    def plot_jtor_maps_boundareis(self):

        # data
        jtor_maps = self.nonlinear_solver.deformable_vs_rigid_jtor

        # rate of change of R and Z current centre of plasma wrt to the unstable mode only
        dRZd_unstable_mode = self.nonlinear_solver.dRZd_unstable_mode
        print(f"Rate of change of Rcurrent wrt unstable mode = {dRZd_unstable_mode[0]:.2e} [m].")
        print(f"Rate of change of Zcurrent wrt unstable mode = {dRZd_unstable_mode[1]:.2e} [m].")

        diff = np.abs(jtor_maps[0] - jtor_maps[1])

        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=60)

        ax1.grid(True, which='both')
        ax1.set_aspect('equal')
        ax1.set_xlim(0.1, 2.15)
        ax1.set_ylim(-2.25, 2.25)
        ax1.set_title(f"Deformable and rigid boundaries")
        ax1.set_xlabel(r'Major radius, $R$ [m]')
        ax1.set_ylabel(r'Height, $Z$ [m]')
        plt.tight_layout()

        self.eq.tokamak.plot(axis=ax1, show=False)
        ax1.plot(self.eq.tokamak.wall.R, self.eq.tokamak.wall.Z, color='k', linewidth=1.2, linestyle="-")
        ax1.contour(self.eq.R, self.eq.Z, jtor_maps[0], levels=[0], colors='b')
        ax1.contour(self.eq.R, self.eq.Z, jtor_maps[1], levels=[0], colors='r')

        ax2.grid(True, which='both')
        ax2.set_aspect('equal')
        ax2.set_xlim(0.1, 2.15)
        ax2.set_ylim(-2.25, 2.25)
        ax2.set_title(f"Abs. diff. in Jtor maps")
        ax2.set_xlabel(r'Major radius, $R$ [m]')
        ax2.set_ylabel(r'Height, $Z$ [m]')

        self.eq.tokamak.plot(axis=ax2, show=False)
        ax2.plot(self.eq.tokamak.wall.R, self.eq.tokamak.wall.Z, color='k', linewidth=1.2, linestyle="-")
        im2 = ax2.contourf(self.eq.R, self.eq.Z, diff, levels=np.linspace(0.01, np.max(diff), 20))
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.09)
        cbar.set_label('Current [A]')

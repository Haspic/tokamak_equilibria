
"""
Documentation:

https://www.datacamp.com/fr/tutorial/genetic-algorithm-python
https://www.mdpi.com/2078-2489/10/12/390
https://en.wikipedia.org/wiki/Mutation_(evolutionary_algorithm)
https://en.wikipedia.org/wiki/Genetic_algorithm
"""


import copy
import numpy as np
import matplotlib.pyplot as plt

from tokamak_generator import Tokamak


def get_normal_ranged_distribute(min_range: float,
                                 max_range: float,
                                 sd: float or None = None,
                                 mid: float or None = None):
    """
    Generate a random parameter using a normal distribution over a bounded interval

    :param min_range: Maximum range
    :param max_range: Minimum range
    :param sd: Standard deviation
    :param mid: Midpoint for the normal distribution. If none will take the midpoint between max_range and min_range.
    :return: Random float
    """

    if min_range == max_range:
        return min_range

    rng = np.random.default_rng()

    if sd is None:
        sd = (max_range - min_range) / 3
        mid = (max_range + min_range) / 2

    param = rng.normal(mid, sd)

    # While out of range, pick again
    while param < min_range or param > max_range:
        param = rng.normal(mid, sd)

    return param


class TokamakCoilOptimizer:
    """
    Genetic algorithm optimizer for coil placement in a tokamak

    Works using PaxisIp equilibrium profile ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )
    """

    def __init__(self, tokamak_dimensions: tuple, population_size: int):
        """
        Initialize variables

        :param tokamak_dimensions: Plotting dimensions
        :param population_size: Initial population size for genetic algorithm
        """

        # Population must be even (for crossover)
        assert population_size % 2 == 0, "Population size must be even"

        self.width, self.height = tokamak_dimensions

        # ----- #

        self.equilibrium_args = {}
        self.coil_currents_args = {}
        self.PaxisIp_profile_args = {}
        self.constraints = ()

        # ----- #

        # initialize population as array of Tokamak objects (see tokamak_generator.py)
        self.tokamak_herd = np.array([Tokamak(tokamak_dimensions) for _ in range(population_size)])

        # ----- #

        # List of each coil's constant parameters
        self.coil_constants = []
        # List of each coil's variable parameters' ranges
        self.coil_ranges = []
        # Same for coupled counterparts. Probably a way to merge this, but I can't think of
        # an easy fix for the moment, so I'll just go with this.
        self.coupled_coil_constants = []
        self.coupled_coil_ranges = []

    """ ----- # ----- # ----- """

    def add_active_coil(self,
                        coil_name: str,
                        coil_part: str,
                        center_range: tuple,
                        dR_range: tuple,
                        dZ_range: tuple,
                        # radius, locked to (0,0)
                        geometry: tuple = (1,1),
                        polarity: int = 1,
                        multiplier: int = 1,
                        ):
        """
        Registers a new coil and its parameters in object variables.

        :param coil_name: Coil name
        :param coil_part: Coil subname (for multiple coils linked to same power supply)

        :param center_range:
        Range of possible center placement for given coil,
        Always start from bottom left point to upper right point.
        Ex: ( (0,0) , (5,5) ) for a square of 5x5
        ( (R1, Z1), (R2, Z2) )

        :param dR_range: Range of (min, max) R-axis-radius
        :param dZ_range: Range of (min, max) Z-axis-radius
        :param geometry: Geometry of coil windings (R-windings, Z-windings) for RxZ number of windings
        :param polarity: "Circuit wiring, enabling coil pairs to be linked in series or anti-series"
        :param multiplier: "Current multiplier, used for splitting current among coils"
        :return: None

        ( See: https://docs.freegsnke.com/notebooks/example00%20-%20build_tokamak_machine.html )
        """

        self.coil_constants.append({
            "coil_name": coil_name,
            "coil_part": coil_part,
            "radius": (0,0),
            "geometry": geometry,
            "polarity": polarity,
            "multiplier": multiplier
        })

        self.coil_ranges.append([center_range, dR_range, dZ_range])

    def add_coupled_active_coil(self,
                                coils_name: str,
                                center_range_top: tuple,
                                dR_range: tuple,
                                dZ_range: tuple,
                                geometry: tuple = (1, 1),
                                polarity: int = 1,
                                multiplier: int = 1
                                ):
        """
        Registers a new couple of symmetrical coils (upper and lower).

        :param coils_name: Coils' name

        :param center_range_top:
        Range of possible placement for upper coil (symmetrical for lower coil),
        Always start from bottom left point to upper right point.
        Ex: ( (0,0) , (5,5) ) for a square of 5x5

        :param dR_range: Range of (min, max) R-axis-radius
        :param dZ_range: Range of (min, max) Z-axis-radius
        :param geometry: Geometry of coil windings (R-windings, Z-windings) for RxZ number of windings
        :param polarity: "Circuit wiring, enabling coil pairs to be linked in series or anti-series"
        :param multiplier: "Current multiplier, used for splitting current among coils"
        :return: None

        (See self.add_active_coil above)
        """

        self.coupled_coil_constants.append({
            "coil_name": coils_name,
            "radius": (0,0),
            "geometry": geometry,
            "polarity": polarity,
            "multiplier": multiplier
        })

        self.coupled_coil_ranges.append([center_range_top, dR_range, dZ_range])

    def add_passive_coil(self, *args, **kwargs):
        """ Passes call to individual tokamak objects as this parameter is not changed within algorithm """
        for tokamak in self.tokamak_herd:
            tokamak.add_passive_coil(*args, **kwargs)

    def add_limiter(self, *args, **kwargs):
        """ Passes call to individual tokamak objects as this parameter is not changed within algorithm """
        for tokamak in self.tokamak_herd:
            tokamak.add_limiter(*args, **kwargs)

    def add_wall(self, *args, **kwargs):
        """ Passes call to individual tokamak objects as this parameter is not changed within algorithm """
        for tokamak in self.tokamak_herd:
            tokamak.add_wall(*args, **kwargs)

    """ ----- # ----- # ----- """

    def _initialize_population(self):
        """
        Initializes tokamak population by adding active coils parameter to individual tokamaks, following
        a normal distribution over defined ranges.

        :return: None
        """

        # ----- SINGLE COILS ----- #
        for i, coil_parameters in enumerate(self.coil_constants):

            center_range, dR_range, dZ_range = self.coil_ranges[i]

            for tokamak in self.tokamak_herd:

                # For each tokamak generate a random coil placement
                center_x = get_normal_ranged_distribute(center_range[0][0], center_range[1][0])
                center_y = get_normal_ranged_distribute(center_range[0][1], center_range[1][1])
                center = (center_x, center_y)

                # Generate a random coil size
                dR = get_normal_ranged_distribute(dR_range[0], dR_range[1])
                dZ = get_normal_ranged_distribute(dZ_range[0], dZ_range[1])

                # Adds coil to tokamak object
                tokamak.add_active_coil(center=center, dR=dR, dZ=dZ, **coil_parameters)

        # ----- COUPLED COILS ----- #
        for i, params in enumerate(self.coupled_coil_constants):

            top_center_range, dR_range, dZ_range = self.coupled_coil_ranges[i]

            for tokamak in self.tokamak_herd:

                # Generate random coil placement
                center_x = get_normal_ranged_distribute(top_center_range[0][0], top_center_range[1][0])
                center_y = get_normal_ranged_distribute(top_center_range[0][1], top_center_range[1][1])

                top_center = (center_x, center_y)
                # Lower coil is symmetric to upper one
                bot_center = (center_x, -center_y)

                # Generate random coil size
                dR = get_normal_ranged_distribute(dR_range[0], dR_range[1])
                dZ = get_normal_ranged_distribute(dZ_range[0], dZ_range[1])

                # Adds upper and lower coil to tokamak object
                tokamak.add_active_coil(center=top_center, coil_part="upper", dR=dR, dZ=dZ, **params)
                tokamak.add_active_coil(center=bot_center, coil_part="lower", dR=dR, dZ=dZ, **params)

        # Build individual tokamaks as freegnske tokamak objects
        for tokamak in self.tokamak_herd:
            tokamak.initialize_object()

    def _instantiate_equilibrium(self, *args, **kwargs):
        """ Passes equilibrium initialization call """
        for tokamak in self.tokamak_herd:
            tokamak.instantiate_equilibrium(*args, **kwargs)

    def _set_coil_current(self, *args, **kwargs):
        """ Passes the call for the addition of a coil current constraint """
        for tokamak in self.tokamak_herd:
            tokamak.set_coil_current(*args, **kwargs)

    def _instantiate_PaxisIp_profile(self, *args, **kwargs):
        """ Passes PaxisIp profile constraint call """
        for tokamak in self.tokamak_herd:
            tokamak.instantiate_PaxisIp_profile(*args, **kwargs)

    def _load_static_nonlinear_solve(self):
        """ Passes solver initialization call """
        for tokamak in self.tokamak_herd:
            tokamak.load_static_nonlinear_solve()

    def _set_constraints(self, *args, **kwargs):
        """ Passes constraints addition call """
        for tokamak in self.tokamak_herd:
            tokamak.set_constraints(*args, **kwargs)

    """ ----- # ----- # ----- """

    def initialize(self,
                   equilibrium_args: dict,
                   coil_currents_args: dict,
                   PaxisIp_profile_args: dict,
                   constraints: tuple):
        """
        Initializes the optimizer and all tokamak objects in self.tokamak_herd

        :param equilibrium_args: kwargs for self.instantiate_equilibrium
        :param coil_currents_args: kwargs for self.coil_currents_args
        :param PaxisIp_profile_args: kwargs for instantiate_PaxisIp_profile
        :param constraints: (null_points, isoflux_set)
        :return: None

        ( See: https://docs.freegsnke.com/notebooks/example00%20-%20build_tokamak_machine.html )
        """

        assert (constraints[0] is not None) or (constraints[1] is not None), "One constraint must be given, algorithm cannot run on forward solves"

        self.equilibrium_args = equilibrium_args
        self.coil_currents_args = coil_currents_args
        self.PaxisIp_profile_args = PaxisIp_profile_args
        self.constraints = constraints

        self._initialize_population()
        self._instantiate_equilibrium(**equilibrium_args)
        for key in coil_currents_args.keys():
            self._set_coil_current(key, coil_currents_args[key])
        self._instantiate_PaxisIp_profile(**PaxisIp_profile_args)
        self._load_static_nonlinear_solve()
        self._set_constraints(*constraints)

    def _update_tokamaks(self):
        """
        Updates the initialization of all tokamak objects in self.tokamak_herd.
        To use jointly with self._iterate when a new selection of tokamak objects have been generated

        :return: None
        """

        for tokamak in self.tokamak_herd:
            tokamak.initialize_object()
        self._instantiate_equilibrium(**self.equilibrium_args)
        for key in self.coil_currents_args.keys():
            self._set_coil_current(key, self.coil_currents_args[key])
        self._instantiate_PaxisIp_profile(**self.PaxisIp_profile_args)
        self._load_static_nonlinear_solve()

    """ ----- # ----- # ----- """

    def view_population(self):
        """
        View a plot of all individual tokamaks structure together.

        :return: None
        """

        fig1, ax1 = plt.subplots(1, 1, figsize=(4, 5), dpi=80)
        plt.tight_layout()

        for tokamak in self.tokamak_herd:
            tk = tokamak.tokamak

            tk.plot(axis=ax1, show=False)
            ax1.plot(tk.limiter.R, tk.limiter.Z, color='k', linewidth=1.2, linestyle="--")
            ax1.plot(tk.wall.R, tk.wall.Z, color='k', linewidth=1.2, linestyle="-")

        ax1.grid(alpha=0.5)
        ax1.set_aspect('equal')
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(-self.height/2, self.height/2)
        ax1.set_xlabel(r'Major radius, $R$ [m]')
        ax1.set_ylabel(r'Height, $Z$ [m]')
        plt.show()

    def view_solved(self, index: int=0):
        """
        Plot a selected tokamak's geometry and its solution profile

        :param index: Index of tokamak in self.tokamak_herd
        :return: None
        """

        tokamak = self.tokamak_herd[index]
        tk = tokamak.tokamak

        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8), dpi=80)

        ax1.grid(zorder=0, alpha=0.75)
        ax1.set_aspect('equal')
        tokamak.eq.tokamak.plot(axis=ax1, show=False)  # plots the active coils and passive structures
        ax1.fill(tk.wall.R, tk.wall.Z, color='k', linewidth=1.2, facecolor='w',
                 zorder=0)  # plots the limiter
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(-self.height / 2, self.height / 2)

        ax2.grid(zorder=0, alpha=0.75)
        ax2.set_aspect('equal')
        tokamak.eq.tokamak.plot(axis=ax2, show=False)  # plots the active coils and passive structures
        ax2.fill(tk.wall.R, tk.wall.Z, color='k', linewidth=1.2, facecolor='w',
                 zorder=0)  # plots the limiter
        tokamak.eq.plot(axis=ax2, show=False)  # plots the equilibrium
        ax2.set_xlim(0, self.width)
        ax2.set_ylim(-self.height / 2, self.height / 2)

        """
        Part of plot that shows constraints, seems not to work with newer version of freegnske (0.13.1), 
        works with version 0.12.0
        """

        # ax3.grid(zorder=0, alpha=0.75)
        # ax3.set_aspect('equal')
        # tokamak.eq.tokamak.plot(axis=ax3, show=False)  # plots the active coils and passive structures
        # ax3.fill(tk.wall.R, tk.wall.Z, color='k', linewidth=1.2, facecolor='w',
        #          zorder=0)  # plots the limiter
        # tokamak.eq.plot(axis=ax3, show=False)  # plots the equilibrium
        # ax3.set_xlim(0, self.width)
        # ax3.set_ylim(-self.height / 2, self.height / 2)
        # tokamak.constraints.plot(axis=ax3, show=False)  # plots the constraints

        plt.tight_layout()
        plt.show()

    """ ----- # ----- # ----- """

    def _crossover(self,
                   crossover_rate: float,
                   tks: tuple):
        """
        Apply crossover process of genetic algorithm ( See: https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm) )

        :param crossover_rate: Crossover rate (float between 0 and 1)
        :param tks: tuple of parents tokamak which to apply crossover
        :return: tuple of child tokamaks
        """

        tk_1, tk_2 = tks
        rng = np.random.default_rng()

        # Randomly chooses to apply crossover or not
        if rng.random() <= crossover_rate:

            # chose one coil id to apply crossover to
            n_coils = len(self.coil_constants) + len(self.coupled_coil_constants)
            coil_choice = rng.integers(0, n_coils)

            # ----- SINGLE COILS ----- #
            for i, coil_parameters in enumerate(self.coil_constants):

                # only apply crossover to chosen coil
                if i == coil_choice:

                    # invert chosen coil in both tokamaks
                    coil_name, coil_part = coil_parameters["coil_name"], coil_parameters["coil_part"]
                    coil_1 = tk_1.active_coils[coil_name][coil_part]
                    coil_2 = tk_2.active_coils[coil_name][coil_part]

                    tk_1.active_coils[coil_name][coil_part] = coil_2
                    tk_2.active_coils[coil_name][coil_part] = coil_1

            # ----- COUPLED COILS ----- #
            for i, params in enumerate(self.coupled_coil_constants):

                # only apply crossover to chosen coil
                if i + len(self.coil_constants) == coil_choice:

                    # invert chosen coil in both tokamaks (both lower and upper couple)
                    coil_name = params["coil_name"]
                    coil_1_lower = tk_1.active_coils[coil_name]["lower"]
                    coil_1_upper = tk_1.active_coils[coil_name]["upper"]
                    coil_2_lower = tk_2.active_coils[coil_name]["lower"]
                    coil_2_upper = tk_2.active_coils[coil_name]["upper"]

                    tk_1.active_coils[coil_name]["lower"] = coil_2_lower
                    tk_1.active_coils[coil_name]["upper"] = coil_2_upper
                    tk_2.active_coils[coil_name]["lower"] = coil_1_lower
                    tk_2.active_coils[coil_name]["upper"] = coil_1_upper

        return tk_1, tk_2

    def _mutate(self,
                mutation_rate: float,
                tk: Tokamak,
                standard_deviation: float=0.01):
        """
        Apply mutation process of genetic algorithm ( See: https://en.wikipedia.org/wiki/Mutation_(evolutionary_algorithm) )

        :param mutation_rate: Mutation rate (float between 0 and 1)
        :param tk: parent tokamak which to apply parameter mutation
        :param standard_deviation: standard deviation of normal distribution for parameter mutation
        :return: child tokamak
        """

        rng = np.random.default_rng()

        if rng.random() <= mutation_rate:

            # chose one coil id to apply crossover to
            n_coils = len(self.coil_constants) + len(self.coupled_coil_constants)
            coil_choice = rng.integers(0, n_coils)

            # ----- SINGLE COILS ----- #
            for i, coil_parameters in enumerate(self.coil_constants):

                # only apply crossover to chosen coil
                if i == coil_choice:

                    # Retrieve coil parameter dictionary
                    '''
                    Looks like:
                    {
                        "R": np.concatenate(r_coords),
                        "Z": np.concatenate(z_coords),
                        "dR": radius_r,
                        "dZ": radius_z,
                        "resistivity": self.eta_copper,
                        "polarity": polarity,
                        "multiplier": multiplier,
                    }
                    (See tokamak_generator.py)
                    '''
                    coil_name, coil_part = coil_parameters["coil_name"], coil_parameters["coil_part"]
                    coil = tk.active_coils[coil_name][coil_part]

                    # Retrieve coil parameters initial values and ranges
                    center_range, dR_range, dZ_range = self.coil_ranges[i]
                    R_range, Z_range = np.array(center_range).T
                    center_R = (min(coil["R"]) + max(coil["R"])) / 2
                    center_Z = (min(coil["Z"]) + max(coil["Z"])) / 2
                    dR = coil["dR"]
                    dZ = coil["dZ"]

                    # Apply mutation using normal distribution
                    new_R = get_normal_ranged_distribute(*R_range, sd=standard_deviation, mid=center_R)
                    new_Z = get_normal_ranged_distribute(*Z_range, sd=standard_deviation, mid=center_Z)
                    new_dR = get_normal_ranged_distribute(*dR_range, sd=standard_deviation, mid=dR)
                    new_dZ = get_normal_ranged_distribute(*dZ_range, sd=standard_deviation, mid=dZ)

                    # since R and Z can be multiple values (for multiple windings coils),
                    # we determine the deviation from the center and apply to all
                    R_deviation = new_R - center_R
                    Z_deviation = new_Z - center_Z

                    coil["R"] += R_deviation
                    coil["Z"] += Z_deviation
                    coil["dR"] = new_dR
                    coil["dZ"] = new_dZ

                    tk.active_coils[coil_name][coil_part] = coil

            # ----- COUPLED COILS ----- #
            for i, params in enumerate(self.coupled_coil_constants):

                # only apply crossover to chosen coil
                if i + len(self.coil_constants) == coil_choice:

                    # Same process as above but with both upper and lower couple coils

                    coil_name = params["coil_name"]
                    coil_upper = tk.active_coils[coil_name]["upper"]
                    coil_lower = tk.active_coils[coil_name]["lower"]

                    top_center_range, dR_range, dZ_range = self.coupled_coil_ranges[i]
                    R_range, Z_range = np.array(top_center_range).T
                    top_center_R = (min(coil_upper["R"]) + max(coil_upper["R"])) / 2
                    top_center_Z = (min(coil_upper["Z"]) + max(coil_upper["Z"])) / 2
                    dR = coil_upper["dR"]
                    dZ = coil_upper["dZ"]

                    new_top_R = get_normal_ranged_distribute(*R_range, sd=standard_deviation, mid=top_center_R)
                    new_top_Z = get_normal_ranged_distribute(*Z_range, sd=standard_deviation, mid=top_center_Z)
                    new_dR = get_normal_ranged_distribute(*dR_range, sd=standard_deviation, mid=dR)
                    new_dZ = get_normal_ranged_distribute(*dZ_range, sd=standard_deviation, mid=dZ)

                    # since R and Z can be multiple values (for multiple windings coils),
                    # we determine the deviation from the center and apply to all
                    top_R_deviation = new_top_R - top_center_R
                    top_Z_deviation = new_top_Z - top_center_Z

                    coil_upper["R"] += top_R_deviation
                    coil_upper["Z"] += top_Z_deviation
                    coil_upper["dR"] = new_dR
                    coil_upper["dZ"] = new_dZ

                    # inverted Z for coupled lower coil
                    coil_lower["R"] += top_R_deviation
                    coil_lower["Z"] -= top_Z_deviation
                    coil_lower["dR"] = new_dR
                    coil_lower["dZ"] = new_dZ

                    tk.active_coils[coil_name]["upper"] = coil_upper
                    tk.active_coils[coil_name]["lower"] = coil_lower

        return tk

    """ ----- # ----- # ----- """

    @staticmethod
    def _evaluate(tokamak_list: list or np.array,
                  target_relative_tolerance: float,
                  target_relative_psit_update: float,
                  l2_reg: float):
        """
        Evaluating process (runs fitness function for all tokamaks and returns result)
        Fitness function is here inverse solve, which we use to determine the maximal current (of any coil)
        needed to create asked plasma profile. We try to minimize this max current.

        :param tokamak_list: List of tokamaks to evaluate

        :param target_relative_tolerance: "maximum relative error on the plasma flux function allowed for convergence"
        :param target_relative_psit_update: "ensures that the relative update to the plasma flux is lower than this target value"
        :param l2_reg: "defines the Tikonov regularisation used by the optmiser"
        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :return: Array of all computed max currents for each indiviual tokamaks
        """

        print("EVALUATING TOKAMAKS SELECTION")

        max_currents = np.array([None for _ in range(len(tokamak_list))])

        for i, tokamak in enumerate(tokamak_list):
            print("###################################################################")
            print(f"Tokamak {i + 1} ... ", end="")

            tokamak.inverse_solve(target_relative_tolerance, target_relative_psit_update, False, l2_reg)

            currents = np.array(list(tokamak.tokamak.getCurrents().values()))
            # Saves value of absolute max current
            max_currents[i] = np.max(np.abs(currents))

        return max_currents

    """ ----- # ----- # ----- """

    def _selection(self, tournament_size: int=4, **kwargs):
        """
        Runs a selection round to select a more fitted population than the previous
        For each spot in the population, runs a tournament with "tournament_size" number
        of competitor. Selects the most fitted and passes it along.

        :param tournament_size: amount of tokamak selected to partake in one tournament (tournament_size <= len(self.tokamak_herd))
        :param kwargs: all arguments to be passed along to self._evaluate function
        :return: returns list of best tokamaks (likely contains multiple duplicate) + best tokamak performance + max_currents
        """

        tk_size = len(self.tokamak_herd)
        max_currents = self._evaluate(self.tokamak_herd, **kwargs)
        selection = []
        best_performance_id = np.argmin(max_currents)
        best_tokamak = copy.deepcopy(self.tokamak_herd[best_performance_id])

        # repeat for the size of the population
        for _ in range(tk_size):
            # choose "tournament_size" number of tokamaks
            tokamak_list = np.random.choice(tk_size, size=tournament_size, replace=False)
            # select the winner among them (lowest max current)
            winner = np.argmin(max_currents[tokamak_list])
            # retrieve id of winner tokamak
            winner_id = tokamak_list[winner]
            # add winner tokamak into newly selected batch
            selection.append(copy.deepcopy(self.tokamak_herd[winner_id]))

        return np.array(selection), best_tokamak, max_currents

    """ ----- # ----- # ----- """

    def _iterate(self,
                 iteration_parameters: dict,
                 target_relative_tolerance: float,
                 target_relative_psit_update: float,
                 l2_reg: float):
        """
        Runs an algorithm iteration. i.e.:
        -> Select a new population by running selection tournament
        -> Apply crossover (depending on rate)
        -> Apply mutation (depending on rate)
        -> Pass along best tokamak to next generation
        -> Replace old population

        :param iteration_parameters: Dictionary of parameters on the following format:

        iteration_parameters = {
            "tournament_size": 10,
            "crossover_rate": 0.5,
            "mutation_rate": 0.75,
            "mutation_standard_deviation": 0.05
        }

        :param target_relative_tolerance: "maximum relative error on the plasma flux function allowed for convergence"
        :param target_relative_psit_update: "ensures that the relative update to the plasma flux is lower than this target value"
        :param l2_reg: "defines the Tikonov regularisation used by the optmiser"
        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :return: max currents list
        """

        # retrieve genetic algorithm arguments dictionary
        tournament_size = iteration_parameters["tournament_size"]
        crossover_rate = iteration_parameters["crossover_rate"]
        mutation_rate = iteration_parameters["mutation_rate"]
        mutation_standard_deviation = iteration_parameters["mutation_standard_deviation"]

        # Select best fitted tokamaks
        selection, best_tokamak, max_currents = self._selection(tournament_size=tournament_size,
                                                                target_relative_tolerance=target_relative_tolerance,
                                                                target_relative_psit_update=target_relative_psit_update,
                                                                l2_reg=l2_reg)

        next_pop = []

        # Apply crossover and mutation
        for i in range(0, len(selection), 2):

            parent_1 = selection[i]
            parent_2 = selection[i + 1]

            # Crossover
            child_1, child_2 = self._crossover(crossover_rate, (parent_1, parent_2))

            # Mutation
            child_1 = self._mutate(mutation_rate, child_1, mutation_standard_deviation)
            child_2 = self._mutate(mutation_rate, child_2, mutation_standard_deviation)

            next_pop.append(child_1)
            next_pop.append(child_2)

        # A thing that would have been nice to add is to pass along the x-top-percentage of best tokamaks instead of
        # only the best. This would allow for different structures to coexist and compete against each-other, while
        # we here increase the chances of falling into a local minima.
        next_pop[0] = best_tokamak

        # Replace old population
        del self.tokamak_herd
        self.tokamak_herd = np.array(next_pop)
        self._update_tokamaks()

        return max_currents

    """ ----- # ----- # ----- """

    def loop(self,
             iterations: int,
             iteration_parameters: dict,
             target_relative_tolerance: float,
             target_relative_psit_update: float,
             l2_reg: float):
        """
        Loop function of genetic algorithm. Call me to start optimization.

        :param iterations: Number of iterations to run
        :param iteration_parameters: Dictionary of iteration parameters on the following format:

        iteration_parameters = {
            "tournament_size": 10,
            "crossover_rate": 0.5,
            "mutation_rate": 0.75,
            "mutation_standard_deviation": 0.05
        }

        :param target_relative_tolerance: "maximum relative error on the plasma flux function allowed for convergence"
        :param target_relative_psit_update: "ensures that the relative update to the plasma flux is lower than this target value"
        :param l2_reg: "defines the Tikonov regularisation used by the optmiser"
        ( See: https://docs.freegsnke.com/notebooks/example01a%20-%20static_inverse_solve_mastu )

        :return:
        """

        print("STARTING LOOP")

        # Keep track of best and worse max currents along the way
        best_max_currents = []
        worst_max_currents = []

        for i in range(iterations):
            print(f"ITERATION {i+1}")

            # Iterate once
            max_currents = self._iterate(iteration_parameters, target_relative_tolerance, target_relative_psit_update, l2_reg)

            # Save results
            best = np.min(max_currents)
            worst = np.max(max_currents)

            best_max_currents.append(best)
            worst_max_currents.append(worst)

            print("\n\n\nRESULTS:")
            print("MIN:", best)
            print("MAX:", worst)


        ''' Optional extra informations '''


        # Best and worse result of final run
        print(best_max_currents)
        print(worst_max_currents)

        # View final population
        self.view_population()

        # Plot evolution of min/max max_current value
        plt.plot(best_max_currents, color="green")
        plt.plot(worst_max_currents, color="red")
        plt.xlabel("Iterations")
        plt.ylabel("Max current")
        plt.show()

        # max_currents = self._evaluate(self.tokamak_herd, target_relative_tolerance, target_relative_psit_update, l2_reg)
        #
        # indexes = np.argsort(max_currents)
        #
        # for i in range(5):
        #
        #     self.view_solved(indexes[i])
        #     self.tokamak_herd[indexes[i]].save_tokamak_as_files(f"checkpoint/tokamak_{max_currents[indexes[i]]}/")

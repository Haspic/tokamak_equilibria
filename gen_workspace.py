
import numpy as np
from genetic_algorithm import TokamakCoilOptimizer

"""
IN FUTURE:

- add a way to select only specific coils to allow to optimise for
"""

""" ----- TOKAMAK SETUP ----- """

optimizer = TokamakCoilOptimizer((1.5, 1.5), population_size=50)

R = 0.76
r = 0.251
h = 0.15
H = h + r
t = 0.1 # thickness coils

# Coil center ranges (messy)
P1_c = ( (R-1.5*r, H+t/2), (R-r+4*t/2, H+2*t) )
P2_c = ( (R+r-2*t, H+t/2) , (R+r+5*t/2, H+2*t))
P3_c = ( (R+r+t*1.5 - 3/4*t, h-t) , (R+r+t*1.5 + t, h+2*t) )
P_R = ( 0.1, 0.1 )

optimizer.add_active_coil(coil_name="solenoid", coil_part="", center_range=((0.210, 0), (0.210, 0)), dR_range=(0.030, 0.030), dZ_range=(0.80, 0.80), geometry=(1, 1))
optimizer.add_coupled_active_coil(coils_name="P1", center_range_top=P1_c, dR_range=P_R, dZ_range=P_R, geometry=(1, 1))
optimizer.add_coupled_active_coil(coils_name="P2", center_range_top=P2_c, dR_range=P_R, dZ_range=P_R, geometry=(1, 1))
optimizer.add_coupled_active_coil(coils_name="P3", center_range_top=P3_c, dR_range=P_R, dZ_range=P_R, geometry=(1, 1))

n = 15
r_limiter = np.concatenate([np.array([R-r]),
                            R + r * np.cos(np.linspace(np.pi, 0, n)),
                            R + r * np.cos(np.linspace(0, -np.pi, n)),
                            np.array([R-r])])
z_limiter = np.concatenate([np.array([0]),
                            h + r * np.sin(np.linspace(np.pi, 0, n)),
                            - h + r * np.sin(np.linspace(0, -np.pi, n)),
                            np.array([0])])
r_wall = r_limiter.copy()
z_wall = z_limiter.copy()

optimizer.add_limiter(r_limiter, z_limiter)
optimizer.add_wall(r_wall, z_wall)

""" ----- Optimizer parameters  ----- """

equilibrium_args = {
            "Rmin": 0.4, "Rmax": 1.1,
            "Zmin": -0.5, "Zmax": 0.5,
            "nx": 65,  # (2**n + 1) # 33 - 65 - 129 - 257
            "ny": 65  # (2**n + 1)
}

coil_currents_args = {"solenoid": 1_000}

PaxisIp_profile_args = {
            "paxis": 1_600,  # pressure on the magnetic axis
            "Ip": 100_000,  # plasma current
            "fvac": 1 * R,  # fvac = R B_{tor}
            "alpha_m": 1.8,  # profile function parameter
            "alpha_n": 1.2  # profile function parameter
}

""" ----- GOAL PROFILE ----- """

from miller_representation import miller

miller = miller(elongation=1.5,
                triangularity=0.4,
                R0=R,
                r=3/5*r)

R, Z = miller.get_coordinates(np.arange(0,2*np.pi, 2*np.pi/10))
isoflux_set = np.array([[R, Z]])

null_points = None
# null_points = miller.get_coordinates(np.array([np.pi/1.9, np.pi/1.9]))

constraints = (null_points, isoflux_set)

optimizer.initialize(equilibrium_args, coil_currents_args, PaxisIp_profile_args, constraints)

""" ----- WORKSPACE ----- """

optimizer.view_population()

iteration_parameters = {
    "tournament_size": 10,
    "crossover_rate": 0.5,
    "mutation_rate": 0.75,
    "mutation_standard_deviation": 0.05
}

optimizer.loop(
    iterations=20,
    iteration_parameters=iteration_parameters,
    target_relative_tolerance=1e-6,
    target_relative_psit_update=1e-3,
    l2_reg=1e-11
)

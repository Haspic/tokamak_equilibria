
import numpy as np
from tokamak_generator import Tokamak

tokamak = Tokamak((1.5, 1.5))

# Tokamak geometry parameters
R = 0.76
r = 0.251
h = 0.15
H = h + r
t = 0.1 # thickness coils

""" Manual definition """
'''
tokamak.add_active_coil(coil_name="solenoid", coil_part="", center=(0.210, 0), dR=0.030, dZ=0.80, radius=(0,0), geometry=(1, 1))
tokamak.add_active_coil(coil_name="P1_u", coil_part="upper", center=(R-r+t/2, H+t), dR=0.1, dZ=0.1, geometry=(1,1))
tokamak.add_active_coil(coil_name="P1_l", coil_part="lower", center=(R-r+t/2, -H-t), dR=0.1, dZ=0.1, geometry=(1,1))
tokamak.add_active_coil(coil_name="P2_u", coil_part="upper", center=(R+r, H+t), dR=0.1, dZ=0.1, geometry=(1,1))
tokamak.add_active_coil(coil_name="P2_l", coil_part="lower", center=(R+r, -H-t), dR=0.1, dZ=0.1, geometry=(1,1))
tokamak.add_active_coil(coil_name="P3_u", coil_part="upper", center=(R+r+t*1.5, h+t/2), dR=0.1, dZ=0.1, geometry=(1,1))
tokamak.add_active_coil(coil_name="P3_l", coil_part="lower", center=(R+r+t*1.5, -h-t/2), dR=0.1, dZ=0.1, geometry=(1,1))

n = 15

r_limiter = np.concatenate([np.array([R-r]),
                            R + r * np.cos(np.linspace(np.pi, 0, n)),
                            # np.array([R+r, R+r, R+r]),
                            R + r * np.cos(np.linspace(0, -np.pi, n)),
                            np.array([R-r])])
z_limiter = np.concatenate([np.array([0]),
                            h + r * np.sin(np.linspace(np.pi, 0, n)),
                            # np.array([h, 0, h]),
                            - h + r * np.sin(np.linspace(0, -np.pi, n)),
                            np.array([0])])

tokamak.add_limiter(r_limiter, z_limiter)

r_wall = r_limiter.copy()
z_wall = z_limiter.copy()

tokamak.add_wall(r_wall, z_wall)

tokamak.initialize_object() # As opposed to tokamk.initialize_object_from_files(...)
'''

# Initiate tokamak object from files in 'tokamak' folder
tokamak.initialize_object_from_files("./tokamak_97682/")

# Instantiate equilibrium object
tokamak.instantiate_equilibrium(
            Rmin=0.4, Rmax=1.1,
            Zmin=-0.5, Zmax=0.5,
            nx=65,  # (2**n + 1) # 33 - 65 - 129 - 257
            ny=65  # (2**n + 1)
)

# Force current for solenoid coil
print(tokamak.eq.tokamak.coils)
tokamak.set_coil_current("solenoid", 1_000)
# Define coil currents when doing forward solve
# tokamak.set_coil_current("P1", 97682.0489027277)
# tokamak.set_coil_current("P2", -95504.00570204547)
# tokamak.set_coil_current("P3", 17718.028392199238)

# Instantiate PaxisIp profile object
tokamak.instantiate_PaxisIp_profile(
            paxis=1_600,  # pressure on the magnetic axis
            Ip=100_000,  # plasma current
            fvac=1 * R,  # fvac = R B_{tor}
            alpha_m=1.8,  # profile function parameter m
            alpha_n=1.2  # profile function parameter n
)

# Instantiate solver object
tokamak.load_static_nonlinear_solve()

# Defining isoflux set of points for constraints
from miller_representation import miller
miller = miller(elongation=1.5,
                triangularity=0.4,
                R0=R,
                r=3/5*r)

R, Z = miller.get_coordinates(np.arange(0,2*np.pi, 2*np.pi/10))
isoflux_set = np.array([[R, Z]])

null_points = None
# null_points = miller.get_coordinates(np.array([np.pi/2, -np.pi/2]))

tokamak.set_constraints(null_points, isoflux_set)

# Perform inverse solve
tokamak.inverse_solve(
    target_relative_tolerance=1e-6,
    target_relative_psit_update=1e-3,
    verbose=False,
    l2_reg=1e-11
)

# Perform forward solve
# tokamak.forward_solve(
#     target_relative_tolerance=1e-6,
#     verbose=True,
# )

# Perform nonlinear solve for profile stability evolution
# tokamak.nonlinear_solve(
#     plasma_resistivity=1e-6,
#     min_dIy_dI=0.1,  # Minimum coupling threshold for excluding vessel modes (must be a number in [0,1]).
#     threshold_dIy_dI=0.8,  # Relative coupling threshold for including vessel modes (must be a number in [0,1]).
#     max_mode_frequency=1e5,  # Threshold frequency for retaining vessel modes.
#     full_timestep=1e-6
# )

print(tokamak.tokamak.getCurrents())
tokamak.plot_solver()

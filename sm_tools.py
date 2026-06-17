# sm_tools.py
# Handy functions we reuse across our self-assembly projects:
# pair potentials, periodic-boundary distances, and a couple of observables.

import numpy as np


# Lennard-Jones pair potential
def lennard_jones(r, epsilon=1.0, sigma=1.0):
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6 * sr6 - sr6)


# magnitude of the Lennard-Jones force, F = -dU/dr
def lennard_jones_force(r, epsilon=1.0, sigma=1.0):
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    return (24.0 * epsilon / r) * (sr12 - sr6)


# WCA potential (purely repulsive Lennard-Jones)
def wca(r, epsilon=1.0, sigma=1.0):
    rc = 2.0 ** (1.0 / 6.0) * sigma
    if r >= rc:
        return 0.0
    return lennard_jones(r, epsilon, sigma) + epsilon


# wrap a 1D displacement into the box (minimum-image convention)
def minimum_image(dr, box):
    return dr - box * int(dr / box)


# distance between two particles in a periodic box
def pair_distance(r1, r2, box):
    dx = minimum_image(r2[0] - r1[0], box)
    dy = minimum_image(r2[1] - r1[1], box)
    dz = minimum_image(r2[2] - r1[2], box)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


# total kinetic energy
def kinetic_energy(velocities, masses):
    v2 = np.sum(velocities ** 2, axis=1)
    return 0.5 * np.sum(masses * v2)


# temperature from equipartition (kB = 1)
def temperature(velocities, masses):
    ke = kinetic_energy(velocities, masses)
    n = velocities.shape[0]
    return 2.0 * ke / (3.0 * n)

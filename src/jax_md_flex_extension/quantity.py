#!/usr/bin/env python

import jax.numpy as jnp
from jax_md import space, util


Array = util.Array
Box = space.Box


def pressure_tensor(
    force: Array,
    stress: Array,
    position: Array,
    velocity: Array,
    mass: Array,
    box: Box,
    **kwargs,
) -> Array:
    """Calculates the pressure tensor of a flexible box according to Eq. 11 in [1]

    References:
    [1] T.-Q. Yu et al. 10.1016/j.chemphys.2010.02.014

    Args:
        force (Array): Forces on atoms (n_atoms, 3)
        stress (Array): Stress on box (3, 3)
        position (Array): Positions of atoms (n_atoms, 3)
        velocity (Array): Velocities of atoms (n_atoms, 3)
        mass (Array): masses of atoms (n_atoms,)
        box (Box): Cell vectors of box (rows)

    Returns:
        Array: [description]
    """
    vol = jnp.linalg.det(box)
    # einsum might not work properly on CPUs
    kinetic = jnp.einsum("ij,ik->jk", mass * velocity, velocity)
    R = space.transform(box, position)
    F_ext_R = jnp.einsum("ij,ik->jk", force, R)

    nruter = 1 / vol * (kinetic + F_ext_R) - stress
    return (nruter + nruter.T) / 2

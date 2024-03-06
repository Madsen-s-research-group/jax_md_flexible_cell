# Copyright 2023-2024 Sebastian Bichermaier <sebastian.bichelmaier@tuwien.ac.at>
# Copyright 2023-2024 Jesús Carrete Montaña <jesus.carrete.montana@tuwien.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

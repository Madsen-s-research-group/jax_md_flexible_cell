#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import copy
import datetime
import json
import pathlib
import pickle
import sys

import flax
import flax.jax_utils
import flax.serialization
import jax
import jax.nn
import jax.numpy as jnp
import tqdm
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator,
)
from neuralil.model import (
    NeuralILwithMorse,
    ResNetCore,
)
from neuralil.plain_ensembles.model import PlainEnsemblewithMorse
from neuralil.plain_ensembles.training import *

from neuralil.utilities import *

import jax_md
from jax_md_flex_extension.simulate import *
import numpy as onp
import flax
import ase.io

import numpy as np
import pickle

#### SET UP SIM PARAMS

bar2GPa = 1e-4
GPa2eV = ase.units.GPa
bar2eV = bar2GPa * GPa2eV

TEMPERATURE = 2900  #(K)
PRESSURE = 1  #(bar)
TIMESTEP = .5  # (fs)
TIMESTEP_ASE = TIMESTEP * ase.units.fs
NSTEPS = 240000
WRITE_EVERY = 100
TAU_T = 50  #(fs)
TAU_P = 500  #(fs)
TAU_T_ASE = TAU_T * ase.units.fs
TAU_P_ASE = TAU_P * ase.units.fs
kb = ase.units.kB
kt = TEMPERATURE * kb  # Converison to eV

pev = PRESSURE * bar2eV

#### SET UP FORCE FIELD
MODEL_FILE = 'model_params.pkl'

with open(MODEL_FILE, 'rb') as f:
    model_info = pickle.load(f)
N_MAX = model_info.n_max
R_CUT = model_info.r_cut
EMBED_D = model_info.embed_d
N_ENSEMBLE = 5
CORE_WIDTHS = model_info.core_widths
n_types = 2
max_nbrs = 65
max_neighbors = max_nbrs

descriptor_generator = PowerSpectrumGenerator(
    N_MAX, R_CUT, n_types, max_neighbors
)

core_model = ResNetCore(CORE_WIDTHS)

individual_model = NeuralILwithMorse(
    n_types,
    EMBED_D,
    R_CUT,
    descriptor_generator,
    descriptor_generator.process_some_data,
    core_model,
)
dynamics_model = PlainEnsemblewithMorse(individual_model, N_ENSEMBLE)

template_params = dynamics_model.init(
    jax.random.PRNGKey(0),
    jnp.zeros((42, 3)),
    jnp.zeros(42, dtype=jnp.asarray(1).dtype),
    jnp.eye(3),
    method=dynamics_model.calc_forces
)

model_params = jax.tree_map(
    jnp.asarray,
    flax.serialization.from_state_dict(template_params, model_info.params)
)


def calc_potential_energy_from_scaled_positions(sp, t, c):
    p = sp @ c
    nruter = dynamics_model.apply(
        model_params, p, t, c, method=dynamics_model.calc_potential_energy
    )
    return nruter.mean()


@jax.jit
def calc_energy_force_stress(positions, box, **kwargs):
    c = box.T
    s = _calc_stress(positions, types, c)
    p = positions @ c
    e, f = dynamics_model.apply(
        model_params,
        p,
        types,
        c,
        method=dynamics_model.calc_potential_energy_and_forces
    )
    return e.mean(axis=0), e.std(axis=0), f.mean(axis=0), jnp.sqrt(
        f.var(axis=0).sum()
    ), s


def _calc_stress(scaled_positions, types, cell):
    deformation_energy = lambda epsilon: calc_potential_energy_from_scaled_positions(
        scaled_positions, types, cell
        @ (np.eye(3) + .5 * (epsilon + epsilon.T))
    )
    nruter = jax.grad(deformation_energy)(jnp.zeros_like(cell)
                                         ) / jnp.fabs(jnp.linalg.det(cell))
    return nruter

#### SET UP INITIAL CONFIG
at_in = ase.io.read('at_start_1500.vasp').repeat([2, 2, 2])
atoms = at_in.copy()

positions = jnp.array(atoms.get_scaled_positions())
types_sorted = sorted(list(set(atoms.get_chemical_symbols())))
type_translate = {typ: index for index, typ in enumerate(types_sorted)}
types = jnp.array([type_translate[i] for i in atoms.get_chemical_symbols()])
cell = jnp.array(atoms.cell[...]).T
masses = jnp.array(atoms.get_masses())

#### SET UP SIM SPACE
disp_fn, shift_fn = jax_md.space.periodic_general(
    cell, fractional_coordinates=True
)

#### SET UP INIT & STEP FUNCTION
init_fn, step_fn = npt_nose_hoover_flex(
    energy_force_stress_function=calc_energy_force_stress,
    shift_fn=shift_fn,
    dt=TIMESTEP_ASE,
    pressure=pev,
    kT=kt,
    barostat_kwargs={"tau": TAU_P_ASE},
    thermostat_kwargs={"tau": TAU_T_ASE}
)

jitted_step = jax.jit(step_fn)

@jax.jit
def get_pressure(state):
    p = pressure_tensor(
        state.force, state.stress_tensor, state.position, state.velocity,
        state.mass, state.box_position
    )
    return 1. / 3. * jnp.trace(p)

rng = jax.random.PRNGKey(42)

#### INITIALIZE & GO 
state = init_fn(rng, positions, box=cell, mass=masses, kT=kt, types=types)

i = 0
with ase.io.Trajectory(f'trajectory.traj', 'w', atoms) as traj:
    with open(f'out_log_flex.log', 'w') as log_file:
        log_file.write(
            '# Time (fs) | T (K) | Volume (A^3) | Pressure (Pa) | Energy (eV) | e-Uncertainty (eV) | f-Uncertainty (eV/A)\n'
        )
        pbar = tqdm(total=NSTEPS)
        temp = jax_md.quantity.temperature(velocity=state.velocity, mass=state.mass) / kb
        vol = jnp.linalg.det(state.box_position)
        invariant_old = 0
        for i in range(NSTEPS):
            state = jitted_step(state)
            if i % 10 == 0:
                temp = jax_md.quantity.temperature(
                    velocity=state.velocity, mass=state.mass
                ) / kb

                pressure = get_pressure(state)
                energy = state.energy
                
                at = atoms.copy()
                at.set_cell(state.box_position.T)
                at.set_scaled_positions(state.position)
                traj.write(at)
                PE = state.energy
                KE = jax_md.quantity.kinetic_energy(velocity=state.velocity, mass=state.mass)
                dim = state.box_position.shape[0]
                DOF = state.position.size

                c = state.thermostat

                vol = jnp.linalg.det(state.box_position) / positions.shape[0]
                log_file.write(
                    f'{i*.1:.3f}\t{temp:.2f}\t{vol:.5f}\t{pressure/bar2eV:.2f}\t{energy:.2f}\t{state.energy_uncertainty:.3f}\t{state.force_uncertainty:.3f}\n'
                )
                pbar.update(10)
                pbar.set_postfix_str(
                    f'T: {temp:.2f} p: {pressure/bar2eV:.2f}   V: {vol:.1f}  sF: {state.force_uncertainty:.1f}'
                )

            prev_state = state

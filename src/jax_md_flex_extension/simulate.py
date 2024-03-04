#!/usr/bin/env python

from jax_md.simulate import *
from jax_md_flex_extension.quantity import *


def kinetic_energy_box(velocity_box: Array, mass: Array) -> float:
    return 0.5 * util.high_precision_sum(
        jnp.diag(velocity_box.T @ velocity_box) * mass
    )


def box_temperature(box_velocity: Array, mass: float = 1.0) -> float:
    _, dim = box_velocity.shape
    return (jnp.trace(box_velocity.T @ box_velocity) * mass) / (dim**2 - dim)


def canonicalize_mass(mass: Union[float, Array]) -> Union[float, Array]:
    if isinstance(mass, float):
        return mass
    elif isinstance(mass, jnp.ndarray):
        if len(mass.shape) == 2 and mass.shape[1] == 1:
            return mass
        elif len(mass.shape) == 1:
            return jnp.reshape(mass, (mass.shape[0], 1))
        elif len(mass.shape) == 0:
            return mass
    elif isinstance(mass, f32) or isinstance(mass, f64):
        return mass
    msg = (
        "Expected mass to be either a floating point number or a one-dimensional"
        "ndarray. Found {}.".format(mass)
    )
    raise ValueError(msg)


@dataclasses.dataclass
class NPTNoseHooverState:
    """State information for an NPT system with Nose-Hoover chain thermostats.

    Attributes:
      position: The current position of particles. An ndarray of floats
        with shape [n, spatial_dimension].
      velocity: The velocity of particles. An ndarray of floats
        with shape [n, spatial_dimension].
      force: The current force on the particles. An ndarray of floats with shape
        [n, spatial_dimension].
      mass: The mass of the particles. Can either be a float or an ndarray
        of floats with shape [n].
      reference_box: A box used to measure relative changes to the simulation
        environment.
      box_position: A positional degree of freedom used to describe the current
        box. The box_position is parameterized as `box_position = (1/d)log(V/V_0)`
        where `V` is the current volume, `V_0` is the reference volume and `d` is
        the spatial dimension.
      box_velocity: A velocity degree of freedom for the box.
      box_mass: The mass assigned to the box.
      barostat: The variables describing the Nose-Hoover chain coupled to the
        barostat.
      thermostat: The variables describing the Nose-Hoover chain coupled to the
        thermostat.
    """

    position: Array
    velocity: Array
    force: Array
    mass: Array

    reference_box: Box

    box_position: Array
    box_velocity: Array
    box_mass: Array

    barostat: NoseHooverChain
    thermostat: NoseHooverChain

    stress_tensor: Array = None
    energy: float = None
    energy_uncertainty: float = None
    force_uncertainty: float = None
    dUdV: float = None


def npt_nose_hoover_flex(
    energy_force_stress_function: Callable,
    shift_fn: ShiftFn,
    dt: float,
    pressure: float,
    kT: float,
    barostat_kwargs: Optional[Dict] = None,
    thermostat_kwargs: Optional[Dict] = None,
    tau_box: float = None,
) -> Simulator:
    """Simulation in the NPT ensemble using a pair of Nose Hoover Chains
    with a completely flexible cell.

    Samples from the canonical ensemble in which the number of particles (N),
    the system pressure (P), and the temperature (T) are held constant. We use a
    pair of Nose Hoover Chains (NHC) described in [1, 2, 3] coupled to the
    barostat and the thermostat respectively. We follow the direct translation
    method outlined in [3] and the interested reader might want to look at that
    paper as a reference.

    Args:
      energy_force_stress_fn: A function that produces the energy, forces and stress
        from a set of particle positions specified as an ndarray
        of shape [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      pressure: Floating point number specifying the target pressure. To update
        the pressure dynamically during a simulation one should pass `pressure`
        as a keyword argument to the step function.
      kT: Floating point number specifying the temperature in units of Boltzmann
        constant. To update the temperature dynamically during a simulation one
        should pass `kT` as a keyword argument to the step function.
      barostat_kwargs: A dictionary of keyword arguments passed to the barostat
        NHC. Any parameters not set are drawn from a relatively robust default
        set.
      thermostat_kwargs: A dictionary of keyword arguments passed to the
        thermostat NHC. Any parameters not set are drawn from a relatively robust
        default set.
      tau_box: barostat constant for the box used to update box mass, optional

    Returns:
      See above.

    [1] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
        "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
        The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
    [2] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
        "Explicit reversible integrators for extended systems dynamics."
        Molecular Physics 87. (1998) 1117-1157.
    [3] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
        Andrea L. Jochim, and Glenn J. Martyna.
        "A Liouville-operator derived measure-preserving integrator for molecular
        dynamics simulations in the isothermal-isobaric ensemble."
        Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
    [4] T.-Q. Yu et al.
    """

    dt = f32(dt)
    dt_2 = f32(dt / 2)

    barostat_kwargs = default_nhc_kwargs(1000 * dt, barostat_kwargs)
    barostat = nose_hoover_chain(dt, **barostat_kwargs)

    thermostat_kwargs = default_nhc_kwargs(100 * dt, thermostat_kwargs)
    thermostat = nose_hoover_chain(dt, **thermostat_kwargs)

    tau_box = tau_box if tau_box is not None else barostat_kwargs["tau"]

    def init_fn(key, R, box, mass=f32(1.0), **kwargs):
        N, dim = R.shape

        _kT = kT if "kT" not in kwargs else kwargs["kT"]
        mass = canonicalize_mass(mass)
        V = jnp.sqrt(_kT / mass) * random.normal(key, R.shape, dtype=R.dtype)
        V = V - jnp.mean(V * mass, axis=0, keepdims=True) / mass
        KE = quantity.kinetic_energy(velocity=V, mass=mass)

        # The box position is the cell
        zero = jnp.zeros((dim, dim), dtype=R.dtype)
        one = jnp.ones((), dtype=R.dtype)
        box_position = box.copy()
        box_velocity = zero

        # adapted mass cf martyna, tuckerman et al, 1996 Molecular Physics
        box_mass = (N + 1) * kT * tau_box**2 * one
        KE_box = kinetic_energy_box(box_velocity, box_mass)

        energy, energy_dev, force, force_dev, stress = (
            energy_force_stress_function(R, box=box, **kwargs)
        )
        state = NPTNoseHooverState(
            R,
            V,
            force,
            mass,
            box,
            box_position,
            box_velocity,
            box_mass,
            barostat.initialize(dim**2 - dim, KE_box, _kT),
            thermostat.initialize(R.size, KE, _kT),
            stress,
            energy,
            energy_dev,
            force_dev,
        )  # pytype: disable=wrong-arg-count
        return state

    def update_box_mass(state, kT):
        N, dim = state.position.shape
        dtype = state.position.dtype
        box_mass = jnp.array((N + 1) * kT * tau_box**2, dtype)
        return dataclasses.replace(state, box_mass=box_mass)

    def box_force(
        box, position, velocity, mass, pressure, force, stress, **kwargs
    ):
        """Generates the force on the box according to Eq. 41 in [1]

        [1] T.-Q. Yu et al.

        Args:
            box (Array): The unit cell vectors (row), (3,3)
            position (Array): Atomic positions, (n_atoms, 3)
            velocity (Array): Velocities of atoms (n_atoms, 3)
            mass (Array): Masses of atoms (n_atoms)
            pressure (float): External pressure in eV/A^3
            force (Array): Forces on atoms (n_atoms, 3)
            stress (Array): Stress tensor (3,3)

        Returns:
            Array: Force on box
        """
        N, dim = velocity.shape

        KE2 = util.high_precision_sum(velocity**2 * mass)
        p_int = pressure_tensor(force, stress, position, velocity, mass, box)

        box_contrib = jnp.linalg.det(box) * (p_int - jnp.eye(dim) * pressure)
        nruter = box_contrib + 1.0 / (N * dim) * KE2 * jnp.eye(dim)
        return nruter

    def sinhx_x(x):
        """Taylor series for sinh(x) / x as x -> 0."""
        return 1 + x**2 / 6 + x**4 / 120 + x**6 / 5040.0

    def exp_iL1(lam, O, R, V, box, **kwargs):
        x = lam * dt
        x_2 = x / 2
        vexpdt = jnp.exp(x)
        sinhV = sinhx_x(x_2)
        vsindt = jnp.exp(x_2) * sinhV

        positions = space.transform(box, R)
        inv_box = jnp.linalg.inv(box)

        vtempx = O * vexpdt[jnp.newaxis, :]
        vtempv = O * vsindt[jnp.newaxis, :]

        roll_mtx = vtempx @ O.T
        roll_mtv = vtempv @ O.T

        tempx = positions @ roll_mtx
        tempv = V @ roll_mtv

        R_new = space.transform(inv_box, tempx)
        R_return = shift_fn(R_new, dt * tempv, box=box, **kwargs)

        hmat_t = O.T @ box
        hmat_t = hmat_t * vexpdt[:, jnp.newaxis]

        hmat_new = O @ hmat_t

        return R_return, hmat_new

    def exp_iL2(lam, O, bTr, A, V):
        x = (lam + bTr) * dt_2
        x_2 = x / 2

        vexpdt = jnp.exp(-x)
        vsindt = jnp.exp(-x_2) * sinhx_x(x_2)

        vtempf = O * vsindt[jnp.newaxis, :]
        vtempv = O * vexpdt[jnp.newaxis, :]

        roll_mtf = vtempf @ O.T
        roll_mtvv = vtempv @ O.T

        tempv = V @ roll_mtvv
        tempf = A @ roll_mtf

        V_new = tempv + tempf * dt_2
        return V_new

    def inner_step(state: NPTNoseHooverState, **kwargs):
        """Propagate the box and particle DOF for a full timestep"""
        _pressure = kwargs.pop("pressure", pressure)

        R, V, M, F = state.position, state.velocity, state.mass, state.force
        R_b, V_b, M_b, S_b = (
            state.box_position,
            state.box_velocity,
            state.box_mass,
            state.stress_tensor,
        )

        N, dim = R.shape

        G_g = box_force(R_b, R, V, M, _pressure, F, S_b, **kwargs)
        V_b = V_b + dt_2 * G_g / M_b

        lam, O = jnp.linalg.eigh(V_b)
        bTr = 1 / (dim * N) * lam.sum()

        V = exp_iL2(lam, O, bTr, F / M, V)

        R, R_b = exp_iL1(lam, O, R, V, R_b, **kwargs)

        energy, energy_dev, F, force_dev, S_b = energy_force_stress_function(
            R, box=R_b, **kwargs
        )

        V = exp_iL2(lam, O, bTr, F / M, V)

        G_g = box_force(R_b, R, V, M, _pressure, F, S_b, **kwargs)
        V_b = V_b + dt_2 * G_g / M_b

        return dataclasses.replace(
            state,
            position=R,
            velocity=V,
            force=F,
            box_position=R_b,
            box_velocity=V_b,
            stress_tensor=S_b,
            reference_box=R_b,
            energy=energy,
            energy_uncertainty=energy_dev,
            force_uncertainty=force_dev,
        )

    def apply_fn(state: NPTNoseHooverState, **kwargs):
        """Propagate state for a full timestep

        Arguments:
           state: The state of the system at time t

        Returns:
           state: The state of the system at time t+dt
        """
        S = state
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        bc = barostat.update_mass(S.barostat, _kT)
        tc = thermostat.update_mass(S.thermostat, _kT)
        S = update_box_mass(S, _kT)

        V_b, bc = barostat.half_step(S.box_velocity, bc, _kT)
        V, tc = thermostat.half_step(S.velocity, tc, _kT)

        S = dataclasses.replace(S, velocity=V, box_velocity=V_b)
        S = inner_step(S, **kwargs)

        KE = quantity.kinetic_energy(velocity=S.velocity, mass=S.mass)
        tc = dataclasses.replace(tc, kinetic_energy=KE)

        KE_box = kinetic_energy_box(S.box_velocity, S.box_mass)
        bc = dataclasses.replace(bc, kinetic_energy=KE_box)

        V, tc = thermostat.half_step(S.velocity, tc, _kT)
        V_b, bc = barostat.half_step(S.box_velocity, bc, _kT)

        S = dataclasses.replace(
            S, thermostat=tc, barostat=bc, velocity=V, box_velocity=V_b
        )

        return S

    return init_fn, apply_fn

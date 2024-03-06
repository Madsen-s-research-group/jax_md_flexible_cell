# Flexible-cell extension for JAX-MD

This repository contains an extension on top of [JAX-MD](https://github.com/jax-md/jax-md) to enable molecular-dynamics simulations in the isothermal-isobaric ensemble where not only the volume, but also the cell shape, can change along the trajectory.

See the file [examples/jax_md_flex.py](examples/jax_md_flex.py) for a full example of its use. Note that the example uses the [NeuralIL](https://doi.org/10.5281/zenodo.10786377) force field. The link will take you to the specific version the example was tested with.
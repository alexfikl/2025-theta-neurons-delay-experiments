# Stability of equilibria in an infinite dimensional network of theta neurons with time delay

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)
[![Zenodo repository](https://zenodo.org/badge/DOI/10.5281/zenodo.20718310.svg)](https://doi.org/10.5281/zenodo.20718310)

This code accompanies the paper. It contains all the scripts used to generate
the figures and results in the paper.

# Dependencies

The dependencies for the project are listed in the `requirements.txt` file. All
dependencies are pinned to the latest version known to work (i.e. as published
with the paper). If you want to reproduce the results exactly, it is highly
recommended to use these dependencies. However, if the results do not reproduce
with newer versions, this is likely a bug!

Besides this, the user is expected to have

* An up to date C compiler (e.g. GCC>=14.0). This is required by the JiTCDDE
  library to compile the native Python extensions that significantly speed up the
  simulations.
* [Optional] An up to date (>=2025) LaTeX installation. This dependency can be
  removed by setting `text.usetex: False` in `default.mplstyle`, as it is only
  required for plotting.

# Installation

It is recommended to install the (Python) dependencies in a [virtual
environment](https://docs.python.org/3/library/venv.html) to avoid conflicts
with other packages. The dependencies can then be installed using
[pip](https://pip.pypa.io/en/stable/user_guide/)
```bash
pip install -r requirements.txt
```
or
[uv](https://docs.astral.sh/uv/pip/packages/#installing-packages)
```bash
uv pip sync requirements.txt
```

# Reproducing the results

The included `justfile` has all the necessary invocations to reproduce the figures
in the paper. You can just run
```bash
just figure1
just figure2
just figure345
just figure6
just figure7
just figure8
just figure9
just figure10
```
to get all the figures. The runs for Figure 7-10 are significantly slower
(10-15 min) than the rest, since they run actual simulations for thousands of
initial conditions. They should generate the corresponding files in the
`experiments/` directory (`npz` archives of the solutions and the plots
themselves).

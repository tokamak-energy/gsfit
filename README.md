# GSFit: Grad-Shafranov Fit
[![The 3-Clause BSD License](https://img.shields.io/pypi/l/prtg-pyprobe)](LICENSE) 
[![Python 3.11|3.12|3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/) 
[![Rust](https://img.shields.io/badge/Rust-1.86-red?logo=rust&logoColor=white)](https://www.rust-lang.org/) 
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) 
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

GSFit is a modern tokamak equilibrium reconstruction tool for post-shot experimental analysis and scientific interpretation.
Developed by Tokamak Energy Ltd. for use on ST40, the worlds highest field spherical tokamak.

The Grad-Shafranov equation is the is equilibrium solution to the ideal, single fluid, MHD equations, assuming axysymmetry.
The Grad-Shafranov equation has two free degrees of freedom, often referred to as the source functions: the pressure gradient `p_prime` and `ff_prime` where `f` gives rise to the the poloidal plasma current. 
GSFit uses diagnostics, most importantly the magnetics, to symultaneiously solve the non-linear Grad-Shafranov equation while optimising the `p_prime` and `ff_prime` degrees of freedom to minimise the difference between the experimental "measured" values and the "calculated" values.
More details about the algorithms used can be found in the excellent paper [J.-M. Moret, *et. al.*, "Tokamak equilibrium reconstruction code LIUQE and its real time implementation", Fusion Eng. Design, 91, 2015](https://doi.org/10.1016/j.fusengdes.2014.09.019).

GSFit's numerical solver is written in Rust for speed and robustness, and the user interface / set-up is written in Python for it's convenience.

The goal of GSFit is high accuracy, not performance; when something can be calculated accurately, it **should** be calculated accurately regardless of computational cost.

GSFit uses the COCOS 13 coordinate system, as described in [O. Sauter and S. Y Medvedev, "Tokamak Coordinate Conventions: COCOS", Comput. Phys. Commun. 184, 2013](https://doi.org/10.1016/j.cpc.2012.09.010).
Where in summary, flux is measured in weber, and the poloidal angle increase counterclockwise, starting from the outboard mid-plane.

**Why Rust and Python?**

Rust is a compiled, high-performance language, nearly as fast as C and Fortran.
It includes modern tooling such as a package manager and centralised registry (cargo and [crates.io](https://crates.io/)), autoformatting ([`rustfmt`](https://github.com/rust-lang/rustfmt)), and has [testing built into the language](https://doc.rust-lang.org/rust-by-example/testing/unit_testing.html).
Rust enforces strict ownership rules, making programs extremely memory efficient without requiring any extra effort.
These rules also eliminate entire classes of memory issues.
Additionally, Rust has zero-cost abstractions, allowing "implementations", whcih are similar to classes in object-oriented programming, to be used without any performance penalty.
For more information the official [Rust book](https://doc.rust-lang.org/stable/book/) gives a complet introduction to the language with examples.

Python is ubiquitous within the fusion industry.
Using Python allows easy integration into existing workflows, for example in
[examples/example_02_mastu_with_synthetic_data_from_freegsnke.ipynb](examples/example_02_mastu_with_synthetic_data_from_freegsnke.ipynb), we reconstruct an equilibrium using synthetic "measured" data produced by the open source forward Grad-Shafranov solver [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke).

# 1. Instiallation and environment

## 1.1 Python environment
Presently, GSFit can be run on Python 3.11, 3.12 and 3.13 (see the banner at the top of this page).

When installing from PyPi, see [section 1.2](#12-installing-from-pypi), you can use the system Python, but it is best practice to use a virtual environment.

There are three main virtual environment providers: [`uv`](https://docs.astral.sh/uv/), [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html), and [`virtualenv`](https://docs.python.org/3/library/venv.html).

`uv` is both an environment and package manager and is the **recomended** option.
This is because `uv` can create an environment with any version of Python, the instillation is done within the user's home directory, and `uv` has been specifically designed for speed.
This makes it simple and quick to test against multiple Python versions.

```shell
# Install the `uv` environment and package manager
python -m pip install uv

# Creating and activating a virtual environment, called `venv_gsfit`
python -m uv venv venv_gsfit --python=3.13
source venv_gsfit/bin/activate
```


## 1.2 Installing from PyPi
GSFit is available on the [PyPi package registry](https://pypi.org/project/gsfit/) as a pre-compiled binary.
The binary is compiled using the "manylinux2014" standard, which typically can be run on any Linux machine newer than 2014.

All of the packages GSFit depends on are listed in the [pyproject.toml](pyproject.toml).
These are divided into essential packages which are always required for any run and optional packages which can be installed for different purposes.
An example of optional packages are machine specific database readers.

```shell
# Install GSFit from the PyPi package registry, with only the essential packages
uv pip install gsfit
# or install GSFit with the "developer" packages, such as `pytest` and `mypy`
uv pip install gsfit[dev]
# or install GSFit with reading/writing to/from ST40's experimental database
# (this will only work within Tokamak Energy's network)
uv pip install --reinstall .[with_st40_mdsplus]
# or any combination
uv pip install --reinstall .[dev,with_st40_mdsplus]
```

## 1.3 Compiling and installin from source code
While GSFit can be installed into the system Python from PyPi, when compiling from source code a virtual environment is <ins>**required**</ins>.

The Rust compiler is easily installed using [`rustup`](https://www.rust-lang.org/tools/install).

GSFIt also requires [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) to perform various LAPACK routines, such as least squares minimisation and eignevalue/eigenvector calculations.

```shell
# Install the Rust compiler (only needs to be done once)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone a copy of GSFit from GitHub
git clone git@github.com:tokamak-energy/gsfit.git
cd gsfit

# Load OpenBLAS (gsfit is statically linked so this is only required when compiling)
module avail
module load OpenBLAS

# Install GSFit
uv pip install --reinstall .
```

## 1.4 IDE
[VS Code](https://code.visualstudio.com/) is the recomended IDE for GSFit.

A particularly useful extension is [`rust-analyser`](https://github.com/rust-lang/rust-analyzer), which *"is a part of a larger rls-2.0 effort to create excellent IDE support for Rust."*

The [`PyO3`](https://crates.io/crates/pyo3) crate requires Python>=3.7.
On machines where the `python3` command links to an unsupported Python version, you can change the following file: `~/.vscode-server/data/Machine/settings.json` to point to your desired Python's "standard library files". Which will be <ins>**similar**</ins> to:
```json
{
    "rust-analyzer.server.extraEnv": {
        "PYO3_CROSS_LIB_DIR": "/home/<user.name>/.local/share/uv/python/cpython-3.13.2-linux-x86_64-gnu/lib"
    }
}
```
and Python's "standard library files" directory can be found by:
```shell
python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
```

# 2. GSFit program layout and flow
## 2.1 Initialisation
GSFit's solver and numerics are programmed in Rust, with the initialisation done within Python.

The communication between Python and Rust is handled by the [`PyO3`](https://crates.io/crates/pyo3) crate.
From the Python side the integration is seemless, and a general user would not know if a fuction / class is written in Python or Rust.
In a similar way a user interacting with `numpy` does not know if they are calling Python, C or Fortran.

GSFit has the following implementations (which are similar to classes):
* **`coils`** for the PF and TF coils. This object contains the geometry; measured values; and reconstructed values
* **`passives`** for the toroidally conducting strucutres, such as the vacuum vessel and passive plates. `passives` also contains the degrees of freedom allowed for the passive conductors
* **`plasma`** contains a $(R, Z)$ grid and the reconstructed equilibrium
* **`bp_probes`** contains the Mrinov coils: geometry; measured values; which sensors to use in the reconstruction and their weights; and the reconstructed values
* **`flux_loops`** the toroidal flux loops. Note the units are in weber.
* **`rogowski_coils`** measuring toroidal currents
* **`isoflux`** contains two time-dependent $(R, Z)$ coordinates which have equal poloidal flux
* **`isoflux_boundary`** contains a time-dependent $(R, Z)$ coordinate which has the same poloidal flux as the plasma boundary. This can be used if we have measurements of where the strike point is, for example if there is an IR camera

All of these implementations are initialised from Python.
As a result, all of the database access is contained within Python, and any pre-processing such as filtering or smoothing experimental data can also be done within Python.

Below is a reproduction of [`examples/example_01_st40_with_experimental_data.py`](examples/example_01_st40_with_experimental_data.py).
```python
# Note, this example will only run inside Tokamak Energy's network

from gsfit import Gsfit

# Construct the GSFit object
gsfit_controller = Gsfit(
    pulseNo=12050,
    run_name="TEST01",
    run_description="Test run",
    write_to_mds=True,
    settings_path="default"
)

# Run (solve the inverse Grad-Shafranov equation, and if requested write to database)
gsfit_controller.run()
```

To improve reliability and tracability the number of arguments needed to initialise the `gsfit_controller` object is <ins>deliberately kept to a minimum</ins>.
The `settings_path="default"` argument tells GSFit to use the settings (JSON files) from this directory [`python/gsfit/settings/default/`](python/gsfit/settings/default/).
By storing all of the settings needed to run GSFit in JSON files allows changes to be tracked through Git.

## 2.2 Adding a new experimental device or coupling to a new forward Grad-Shafranov code
The information needed to run GSFit comes from two sources:

1. **[`python/gsfit/settings/`](python/gsfit/settings/)**: Contains parameters in JSON files, such as:
    * the maximum number of iterations,
    * which magnetic sensors to use in the reconstruction,
    * the degrees of freedom given to `p_prime` and `ff_prime`,
    * <ins>**and**</ins> which `database_reader` to use.
2. **[`python/gsfit/database_readers/`](python/gsfit/database_readers/)**: Connects to machine specific databases to read experimental results in, such as:
    * geometry and current in poloidal field coils
    * measured signals from magnetic sensors
    * isoflux constraints

These two sources of information are combined to initialise the Rust objects needed to run GSFit (as described in [section 2.1](#21-initialisation)).

Writing to the database is done through [`python/gsfit/database_writers`](python/gsfit/database_writers), in a similar format to `database_readers`.

Included are the settings and readers for ST40's experimental data, and synthetic data produced by [FreeGS](https://github.com/freegs-plasma/freegs) and [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke/) simulations.
We welcome and are happy to inclue configurations for other devices.

## 2.3 Where are the Green's tables stored?
We can classify objects as either "current sources" or "sensors".
* `coils`, `passives`, and `plasma` are "current sources".
* `bp_probes`, `flux_loops`, `rogowski_coils`, `isoflux`, and `plasma` are "sensors".

Note, `plasma` is both a current source and a sensor, this is because it contains the plasma current **and** the 2D (R, Z) grid where we want to take measurement on.

Principles about where data should be:
* "sensors" contain the Greens tables.
* "current sources" contain the currents.

the sensors and currens are linked by names, every magnetic probe and PF coil must have a unique name.

<!--
So the field which `bp_probe` `P101` detects will be:
```rust
field_bp_probe_101_detects = bp_probes["P101"]["greens"]["pf"]["SOL"] * coils["pf"]["SOL"]["current"]["measured"] + ... // PF coils
    + bp_probes["P101"]["greens"]["plasma"] * plasma["two_d"]["j"] // plasma
    + ... 
```
-->

## 2.4 Decision on coils vs PSU's
On many tokamaks, including ST40, several PF coils are connected to a single PSU.
For ST40 we have the `BVLT` and `BVLB` PF coils, both connected to the `BVL` PSU.

Several equilibrium reconstruction codes treat the PSU current as a measurement with an associated degree of freedom.
To allow for this degree of freedom we would need to tell GSFit which PF coil is connected to which PSU.
This is quite burdensom and **to keep GSFit simple to initialise, we have decided to only consider "coils"**.
The work-around, if you want to have the PSU currents as a degree of freedom, is to combine all of the PF coils connected to a single power supply into a single coil.
On ST40 we could create a PF coil called `BVL_combined` which contains both `BVLT` and `BVLB`.

# 3. Planned future development
* Pressure constrained.
* Spline `p_prime` and `ff_prime`
* Interfacing to and from IMAS & OMAS.
* More kinetic sensors, such as MSE, polarimeters, and interferometers.

# 4. Citing GSFit
We intend on publishing a paper on GSFit.
While GSFit is unpublished please cite it as "P. F. Buxton, GSFit, https://github.com/tokamak-energy/gsfit, 2025"

Please use the **GSFit** nomenclature, <ins>**not**</ins> GS-Fit, GSFIT, or g/s fit.


<!-- # 4 Useful Rust code for debugging
```rust
use std::time::Instant;
let start = Instant::now();
// code you want to time
let duration = start.elapsed();
println!("Time elapsed = {:?}", duration);
```

Write to file:
```rust
// Imports
use std::fs::File;
use std::io::{BufWriter, Write};
// write to file
let file = File::create("psi_2d_coils.csv").expect("can't make file");
let mut writer = BufWriter::new(file);
for row in psi_2d_coils.rows() {
    let line: String = row.iter()
        .map(|&value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(writer, "{}", line).expect("can't write line");
}
writer.flush().expect("can't flush writer");
``` -->
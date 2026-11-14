from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .dd_support import ParametersInput
from .imas import CoreProfiles as CoreProfilesIds
from .imas import CoreSources as CoreSourcesIds
from .imas import Equilibrium as EquilibriumIds
from .imas import GreensTables
from .imas import Magnetics as MagneticsIds
from .imas import Path
from .imas import PfActive as PfActiveIds
from .imas import PfPassive as PfPassiveIds
from .imas import PulseSchedule as PulseScheduleIds
from .imas import Tf as TfIds
from .imas import Wall as WallIds

_T = TypeVar("_T")

def solve_grad_shafranov(
    equilibrium: Equilibrium,
) -> None:
    """
    :param equilibrium: Equilibrium object, note this is mutated and contains the solution. Its equilibrium IDS supplies the limiter,
        the vacuum toroidal field, the PF coil currents, the passive loops' regularisations and any gaps, so
        `Equilibrium.copy_from_wall_ids`, `Equilibrium.copy_from_tf_ids`, `Equilibrium.copy_from_pf_active_ids` and `Equilibrium.copy_from_pf_passive_ids`
        have to have been called first. `Equilibrium.copy_from_pulse_schedule_ids` is optional: without it the equilibrium has no gaps.
        The passive loops have to be the ones the Greens tables, `greens_tables/grid/pf_passive`, were calculated with, by `Equilibrium.greens_with_pf_passive`.
        The fitted passive degrees of freedom are written to every time-slice's `passive_dof_values`, in the order of `greens_tables/grid/pf_passive`.
        The magnetic sensors fitted are those whose fit settings, the equilibrium settings' `constraints`, include them, with their measurements,
        which `Equilibrium.copy_from_magnetics_ids` copied in, their `time_slice(itime)/constraints/<sensor>(i1)/weight` and `.../expected_value`,
        a custom key, and their Greens tables, which `Equilibrium.greens_with_magnetics` calculated. Each is weighted by `weight / expected_value`.
        When the equilibrium settings' `constraints/pressure` include pressure, the pressure constraints fitted are those of
        `time_slice(itime)/constraints/pressure(i1)` whose `measured` is finite, which `Equilibrium.copy_from_core_profiles_ids` made, with
        their `weight` and `expected_value`; their `reconstructed`, `position/psi` and `chi_squared` are written by the post-processor.
        The constraints at points which can move, `time_slice(itime)/constraints/isoflux(i1)`, `.../isoflux_boundary(i1)` and
        `.../stationary_point(i1)`, custom keys, are fitted where their fit settings include them and their points are known, with their
        points, which `Equilibrium.copy_from_magnetics_ids` copied in from the magnetics IDS, and their Greens tables, one for each
        time-slice, which `Equilibrium.greens_with_magnetics` calculated; their `reconstructed` and `chi_squared` are written by the
        post-processor

    The times to reconstruct are read from `equilibrium`, which was built with one equilibrium
    time-slice per time.
    """
    ...

def solve_circuit_equations(
    pf_active: PfActive,
    pf_passive: PfPassive,
    times_to_solve: npt.NDArray[np.float64],
    adaptive_time_stepping: bool,
) -> tuple[PfActive, PfPassive]:
    """
    Solves the circuit equations

    Note: the adaptive time-stepping can produce a lot of simulated time-points,
    so `adaptive_time_stepping = False` should be used to avoid unwieldly large outputs

    :param pf_active: The PF coils, each with its current prescribed, or its voltage when it has one (`coil(i1)/voltage`)
    :param pf_passive: The passive conductors, whose filaments, `loop(i1)/element(i2)`, carry eddy currents
    :param times_to_solve: Times to solve the circuit equations at [second]
    :param adaptive_time_stepping: The calculation always uses adaptive time stepping, this flag interpolates results onto `times_to_solve`
    :return: A copy of `pf_active`, whose `coil(i1)/current` is the simulated current, on the simulated timebase; and a copy of
        `pf_passive`, whose `element_current` is the simulated current in each `loop(i1)/element(i2)`, on the simulated timebase, `loop(i1)/time`
    """
    ...

def greens_py(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    :param r: (by convention) Sensor radial positions, must be `> 0.0` [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions, must be `> 0.0` [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]

    Note: the inputs are symmetrical
    """
    ...

def greens_d_psi_d_r(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    First derivative of the Green's function with respect to `r`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d_psi_d_z(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    First derivative of the Green's function with respect to `z`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d2_psi_d_r2(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Second derivative of the Green's function with respect to `r`.

    When a sensor coincides with a current source, the source's `d_r` and `d_z`
    are used to evaluate the analytic self-term
    (see `documentation/jump_condition_dbr_dz.md`).

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d2_psi_d_r_d_z(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Mixed second derivative of the Green's function with respect to `r` and `z`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d2_psi_d_z2(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Second derivative of the Green's function with respect to `z`.

    When a sensor coincides with a current source, the source's `d_r` and `d_z`
    are used to evaluate the analytic self-term
    (see `documentation/jump_condition_dbr_dz.md`); if they are omitted the
    self-term is zero.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

def greens_d3_psi_d_z3(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Third derivative of the Green's function with respect to `z` three times.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

def greens_d3_psi_d_r2_d_z(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Third derivative of the Green's function with respect to `r` twice and `z`.

    Computed from the homogeneous Grad-Shafranov equation:
    `d3_psi_d_r2_d_z = d2_psi_d_r_d_z / r - d3_psi_d_z3`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

def greens_d3_psi_d_r_d_z2(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Third derivative of the Green's function with respect to `r` and `z` twice.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

class PfPassive:
    """The machine's passive conductors, e.g. the vacuum vessel, the passive plates and the PF coil cases, stored as an IMAS
    `pf_passive` IDS.

    For each passive conductor, `loop(i1)`: `name`; `resistivity`; one `element(i2)` per filament, a parallelogram
    (`geometry/oblique/r`, `z`, `length_alpha`, `length_beta`, `alpha` and `beta`) of one turn (`turns_with_sign` = 1), with its `area`.

    The data dictionary describes a parallelogram from one of its corners, the reference point `oblique/r` and `z`. One side,
    `length_alpha` long, leaves it at angle `alpha` counter-clockwise from the R axis; the other, `length_beta` long, leaves it at angle
    `beta` counter-clockwise from the Z axis.

    These are machine data only, the conductors themselves, whichever code made the IDS. What the reconstruction fits their currents with is
    the equilibrium's, from its fit settings, `equilibrium/pf_passive_dof.xml`, which say which passive conductors to fit, found here by their
    `name`, and how: `Equilibrium.copy_from_pf_passive_ids` makes their degrees of freedom from these conductors' filaments.

    Read the IDS back through `pf_passive_ids` and a path from `gsfit_rs.imas.pf_passive_paths`. The current `solve_circuit_equations`
    simulates in each element, which the data dictionary has no node for, is kept in the class, and read with `element_current`.

    The passive conductors are constructed with the settings they are built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `pf_passive/code/parameters`. Rust also reads them into structs generated
    from the XML schema `pf_passive.xsd`, kept with the class rather than in the IDS, as the pf_passive IDS can also be made by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> PfPassive:
        """
        Construct empty passive conductors, ready for `add_loop` to be called.

        :param parameters_input: The settings the passive conductors are built from, e.g. which run of which database they are read from, whose
            XML is `pf_passive.xml` with each file it links to written in its link's place, and whose schema is the XML schema it follows, as
            `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is written to `pf_passive/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, or when the XML cannot be read into the structs, as the
            XML schema is not the one they were generated from
        """
        ...
    def element_current(self, loop_name: str) -> npt.NDArray[np.float64]:
        """
        The current in each of a passive conductor's elements, simulated by `solve_circuit_equations`, on `loop(i1)/time`, shape = [n_element, n_time] [ampere].

        :param loop_name: the passive conductor, as named in `add_loop`
        :raises ValueError: When there is no passive conductor with this name, or these passive conductors are not the ones `solve_circuit_equations`
            returned, which are the only ones to hold a simulated current
        """
        ...
    def add_loop(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        length_alpha: npt.NDArray[np.float64],
        length_beta: npt.NDArray[np.float64],
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
        resistivity: float,
    ) -> None:
        """
        Append a passive conductor to `pf_passive/loop`.

        :param name: short identifier for the passive conductor, unique within the passive conductors, e.g. `"IVC"`
        :param r: major radius of each filament's reference point, the corner which `alpha` and `beta` are measured from [metre]
        :param z: height of each filament's reference point [metre]
        :param length_alpha: length of each filament's side inclined at `alpha` [metre]
        :param length_beta: length of each filament's side inclined at `beta` [metre]
        :param alpha: inclination of each filament's first side, counter-clockwise from the R axis [radian]
        :param beta: inclination of each filament's second side, counter-clockwise from the Z axis [radian]
        :param resistivity: resistivity of the conductor, the same for every filament [ohm * metre]
        :raises ValueError: When the name is empty or already used, the arrays are of different lengths or empty, or a filament has no positive
            `r` and lengths

        Each filament becomes one `element(i2)`: a parallelogram of one turn, with a positive `r` and positive lengths, and its `area`.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.pf_passive_paths`, straight out of the pf_passive IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.PfPassive.get`.
        """
        ...
    @property
    def pf_passive_ids(self) -> PfPassiveIds:
        """A copy of the whole pf_passive IDS, read with `gsfit_rs.imas.pf_passive_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Equilibrium:
    """The equilibrium, stored as an IMAS `equilibrium` IDS.

    It is built from GSFit's settings alone, which everything else is made from: one `time_slice(itime)` per time to reconstruct, each with the (R, Z) grid in `profiles_2d(0)`,
    `profiles_1d/psi_norm`, and the weight of each magnetic sensor in the settings, in `constraints/<sensor>(i1)/weight`. The Greens tables,
    which are GSFit's own, starting with those between the grid points, `greens_tables/grid/plasma`, and the p' and ff' source functions are
    held alongside the IDS.

    Read the IDS's data back through `get`, or `equilibrium_ids`, and a path from `gsfit_rs.imas.equilibrium_paths`. Read the Greens tables
    through `greens_tables`, a copy, and a path from `gsfit_rs.imas.greens_tables_paths`.

    The equilibrium is constructed with the settings it is built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `equilibrium/code/parameters`. Rust also reads them into structs
    generated from the XML schema `equilibrium.xsd`, kept with the class rather than in the IDS, as the equilibrium IDS can also be made by
    another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> Equilibrium:
        """
        Construct the equilibrium from GSFit's settings.

        :param parameters_input: GSFit's settings, whose XML is `equilibrium.xml` with each file it links to written in its link's place, and
            whose schema is the XML schema it follows, as `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is
            written to `equilibrium/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, when the XML cannot be read into the structs, as the
            XML schema is not the one they were generated from, or when the settings cannot make an equilibrium, e.g. a radial grid which goes
            negative
        """
        ...
    def copy_from_wall_ids(self, wall: Wall) -> None:
        """
        Copy the limiter from the wall IDS into the equilibrium IDS.

        Copies `wall/description_2d(0)/limiter/unit(i)` into `equilibrium/limiter/unit(i)`, in the same order, so `unit(0)` is
        still the vacuum vessel contour. The solver and the post-processor read the limiter from the equilibrium IDS, so this has
        to be called before `solve_grad_shafranov`.

        Any limiter already in the equilibrium IDS is replaced, so calling this twice does not repeat the units.

        :param wall: The wall, holding the limiter
        :raises ValueError: When the wall has no limiter units
        """
        ...
    def copy_from_tf_ids(self, tf: Tf) -> None:
        """
        Copy the vacuum toroidal field from the tf IDS into the equilibrium IDS.

        Fills `equilibrium/vacuum_toroidal_field/r0` from `tf/r0`, and `equilibrium/vacuum_toroidal_field/b0` from
        `tf/b_field_phi_vacuum_r`. That is on the experimental timebase, so it is interpolated onto the reconstruction times, the
        times of the time-slices `Equilibrium` was built with, and divided by `r0`. The solver and the post-processor read the vacuum
        toroidal field from the equilibrium IDS, so this has to be called before `solve_grad_shafranov`.

        Any vacuum toroidal field already in the equilibrium IDS is replaced.

        :param tf: The toroidal field
        :raises ValueError: When `tf/b_field_phi_vacuum_r` is unset, or when it cannot be interpolated onto the reconstruction
            times, e.g. because its timebase does not cover them
        """
        ...
    def copy_from_pf_active_ids(self, pf_active: PfActive) -> None:
        """
        Copy the PF coil currents from the pf_active IDS onto every time-slice of the equilibrium IDS.

        Fills, for every coil `pf_active/coil(i1)`, in the same order, on every time-slice:
        `equilibrium/time_slice(itime)/constraints/pf_current(i1)/name`, a custom key, from `pf_active/coil(i1)/name`;
        `.../measured`, `pf_active/coil(i1)/current` interpolated onto the time-slice's time [ampere];
        `.../source`, the path `pf_active/coil(i1)/current`; and `.../time_measurement`, the time-slice's own time [second].

        The solver reads the PF coil currents from the equilibrium IDS, so this has to be called before `solve_grad_shafranov`.
        The solver multiplies them by the Greens tables, `greens_tables/grid/pf_active(i1)`, so the coils have to be in the same order as
        those, which the solver checks by name.

        Any PF coil currents already on a time-slice are replaced, so calling this twice does not repeat them.

        :param pf_active: The PF coils, holding their currents
        :raises ValueError: When `pf_active/coil` is empty, or when a coil's current cannot be interpolated onto the
            reconstruction times, e.g. because its timebase does not cover them
        """
        ...
    def copy_from_pf_passive_ids(self, pf_passive: PfPassive) -> None:
        """
        Make the degrees of freedom the passive conductors' currents are fitted with, from the fit settings and the passive conductors'
        filaments, and copy them, with their regularisations, onto every time-slice of the equilibrium IDS.

        The fit settings, `equilibrium/pf_passive_dof.xml`, say which passive conductors to fit, each found in `pf_passive/loop` by its
        `name`, and how. Fills, for each of them, in the order of the fit settings, on every time-slice, these custom keys:
        `equilibrium/time_slice(itime)/constraints/pf_passive_current(i1)/name`, the passive conductor's `name`; `.../dof(i2)`, one for each
        degree of freedom, as its `current_distribution_type` says: `constant_current_density`, one, whose `current_distribution` is each
        element's `area`, so that its amplitude is the current density, or `eig`, the conductor's `n_dof` slowest decaying eigenmodes,
        `eig_01`, `eig_02`, ..., each with its `current_distribution`, 100 times the eigenvector normalised to a length of 1, and its
        `time_constant` [second]; `.../regularisations`, from the fit settings, shape = (n_regularisations, n_dof); and
        `.../regularisations_weight`, from the fit settings [dimensionless].

        `pf_passive_current(i1)` is where the data dictionary puts the current in a passive loop. GSFit does not measure that current: it
        fits it, as an amplitude for each of the loop's degrees of freedom, and the regularisations are what constrain the fit. So `measured`,
        `source` and `time_measurement` are left unset.

        `greens_with_pf_passive` reads the degrees of freedom, and the solver the regularisations, from the equilibrium IDS, so this has to be
        called before both. The solver fits the amplitudes with the Greens tables, `greens_tables/grid/pf_passive(i1)/dof(i2)`, so the loops and
        their degrees of freedom have to be in the same order as those, which the solver checks by name.

        Fit settings with no passive conductors are allowed: the equilibrium then has none. Any passive conductors already on a time-slice
        are replaced, so calling this twice does not repeat them.

        :param pf_passive: The passive conductors, holding their filaments
        :raises ValueError: When a passive conductor in the fit settings is not in the pf_passive IDS, or is in the fit settings twice, when
            one of its elements has no positive `area`, or when its fit settings cannot make its degrees of freedom: an `n_dof` which is not
            1 for `constant_current_density`, or is more than its elements for `eig`, or regularisations which are not of shape
            (n_regularisations, n_dof), with one weight each
        """
        ...
    def copy_from_pulse_schedule_ids(self, pulse_schedule: PulseSchedule) -> None:
        """
        Copy the gap definitions from the pulse_schedule IDS onto every time-slice of the equilibrium IDS.

        Fills `equilibrium/time_slice(itime)/boundary/gap(i)/name`, `.../description`, `.../r`, `.../z` and `.../angle` from
        `pulse_schedule/position_control/gap(i)`, in the same order, on every time-slice. `.../value` is left unset, for the
        equilibrium post-processor to calculate. A time-slice which does not converge keeps the definitions, with its values unset.

        The pulse schedule is optional: without this call, or with a pulse schedule which has no gaps, the equilibrium has no gaps.
        Any gaps already on a time-slice are replaced, so calling this twice does not repeat them.

        :param pulse_schedule: The pulse schedule, holding the gap definitions
        """
        ...
    def copy_from_magnetics_ids(self, magnetics: Magnetics) -> None:
        """
        Copy the magnetic sensors' measurements from the magnetics IDS onto every time-slice of the equilibrium IDS.

        Only the sensors in the fit settings, the equilibrium settings' `constraints`, are copied, and each of them has to be in the
        magnetics IDS. Each measurement is on the experimental timebase, so it is interpolated onto the time-slice's time. On every
        time-slice, for every sensor in the fit settings, in the order of the magnetics IDS: `magnetics/b_field_pol_probe(i1)/field` fills
        `equilibrium/time_slice(itime)/constraints/b_field_pol_probe(i1)` [tesla]; `magnetics/flux_loop(i1)/flux` fills
        `.../constraints/flux_loop(i1)` [weber]; `magnetics/rogowski_coil(i1)/current` fills `.../constraints/rogowski_coil(i1)`, a custom
        key, as the data dictionary has no constraint for Rogowski coils [ampere]; `magnetics/diamagnetic_flux(0)/data` fills
        `.../constraints/diamagnetic_flux` [weber]; and `magnetics/ip(0)/data` fills `.../constraints/ip` [ampere].

        The constraints at points which can move, custom keys of both IDSs, are copied in the same way, each point interpolated onto the
        time-slice's time, and NaN, which is not fitted, where it is not known there, e.g. outside its timebase:
        `magnetics/isoflux(i1)/position_1` and `/position_2` fill `.../constraints/isoflux(i1)/position_1` and `/position_2`;
        `magnetics/isoflux_boundary(i1)/position` fills `.../constraints/isoflux_boundary(i1)/position`; and
        `magnetics/stationary_point(i1)/position` fills `.../constraints/stationary_point(i1)/position` [metre]. Their `measured` is zero:
        the flux difference between the two points, the flux at the point less that at the boundary, and both derivatives of the flux.

        Each constraint gets `name`, a custom key: the sensor's `name`, or the `method_name` for `ip`, while `diamagnetic_flux` has none, as
        in the data dictionary, because there is only one; `measured`, the measurement interpolated onto the time-slice's time; `source`,
        the path the measurement was read from; and `time_measurement`, the time-slice's own time [second]. GSFit accepts a single
        diamagnetic flux, so the signals of several diamagnetic loops are averaged into one before it is added to the magnetics IDS; it is
        left unset without fit settings for it. `ip`, which has no fit settings, takes the magnetics IDS's first, generally recommended,
        method, and is left unset when it has none. Each sensor keeps the `weight` and `expected_value`, a custom key, `Equilibrium` was made
        with, from its fit settings, found by its name, or for the diamagnetic flux, from its own.

        Any magnetic constraints already on a time-slice are replaced, so calling this twice does not repeat them.

        :param magnetics: The magnetic sensors, holding their measurements
        :raises ValueError: When a sensor in the fit settings is not in the magnetics IDS, when the magnetics IDS holds more than one
            diamagnetic flux, or when a measurement cannot be interpolated onto the reconstruction times, e.g. because its timebase does
            not cover them. The diamagnetic flux is the exception: a time-slice it cannot be interpolated onto gets a `weight` of 0 instead
        """
        ...
    def copy_from_core_profiles_ids(self, core_profiles: CoreProfiles) -> None:
        """
        Make the pressure constraints from the core_profiles IDS, on every time-slice of the equilibrium IDS.

        When the fit settings, the equilibrium settings' `constraints/pressure`, include pressure, there is one constraint for each point
        of `core_profiles/profiles_2d(itime)`, which must be on the same irregular (R, Z) grid, `irregular_rz_na`, at every time.
        `core_profiles/profiles_2d(itime)/pressure_thermal` is on the experimental timebase, so it is interpolated onto the time-slice's
        time. `equilibrium/time_slice(itime)/constraints/pressure(i1)` gets, for point i1: `name`, a custom key, `sensor_name_prefix`
        followed by the point's two-digit number, from 1, e.g. `PRESSURE01`; `measured`, the interpolated thermal pressure, NaN when it
        cannot be interpolated, which the solver does not fit [pascal]; `position/r` and `position/z`, the point [metre]; `source`, the
        path it was read from; `time_measurement`, the time-slice's own time [second]; `weight`, the fit settings' `weight`; and
        `expected_value`, a custom key, the fit settings' `expected_value`, so that the least-squares weight is `weight / expected_value`
        [pascal].

        When the fit settings do not include pressure, there are no pressure constraints. Any pressure constraints already on a time-slice
        are replaced, so calling this twice does not repeat them.

        :param core_profiles: The kinetic profiles, holding `core_profiles/profiles_2d`
        :raises ValueError: When the fit settings include pressure but have no `sensor_name_prefix`, `weight` or `expected_value`, or when
            `core_profiles/profiles_2d` has fewer than two times, is not on one `irregular_rz_na` grid at every time, or has a
            `pressure_thermal` without one value per point
        """
        ...
    def greens_with_pf_active(self, pf_active: PfActive) -> None:
        """
        Calculate the Greens tables between the grid and each PF coil, `greens_tables/grid/pf_active(i1)`, one for each coil in
        `pf_active/coil(i1)`, in the same order.

        :param pf_active: The PF coils, holding their filaments
        """
        ...
    def greens_with_pf_passive(self, pf_passive: PfPassive) -> None:
        """
        Calculate the Greens tables between the grid and each passive degree of freedom, `greens_tables/grid/pf_passive(i1)/dof(i2)`, in the
        order of `time_slice(itime)/constraints/pf_passive_current(i1)`, and then of its `dof(i2)`, which is the order of
        `time_slice(itime)/passive_dof_values`. The degrees of freedom are the equilibrium's, which `copy_from_pf_passive_ids` makes, so it
        has to be called first, with the same `pf_passive`.

        :param pf_passive: The passive conductors, holding their filaments
        :raises ValueError: When the equilibrium's passive conductors are not the ones in the fit settings, as `copy_from_pf_passive_ids`
            has not been called, or when one of them is not in `pf_passive`
        """
        ...
    def greens_with_magnetics(self, magnetics: Magnetics, pf_active: PfActive, pf_passive: PfPassive) -> None:
        """
        Calculate the magnetic sensors' Greens tables: what each sensor measures, per ampere in each current source.

        Fills, for every sensor in the fit settings, the equilibrium settings' `constraints`, which are the sensors `copy_from_magnetics_ids`
        copies, with the same name and in the order of the magnetics IDS, `greens_tables/b_field_pol_probe(i1)`, `greens_tables/flux_loop(i1)` and
        `greens_tables/rogowski_coil(i1)`, each with the sensor's tables with: `plasma`, the
        plasma current in each grid cell, shape = (n_z, n_r); `d_plasma_d_z`, the same differentiated with respect to the sensor's
        height, shape = (n_z, n_r); `pf_active`, each PF coil, in the order of `greens_tables/grid/pf_active`; and `pf_passive`, each passive
        degree of freedom, in the order of `greens_tables/grid/pf_passive(:)/dof(:)`.

        `greens_tables/grid/pf_active` and `greens_tables/grid/pf_passive` name the PF coils and the passive degrees of freedom, so
        `Equilibrium.greens_with_pf_active` and `Equilibrium.greens_with_pf_passive` have to be called first, with the same `pf_active` and
        `pf_passive`. Any sensors' Greens tables already in `greens_tables` are replaced, so calling this twice does not repeat them.

        The constraints at points which can move in the fit settings get the same tables, but one for each time-slice, as their points
        can move: `greens_tables/isoflux(i1)`, `greens_tables/isoflux_boundary(i1)` and `greens_tables/stationary_point(i1)`, the last with
        the tables of both derivatives of the flux, `d_psi_d_r` and `d_psi_d_z`. Their points are interpolated as `copy_from_magnetics_ids`
        interpolates them, so the two can be called in either order.

        :param magnetics: The magnetic sensors, holding where they are
        :param pf_active: The PF coils, holding their filaments
        :param pf_passive: The passive conductors, holding their filaments
        :raises ValueError: When a sensor in the fit settings is not in the magnetics IDS, when the PF coils, or the equilibrium's passive
            degrees of freedom, are not the ones `greens_tables/grid/pf_active` and `greens_tables/grid/pf_passive` were calculated with, in the same order,
            when a passive conductor the equilibrium fits is not in `pf_passive`, or when the points of a constraint at points which can move
            cannot be interpolated onto the time-slices' times
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.equilibrium_paths`, straight out of the equilibrium IDS.

        This is how the data is read, e.g. `equilibrium.get(equilibrium_paths.time_slice[0].global_quantities.ip)`: a path holds
        no data, so the IDS is only borrowed for the read, never copied. The shape of the result follows the shape of the
        index; see `gsfit_rs.imas.Equilibrium.get`.
        """
        ...
    @property
    def equilibrium_ids(self) -> EquilibriumIds:
        """A copy of the whole equilibrium IDS, read with `gsfit_rs.imas.equilibrium_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...
    @property
    def greens_tables(self) -> GreensTables:
        """A copy of the Greens tables, read with `gsfit_rs.imas.greens_tables_paths`, e.g.
        `equilibrium.greens_tables.get(greens_tables_paths.grid.pf_active[:].psi)`.

        This copies every table on every access, so take it once and read it with `get` as often as needed. It is a
        detached snapshot: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class CoreProfiles:
    """The kinetic profiles, stored as an IMAS `core_profiles` IDS.

    Two parts are filled, each by its own setter.

    `profiles_2d(itime)`, by `set_profiles_2d_irregular_rz`, one per measurement time, at a list of points in the poloidal cross-section:
    the irregular (R, Z) grid `irregular_rz_na`, whose points are `grid/dim1(i)`, `grid/dim2(i)`, and whose fields are on `values(i, 0)`.
    Each holds the electron pressure, `electrons/pressure`, a custom key, as the data dictionary has no electrons in `profiles_2d`, and the
    thermal pressure, `pressure_thermal`. A value which was not measured, or is not physical, is NaN.
    `Equilibrium.copy_from_core_profiles_ids` makes the pressure constraints from `pressure_thermal`, one per point.

    `profiles_1d(itime)`, the kinetic profiles, one per time, on the radial grid `grid/rho_pol_norm = sqrt(psi_norm)`, which, with `time`,
    every quantity of a `profiles_1d(itime)` shares. Each part has its own setter, and they can be set in any order, as long as each is at
    the same times and on the same grid:
    * `set_profiles_1d_electrons`: `electrons/density_thermal` and `electrons/temperature`
    * `set_profiles_1d_zeff`: `zeff`
    * `set_profiles_1d_thermal_ions`: one species' `ion(i1)/density_thermal` and `ion(i1)/temperature`
    * `set_profiles_1d_fast_ions`: one species' fast ions, e.g. the beam ions from ASCOT5: `ion(i1)/density_fast`, and their pressures
      along and across the field, `ion(i1)/pressure_fast_parallel = sum(m v_par ** 2)` and
      `ion(i1)/pressure_fast_perpendicular = sum(m v_perp ** 2 / 2)`
    * `set_profiles_1d_e_field_parallel`: `e_field/parallel = <E . B> / B0`, and the `vacuum_toroidal_field` whose `b0` is `B0`
    A species' thermal and fast ions share its `ion(i1)`, which is found by its `name`; a species which is not there yet is added after the
    others. The bootstrap current takes `ion(0)` as the main ion, so set the main ion first.

    `calculate_plasma_current_components` reads `profiles_1d`: the bootstrap current the electrons, `zeff` and the thermal ions; the ohmic
    current the electrons, `zeff` and `e_field/parallel`; and the shielding of the beam-driven current the electrons and `zeff`. The fast
    ions are read by nothing yet.

    Read it back through `get` and a path from `gsfit_rs.imas.core_profiles_paths`.

    The kinetic profiles are constructed with the settings they are built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `core_profiles/code/parameters`. Rust also reads them into structs
    generated from the XML schema `core_profiles.xsd`, kept with the class rather than in the IDS, as the core_profiles IDS can also be made
    by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> CoreProfiles:
        """
        Construct the kinetic profiles from their settings, ready for `set_profiles_2d_irregular_rz`.

        :param parameters_input: The settings the kinetic profiles are built from, e.g. which run of which database they are read from,
            whose XML is `core_profiles.xml` with each file it links to written in its link's place, and whose schema is the XML schema it
            follows, as `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is written to `core_profiles/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, or when the XML cannot be read into the structs, as
            the XML schema is not the one they were generated from
        """
        ...
    def set_profiles_2d_irregular_rz(
        self,
        time: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        electrons_pressure: npt.NDArray[np.float64],
        pressure_thermal: npt.NDArray[np.float64],
    ) -> None:
        """
        Set `core_profiles/profiles_2d(itime)`, one per measurement time, at a list of points: the irregular (R, Z) grid
        `irregular_rz_na`. Any `profiles_2d` already set is replaced.

        :param time: the measurement times, shape = [n_time] [second]
        :param r: the major radius of each point, `grid/dim1`, shape = [n_point] [metre]
        :param z: the height of each point, `grid/dim2`, shape = [n_point] [metre]
        :param electrons_pressure: the electron pressure, `electrons/pressure`, shape = [n_time, n_point] [pascal]
        :param pressure_thermal: the thermal pressure, electrons and ions, `pressure_thermal`, shape = [n_time, n_point] [pascal]
        :raises ValueError: When there is no time or no point, a time is not finite or the times do not increase, a point is not at a
            positive finite major radius and a finite height, or a pressure does not have one value per time and point

        Store the **experimental** times, not the reconstruction times: `Equilibrium.copy_from_core_profiles_ids` interpolates onto them
        itself. A value which was not measured, or is not physical, is NaN.
        """
        ...
    def set_profiles_1d_fast_ions(
        self,
        time: npt.NDArray[np.float64],
        rho_pol_norm: npt.NDArray[np.float64],
        ion_name: str,
        ion_element_a: float,
        ion_element_z_n: int,
        ion_z_ion: float,
        density_fast: npt.NDArray[np.float64],
        pressure_fast_parallel: npt.NDArray[np.float64],
        pressure_fast_perpendicular: npt.NDArray[np.float64],
    ) -> None:
        """
        Set the fast ions of one species on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
        `ion(i1)/density_fast`, `ion(i1)/pressure_fast_parallel` and `ion(i1)/pressure_fast_perpendicular`.

        The fast ions of a species share its `ion(i1)` with its thermal ions, so `ion(i1)` is found by its `name`, and a species which is not
        there yet is added after the others. Its fast ions already set are replaced.

        Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
        `rho_pol_norm`; when it is set, e.g. by the fast ions of another species, the fast ions must be at its times and on its
        `grid/rho_pol_norm`.

        :param time: the times, shape = [n_time] [second]
        :param rho_pol_norm: the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)`, shape = [n_rho_pol_norm] [dimensionless]
        :param ion_name: the species, `ion(i1)/name`, e.g. "D"
        :param ion_element_a: the mass of its nucleus, `ion(i1)/element(1)/a` [atomic_mass_unit]
        :param ion_element_z_n: the charge of its nucleus, `ion(i1)/element(1)/z_n` [dimensionless]
        :param ion_z_ion: its charge, `ion(i1)/z_ion` [dimensionless]
        :param density_fast: the density of its fast ions, `ion(i1)/density_fast`, shape = [n_time, n_rho_pol_norm] [1 / metre ** 3]
        :param pressure_fast_parallel: their pressure along the field, `sum(m v_par ** 2)`, `ion(i1)/pressure_fast_parallel`,
            shape = [n_time, n_rho_pol_norm] [pascal]
        :param pressure_fast_perpendicular: their pressure across the field, `sum(m v_perp ** 2 / 2)`, `ion(i1)/pressure_fast_perpendicular`,
            shape = [n_time, n_rho_pol_norm] [pascal]
        :raises ValueError: When there is no time or no radial point, a time is not finite or the times do not increase, a radial point is
            not finite and at least 0 or the radial points do not increase, the species is not named or not physical, a profile does not
            have one value per time and radial point, `profiles_1d` is set at other times or on another `grid/rho_pol_norm`, or it holds a
            species of the same name which is another species. Nothing is changed then
        """
        ...
    def set_profiles_1d_electrons(
        self,
        time: npt.NDArray[np.float64],
        rho_pol_norm: npt.NDArray[np.float64],
        density_thermal: npt.NDArray[np.float64],
        temperature: npt.NDArray[np.float64],
    ) -> None:
        """
        Set the thermal electrons on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
        `electrons/density_thermal` and `electrons/temperature`. Any already set are replaced.

        Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
        `rho_pol_norm`; when it is set, the electrons must be at its times and on its `grid/rho_pol_norm`.

        :param time: the times, shape = [n_time] [second]
        :param rho_pol_norm: the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)`, shape = [n_rho_pol_norm] [dimensionless]
        :param density_thermal: the electron density, `electrons/density_thermal`, shape = [n_time, n_rho_pol_norm] [1 / metre ** 3]
        :param temperature: the electron temperature, `electrons/temperature`, shape = [n_time, n_rho_pol_norm] [electron_volt]
        :raises ValueError: When there is no time or no radial point, a time is not finite or the times do not increase, a radial point is
            not finite and at least 0 or the radial points do not increase, a profile does not have one value per time and radial point, or
            `profiles_1d` is set at other times or on another `grid/rho_pol_norm`. Nothing is changed then
        """
        ...
    def set_profiles_1d_zeff(
        self,
        time: npt.NDArray[np.float64],
        rho_pol_norm: npt.NDArray[np.float64],
        zeff: npt.NDArray[np.float64],
    ) -> None:
        """
        Set the effective charge on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`: `zeff`. Any
        already set is replaced.

        Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
        `rho_pol_norm`; when it is set, `zeff` must be at its times and on its `grid/rho_pol_norm`.

        :param time: the times, shape = [n_time] [second]
        :param rho_pol_norm: the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)`, shape = [n_rho_pol_norm] [dimensionless]
        :param zeff: the effective charge, `zeff`, shape = [n_time, n_rho_pol_norm] [dimensionless]
        :raises ValueError: When there is no time or no radial point, a time is not finite or the times do not increase, a radial point is
            not finite and at least 0 or the radial points do not increase, `zeff` does not have one value per time and radial point, or is
            not finite and at least 1 somewhere (NaN is allowed), or `profiles_1d` is set at other times or on another `grid/rho_pol_norm`.
            Nothing is changed then
        """
        ...
    def set_profiles_1d_thermal_ions(
        self,
        time: npt.NDArray[np.float64],
        rho_pol_norm: npt.NDArray[np.float64],
        ion_name: str,
        ion_element_a: float,
        ion_element_z_n: int,
        ion_z_ion: float,
        density_thermal: npt.NDArray[np.float64],
        temperature: npt.NDArray[np.float64],
    ) -> None:
        """
        Set the thermal ions of one species on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
        `ion(i1)/density_thermal` and `ion(i1)/temperature`.

        The thermal ions of a species share its `ion(i1)` with its fast ions, so `ion(i1)` is found by its `name`, and a species which is not
        there yet is added after the others. Its thermal ions already set are replaced. The bootstrap current takes `ion(0)` as the main
        ion, so set the main ion first.

        Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
        `rho_pol_norm`; when it is set, the ions must be at its times and on its `grid/rho_pol_norm`.

        :param time: the times, shape = [n_time] [second]
        :param rho_pol_norm: the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)`, shape = [n_rho_pol_norm] [dimensionless]
        :param ion_name: the species, `ion(i1)/name`, e.g. "D"
        :param ion_element_a: the mass of its nucleus, `ion(i1)/element(1)/a` [atomic_mass_unit]
        :param ion_element_z_n: the charge of its nucleus, `ion(i1)/element(1)/z_n` [dimensionless]
        :param ion_z_ion: its charge, `ion(i1)/z_ion` [dimensionless]
        :param density_thermal: the density of its thermal ions, `ion(i1)/density_thermal`, shape = [n_time, n_rho_pol_norm] [1 / metre ** 3]
        :param temperature: their temperature, `ion(i1)/temperature`, shape = [n_time, n_rho_pol_norm] [electron_volt]
        :raises ValueError: When there is no time or no radial point, a time is not finite or the times do not increase, a radial point is
            not finite and at least 0 or the radial points do not increase, the species is not named or not physical, a profile does not
            have one value per time and radial point, `profiles_1d` is set at other times or on another `grid/rho_pol_norm`, or it holds a
            species of the same name which is another species. Nothing is changed then
        """
        ...
    def set_profiles_1d_e_field_parallel(
        self,
        time: npt.NDArray[np.float64],
        rho_pol_norm: npt.NDArray[np.float64],
        e_field_parallel: npt.NDArray[np.float64],
        vacuum_toroidal_field_r0: float,
        vacuum_toroidal_field_b0: npt.NDArray[np.float64],
    ) -> None:
        """
        Set the parallel electric field on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
        `e_field/parallel = <E . B> / B0`, and `core_profiles/vacuum_toroidal_field`, whose `b0` is `B0`. Any already set are replaced.

        Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
        `rho_pol_norm`; when it is set, the field must be at its times and on its `grid/rho_pol_norm`. `vacuum_toroidal_field/b0` is on
        the same times.

        :param time: the times, shape = [n_time] [second]
        :param rho_pol_norm: the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)`, shape = [n_rho_pol_norm] [dimensionless]
        :param e_field_parallel: the flux-surface averaged parallel electric field, `<E . B> / B0`, `e_field/parallel`,
            shape = [n_time, n_rho_pol_norm] [volt / metre]
        :param vacuum_toroidal_field_r0: the major radius at which `vacuum_toroidal_field_b0` is given, `vacuum_toroidal_field/r0` [metre]
        :param vacuum_toroidal_field_b0: the vacuum toroidal field at `vacuum_toroidal_field_r0`, `B0`, which `e_field_parallel` is
            normalised by, `vacuum_toroidal_field/b0`, shape = [n_time] [tesla]
        :raises ValueError: When there is no time or no radial point, a time is not finite or the times do not increase, a radial point is
            not finite and at least 0 or the radial points do not increase, `e_field_parallel` does not have one value per time and radial
            point, the vacuum toroidal field is not a positive finite radius and one finite, non-zero field per time, or `profiles_1d` is set
            at other times or on another `grid/rho_pol_norm`. Nothing is changed then
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.core_profiles_paths`, straight out of the core_profiles IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.CoreProfiles.get`.
        """
        ...
    @property
    def core_profiles_ids(self) -> CoreProfilesIds:
        """A copy of the whole core_profiles IDS, read with `gsfit_rs.imas.core_profiles_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class CoreSources:
    """The sources of the plasma, stored as an IMAS `core_sources` IDS.

    The neutral beams' source is set from Python, by `set_source_nbi_profiles_1d`: `source(i1)`, with `identifier/name` "NBI", holding the
    beam's species, `species/type` an ion and `species/ion`, whose charge, `Z_b`, the electrons' shielding needs, and `profiles_1d(itime)`,
    one per time, on the radial grid `grid/rho_pol_norm = sqrt(psi_norm)`, e.g. from ASCOT5. Each holds the fast ions' parallel current,
    `j_parallel = <J_fast . B> / B0`, with `B0` = `core_sources/vacuum_toroidal_field/b0`.

    `j_parallel` is the current of the fast ions alone, before the electrons shield it. The data dictionary's `j_parallel` is the current a
    source drives, which for the neutral beams is after the shielding, `1 - (Z_b / Z_eff) (1 - L31)`. But GSFit applies the shielding
    itself, with the same trapped fraction and `L31` as the bootstrap current, so it keeps the current the fast-ion code gives:
    `calculate_plasma_current_components` makes the beam-driven current from it, and does not write that back here. It does write the
    bootstrap and ohmic currents here, as the sources "bootstrap_current" and "ohmic", on the equilibrium's flux surfaces and times.

    Every source shares `core_sources/vacuum_toroidal_field`, which normalises `j_parallel`, and the times of its `profiles_1d`.

    Read it back through `get` and a path from `gsfit_rs.imas.core_sources_paths`.

    The sources are constructed with the settings they are built from. Rust validates them against the XML schema in `parameters_input`,
    and only their XML is kept: it is written to `core_sources/code/parameters`. Rust also reads them into structs generated from the XML
    schema `core_sources.xsd`, kept with the class rather than in the IDS, as the core_sources IDS can also be made by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> CoreSources:
        """
        Construct the sources from their settings, ready for `set_source_nbi_profiles_1d`.

        :param parameters_input: The settings the sources are built from, whose XML is `core_sources.xml` with each file it links to
            written in its link's place, and whose schema is the XML schema it follows, as `serialise_python_dataclass_to_imas_parameters_input`
            writes them. Their XML is written to `core_sources/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, or when the XML cannot be read into the structs, as
            the XML schema is not the one they were generated from
        """
        ...
    def set_source_nbi_profiles_1d(
        self,
        time: npt.NDArray[np.float64],
        rho_pol_norm: npt.NDArray[np.float64],
        ion_name: str,
        ion_element_a: float,
        ion_element_z_n: int,
        ion_z_ion: float,
        j_parallel: npt.NDArray[np.float64],
        vacuum_toroidal_field_r0: float,
        vacuum_toroidal_field_b0: npt.NDArray[np.float64],
    ) -> None:
        """
        Set the neutral beams' source, `core_sources/source(i1)` with `identifier/name` "NBI": the beam's species, `species`, and its
        `profiles_1d(itime)/j_parallel`, one per time, on the radial grid `grid/rho_pol_norm`, and `core_sources/vacuum_toroidal_field`, which
        normalises it. Any neutral beams' source already set is replaced; otherwise it is added after the other sources.

        Every source shares the times and `vacuum_toroidal_field` of `core_sources`. To have the bootstrap and ohmic currents written beside
        it by `calculate_plasma_current_components`, set it at the equilibrium's times, with NaN at a time with no fast ions, and with the
        equilibrium's `vacuum_toroidal_field`.

        :param time: the times, shape = [n_time] [second]
        :param rho_pol_norm: the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)`, shape = [n_rho_pol_norm] [dimensionless]
        :param ion_name: the beam's species, `species/ion/name`, e.g. "D"
        :param ion_element_a: the mass of its nucleus, `species/ion/element(1)/a` [atomic_mass_unit]
        :param ion_element_z_n: the charge of its nucleus, `species/ion/element(1)/z_n` [dimensionless]
        :param ion_z_ion: its charge, `Z_b`, `species/ion/z_ion` [dimensionless]
        :param j_parallel: the fast ions' parallel current, `<J_fast . B> / B0`, before the electrons shield it, `profiles_1d(itime)/j_parallel`,
            shape = [n_time, n_rho_pol_norm] [ampere / metre ** 2]
        :param vacuum_toroidal_field_r0: the major radius at which `vacuum_toroidal_field_b0` is given, `vacuum_toroidal_field/r0` [metre]
        :param vacuum_toroidal_field_b0: the vacuum toroidal field at `vacuum_toroidal_field_r0`, `B0`, which `j_parallel` is normalised by,
            `vacuum_toroidal_field/b0`, shape = [n_time] [tesla]
        :raises ValueError: When there is no time or no radial point, a time is not finite or the times do not increase, a radial point is
            not finite and at least 0 or the radial points do not increase, the species is not named or not physical, `j_parallel` does not
            have one value per time and radial point, the vacuum toroidal field is not a positive finite radius and one finite, non-zero
            field per time, or another source is at other times or normalised by another vacuum toroidal field. Nothing is changed then
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.core_sources_paths`, straight out of the core_sources IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.CoreSources.get`.
        """
        ...
    @property
    def core_sources_ids(self) -> CoreSourcesIds:
        """A copy of the whole core_sources IDS, read with `gsfit_rs.imas.core_sources_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

def calculate_plasma_current_components(
    equilibrium: Equilibrium,
    core_profiles: CoreProfiles,
    core_sources: CoreSources,
) -> PlasmaCurrentComponents:
    """
    Calculate the components of the toroidal plasma current on every time-slice of the equilibrium, from the IDSs.

    In this order:
    * the diamagnetic and Pfirsch-Schlüter currents, from the equilibrium's `p'` and flux surfaces
    * the bootstrap current, Redl's, from the kinetic profiles, written to `core_sources` as the source "bootstrap_current"
    * the ohmic current, the neoclassical conductivity times the kinetic profiles' `e_field/parallel`, written to `core_sources` as the
      source "ohmic", with the conductivity
    * the beam-driven current, only when `core_sources` has the neutral beams' source ("NBI"): its fast ions' current, shielded by the
      electrons, `(1 - (Z_b / Z_eff) (1 - L31)) <J_fast . B>`, which is not written to `core_sources`

    and the electron collisionality, `nu_e*`, which the bootstrap current and the neoclassical conductivity take, is written to the kinetic
    profiles, as `core_profiles/profiles_1d(:)/electrons/collisionality_norm`, at the points of their own grid; it is infinite on the
    magnetic axis, and NaN where the equilibrium's `q` is NaN, as on the boundary.

    A time-slice the solver could not use, or without the kinetic profiles, or the neutral beams' source, which a component needs at its
    time, has NaN in that component. The sources written are on the equilibrium's flux surfaces and times, and every source of
    `core_sources` shares its times and `vacuum_toroidal_field`, so any source already in `core_sources`, such as the neutral beams', must be
    at the equilibrium's times, with the equilibrium's `vacuum_toroidal_field`.

    :param equilibrium: the post-processed equilibrium; only its IDS is read
    :param core_profiles: the kinetic profiles, on `core_profiles/profiles_1d(:)`, see `CoreProfiles`, which the electron collisionality is
        written into
    :param core_sources: the plasma's sources, which the bootstrap and ohmic currents are written into
    :return: the components
    :raises ValueError: When a component or the electron collisionality cannot be calculated, with the reason; `core_profiles` and
        `core_sources` are not changed then
    """
    ...

class PlasmaCurrentComponents:
    """The toroidal plasma current, split into its components, on the flux surfaces of each equilibrium time-slice, as
    `calculate_plasma_current_components` makes it.

    Every quantity is an array of shape [n_time, n_psi_norm], one row per equilibrium time-slice, on its flux surfaces, `psi_norm`. A
    component which was not calculated, e.g. the beam-driven current without the neutral beams' source, is NaN.
    """

    @property
    def time(self) -> npt.NDArray[np.float64]:
        """The equilibrium time-slices' times, shape = [n_time] [second]"""
        ...
    @property
    def psi(self) -> npt.NDArray[np.float64]:
        """The poloidal flux of each flux surface, the equilibrium's `profiles_1d/psi`, shape = [n_time, n_psi_norm] [weber]"""
        ...
    @property
    def psi_norm(self) -> npt.NDArray[np.float64]:
        """The normalised poloidal flux of each flux surface, the equilibrium's `profiles_1d/psi_norm`, shape = [n_time, n_psi_norm]
        [dimensionless]"""
        ...
    @property
    def rho_tor_norm(self) -> npt.NDArray[np.float64]:
        """The normalised toroidal flux coordinate of each flux surface, the equilibrium's `profiles_1d/rho_tor_norm`,
        shape = [n_time, n_psi_norm] [dimensionless]"""
        ...
    @property
    def area(self) -> npt.NDArray[np.float64]:
        """The poloidal cross-section inside each flux surface, the equilibrium's `profiles_1d/area`, shape = [n_time, n_psi_norm]
        [metre ** 2]"""
        ...
    @property
    def beam_driven(self) -> PlasmaCurrentComponent:
        """The beam-driven current: the neutral beams' fast ions' current, shielded by the electrons"""
        ...
    @property
    def bootstrap(self) -> PlasmaCurrentComponent:
        """The bootstrap current"""
        ...
    @property
    def diamagnetic(self) -> PlasmaCurrentComponent:
        """The diamagnetic current"""
        ...
    @property
    def ohmic(self) -> PlasmaCurrentComponent:
        """The ohmic current"""
        ...
    @property
    def pfirsch_schluter(self) -> PlasmaCurrentComponent:
        """The Pfirsch-Schlüter current"""
        ...

class PlasmaCurrentComponent:
    """One component of the toroidal plasma current, on the flux surfaces of each equilibrium time-slice."""

    @property
    def j_phi(self) -> npt.NDArray[np.float64]:
        """The toroidal current density on each flux surface, `<j_phi / R> / <1 / R>`, as the data dictionary defines
        `core_profiles/profiles_1d/j_phi`, so that its integral over the cross-section is the toroidal current,
        shape = [n_time, n_psi_norm] [ampere / metre ** 2]"""
        ...
    @property
    def current_phi_inside(self) -> npt.NDArray[np.float64]:
        """The toroidal current inside each flux surface, the integral of `j_phi` over the cross-section, by the trapezium rule in
        `area`, shape = [n_time, n_psi_norm] [ampere]"""
        ...

class Tf:
    """The machine's toroidal field, stored as an IMAS `tf` IDS.

    Two nodes are filled: `tf/r0`, the reference major radius, from the settings, and
    `tf/b_field_phi_vacuum_r`, the vacuum field times major radius on the experimental
    timebase.

    `b_field_phi_vacuum_r` is the vacuum poloidal-current function
    `f_vac = R0 * B_phi0 = mu_0 * i_rod / (2 * pi)`, so the rod current is not stored
    separately: `solve_grad_shafranov` recovers it as `i_rod = 2 * pi * f_vac / mu_0`.

    It is signed: positive means counter-clockwise viewed from above.

    Read it back through `tf_ids` and a path from `gsfit_rs.imas.tf_paths`.

    The toroidal field is constructed with the settings it is built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `tf/code/parameters`. Rust also reads them into structs generated
    from the XML schema `tf.xsd`, kept with the class rather than in the IDS, as the tf IDS can also be made by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> Tf:
        """
        Construct the toroidal field from its settings, ready for `set_b_field_phi_vacuum_r`.

        Their `vacuum_toroidal_field_reference_radius` sets `tf/r0`, the machine's official reference radius, typically the middle of
        the vessel at the equatorial midplane. `Equilibrium.copy_from_tf_ids` copies it onto `equilibrium/vacuum_toroidal_field/r0`, so
        that the two IDSs cannot disagree.

        :param parameters_input: The settings the toroidal field is built from, e.g. the reference major radius, and which
            run of which database the field is read from, whose XML is `tf.xml` with each file it links to written in its link's place, and
            whose schema is the XML schema it follows, as `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is
            written to `tf/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, e.g. a reference radius which is not a positive
            number, or when the XML cannot be read into the structs, as the XML schema is not the one they were generated from
        """
        ...
    def set_b_field_phi_vacuum_r(
        self,
        time: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
    ) -> None:
        """
        Set `tf/b_field_phi_vacuum_r`, the vacuum field times major radius.

        :param time: the experimental timebase [second]
        :param data: vacuum toroidal field times major radius [tesla * metre]

        Store the **experimental** signal, not one interpolated onto the reconstruction
        times: `Equilibrium.copy_from_tf_ids` interpolates it itself.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.tf_paths`, straight out of the tf IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.Tf.get`.
        """
        ...
    @property
    def tf_ids(self) -> TfIds:
        """A copy of the whole tf IDS, read with `gsfit_rs.imas.tf_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Wall:
    """The machine's wall, stored as an IMAS `wall` IDS.

    Only the limiter is filled so far:
    `wall/description_2d(0)/limiter/unit(i)/outline/r` and `.../z`.

    `Equilibrium.copy_from_wall_ids` copies the units into the equilibrium IDS, as
    `equilibrium/limiter/unit(i)`, in the same order, and the solver reads them from there by
    position. So the order units are added in is part of the contract: `unit(0)` is the vacuum
    vessel contour, and the solver uses that one, and only that one, as the region the plasma
    is allowed to occupy. Every unit contributes candidate limit points.

    Read it back through `wall_ids` and a path from `gsfit_rs.imas.wall_paths`.

    The wall is constructed with the settings it is built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `wall/code/parameters`. Rust also reads them into structs generated
    from the XML schema `wall.xsd`, kept with the class rather than in the IDS, as the wall IDS can also be made by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> Wall:
        """
        Construct an empty wall, ready for `add_limiter_unit` to be called.

        :param parameters_input: The settings the wall is built from, e.g. which run of which database the limiter is read from, whose XML is
            `wall.xml` with each file it links to written in its link's place, and whose schema is the XML schema it follows, as
            `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is written to `wall/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, or when the XML cannot be read into the structs, as the
            XML schema is not the one they were generated from
        """
        ...
    def add_limiter_unit(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a limiter unit to `wall/description_2d(0)/limiter/unit`.

        :param name: short identifier for the unit, e.g. `"vacuum_vessel"`
        :param r: outline radial points [metre]
        :param z: outline vertical points [metre]

        The **first** unit added is the vacuum vessel contour.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.wall_paths`, straight out of the wall IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.Wall.get`.
        """
        ...
    @property
    def wall_ids(self) -> WallIds:
        """A copy of the whole wall IDS, read with `gsfit_rs.imas.wall_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class PulseSchedule:
    """The machine's pulse schedule, stored as an IMAS `pulse_schedule` IDS.

    Only the gap definitions are filled so far:
    `pulse_schedule/position_control/gap(i)/name`, `.../r`, `.../z` and `.../angle`.

    A gap is a reference point and a direction. `Equilibrium.copy_from_pulse_schedule_ids` copies the
    definitions onto every time-slice's `equilibrium/time_slice(itime)/boundary/gap`, and
    `solve_grad_shafranov` calculates each value: the distance from the reference point to the
    plasma boundary along that direction.

    The angle is stored in the equilibrium IDS's convention, clockwise from `grad(R)` in the usual
    plot with `R` to the right and `Z` upwards, so the gap points along
    `(cos(angle), -sin(angle))`. A database reader holding a counter-clockwise angle must convert it.

    Read it back through `pulse_schedule_ids` and a path from `gsfit_rs.imas.pulse_schedule_paths`.

    The pulse schedule is constructed with the settings it is built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `pulse_schedule/code/parameters`. Rust also reads them into structs generated
    from the XML schema `pulse_schedule.xsd`, kept with the class rather than in the IDS, as the pulse_schedule IDS can also be made by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> PulseSchedule:
        """
        Construct an empty pulse schedule, with no gaps, ready for `add_gap` to be called.

        :param parameters_input: The settings the pulse schedule is built from, e.g. which run of which database the gaps are read from, whose XML is
            `pulse_schedule.xml` with each file it links to written in its link's place, and whose schema is the XML schema it follows, as
            `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is written to `pulse_schedule/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, or when the XML cannot be read into the structs, as the
            XML schema is not the one they were generated from
        """
        ...
    def add_gap(
        self,
        name: str,
        r: float,
        z: float,
        angle: float,
    ) -> None:
        """
        Append a gap to `pulse_schedule/position_control/gap`.

        :param name: short identifier for the gap, unique within the pulse schedule, e.g. `"IMGAP"`
        :param r: major radius of the reference point [metre]
        :param z: height of the reference point [metre]
        :param angle: direction the gap is measured in, clockwise from `grad(R)` [radian]
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.pulse_schedule_paths`, straight out of the pulse_schedule IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.PulseSchedule.get`.
        """
        ...
    @property
    def pulse_schedule_ids(self) -> PulseScheduleIds:
        """A copy of the whole pulse_schedule IDS, read with `gsfit_rs.imas.pulse_schedule_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class PfActive:
    """The machine's active poloidal field (PF) coils, stored as an IMAS `pf_active` IDS.

    For each coil, `coil(i1)`: `name`; one `element(i2)` per filament, a rectangle
    (`geometry/rectangle/r`, `z`, `width` and `height`) of one turn (`turns_with_sign` = 1);
    `inductance`, the self-inductance calculated from the filaments; `current`, per turn, on the
    experimental timebase; and `voltage`, only when the coil's voltage is prescribed.

    A coil's current is prescribed, unless the coil has a voltage: `add_pf_coil` gives it its `current`, and
    `set_pf_voltage_prescribed` gives it its `voltage`, after which `solve_circuit_equations` calculates its current.

    `coil(i1)/current` is signed: positive means counter-clockwise viewed from above.

    The circuits and supplies are left empty: a power supply which feeds several coils in series is
    described by giving each coil its current.

    `Equilibrium.copy_from_pf_active_ids` copies each coil's current onto every time-slice of the equilibrium
    IDS, as `constraints/pf_current(i1)`, interpolated onto the reconstruction times.

    Read it back through `pf_active_ids` and a path from `gsfit_rs.imas.pf_active_paths`.

    The PF coils are constructed with the settings they are built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `pf_active/code/parameters`. Rust also reads them into structs generated
    from the XML schema `pf_active.xsd`, kept with the class rather than in the IDS, as the pf_active IDS can also be made by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> PfActive:
        """
        Construct empty PF coils, ready for `add_pf_coil` to be called.

        :param parameters_input: The settings the PF coils are built from, e.g. which run of which database the coils are read from, whose XML is
            `pf_active.xml` with each file it links to written in its link's place, and whose schema is the XML schema it follows, as
            `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is written to `pf_active/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, or when the XML cannot be read into the structs, as the
            XML schema is not the one they were generated from
        """
        ...
    def add_pf_coil(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        width: npt.NDArray[np.float64],
        height: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a coil to `pf_active/coil`, with its current prescribed.

        :param name: short identifier for the coil, unique within the PF coils, e.g. `"BVL"`
        :param r: major radius of each filament's centre [metre]
        :param z: height of each filament's centre [metre]
        :param width: radial full width of each filament [metre]
        :param height: vertical full height of each filament [metre]
        :param time: the experimental timebase [second]
        :param current: the current per turn [ampere]

        Each filament becomes one `element(i2)`: a rectangle of one turn, with a positive `r`, `width` and `height`.
        The coil's self-inductance, `coil(i1)/inductance`, is calculated from them.
        Store the **experimental** current, not one interpolated onto the reconstruction times:
        `Equilibrium.copy_from_pf_active_ids` interpolates it itself. A single time holds the current at that time only.
        """
        ...
    def set_pf_voltage_prescribed(
        self,
        name: str,
        time: npt.NDArray[np.float64],
        voltage: npt.NDArray[np.float64],
    ) -> None:
        """
        Prescribe a coil's voltage, rather than its current.

        :param name: the coil, as named in `add_pf_coil`
        :param time: the experimental timebase [second]
        :param voltage: voltage on the coil terminals [volt]

        Writes `coil(i1)/voltage`, which is what makes the coil's voltage prescribed. The coil's `current` is kept.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.pf_active_paths`, straight out of the pf_active IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.PfActive.get`.
        """
        ...
    @property
    def pf_active_ids(self) -> PfActiveIds:
        """A copy of the whole pf_active IDS, read with `gsfit_rs.imas.pf_active_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Magnetics:
    """The machine's magnetic sensors, stored as an IMAS `magnetics` IDS.

    For each poloidal field probe, `b_field_pol_probe(i1)`: `name`; `position/r` and `z`, the probe's centre; `poloidal_angle`,
    the direction of the probe's normal; and `field`, the measured field along the normal, on the experimental timebase.

    For each flux loop, `flux_loop(i1)`: `name`; `type` = 1, `poloidal_flux`, a loop all the way around the machine; `position(0)/r`
    and `z`, the loop's one point in the poloidal plane; and `flux`, the measured poloidal flux, on the experimental timebase.

    For each Rogowski coil, `rogowski_coil(i1)`: `name`; `position(i2)/r` and `z`, the coil's contour; `current`, the measured
    current, on the experimental timebase. The stretches of the contour which have no winding, the gaps, are kept with the class, as the
    data dictionary has no node for them.

    For the diamagnetic flux, `diamagnetic_flux(0)`: `method_name`, `data` and `time`. The data dictionary allows one entry for each
    method of calculating it, but GSFit accepts only one, as the equilibrium IDS has a single diamagnetic flux constraint, so the
    signals of several diamagnetic loops are averaged into one before it is added.

    For the plasma current, `ip(i1)`, one entry for each method of calculating it: `method_name`, `data` and `time`.

    For now, `b_field_pol_probe(i1)/poloidal_angle` keeps GSFit's present definition, COCOS 13, as MDSplus stores it: counter-clockwise
    from `grad(R)`, in the usual plot with `R` to the right and `Z` upwards, so the probe's normal is
    `cos(poloidal_angle) * grad(R) + sin(poloidal_angle) * grad(Z)`. It is the angle GSFit has always called `angle_pol`. This is not the
    data dictionary's COCOS 17, which measures it clockwise, within [0, 2 * pi]. `toroidal_angle` is left empty.

    `flux_loop(i1)/flux` is GSFit's `psi`, the total poloidal flux, with no factor of 2 * pi. `ip(i1)/data` and
    `diamagnetic_flux(i1)/data` are positive counter-clockwise viewed from above.

    Read it back through `magnetics_ids` and a path from `gsfit_rs.imas.magnetics_paths`.

    The magnetic sensors are constructed with the settings they are built from. Rust validates them against the XML schema in
    `parameters_input`, and only their XML is kept: it is written to `magnetics/code/parameters`. Rust also reads them into structs generated
    from the XML schema `magnetics.xsd`, kept with the class rather than in the IDS, as the magnetics IDS can also be made by another code.
    """

    def __new__(cls, parameters_input: ParametersInput) -> Magnetics:
        """
        Construct empty magnetic sensors, ready for `add_b_field_pol_probe`, `add_flux_loop`, `add_rogowski_coil`,
        `add_diamagnetic_flux` and `add_ip` to be called.

        :param parameters_input: The settings the magnetic sensors are built from, e.g. which run of which database the sensors are read from, whose XML is
            `magnetics.xml` with each file it links to written in its link's place, and whose schema is the XML schema it follows, as
            `serialise_python_dataclass_to_imas_parameters_input` writes them. Their XML is written to `magnetics/code/parameters`
        :raises ValueError: When there is no XML schema, when the XML does not follow it, or when the XML cannot be read into the structs, as the
            XML schema is not the one they were generated from
        """
        ...
    def add_b_field_pol_probe(
        self,
        name: str,
        r: float,
        z: float,
        poloidal_angle: float,
        time: npt.NDArray[np.float64],
        field: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a poloidal field probe to `magnetics/b_field_pol_probe`.

        :param name: short identifier for the probe, unique within the poloidal field probes, e.g. `"P101"`
        :param r: major radius of the probe's centre [metre]
        :param z: height of the probe's centre [metre]
        :param poloidal_angle: angle of the probe's normal, counter-clockwise from `grad(R)`: GSFit's COCOS 13, not the data dictionary's COCOS 17 [radian]
        :param time: the experimental timebase [second]
        :param field: the measured field along the probe's normal [tesla]
        """
        ...
    def add_flux_loop(
        self,
        name: str,
        r: float,
        z: float,
        time: npt.NDArray[np.float64],
        flux: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a flux loop to `magnetics/flux_loop`.

        :param name: short identifier for the flux loop, unique within the flux loops, e.g. `"L001"`
        :param r: major radius of the loop [metre]
        :param z: height of the loop [metre]
        :param time: the experimental timebase [second]
        :param flux: the measured poloidal flux, GSFit's `psi` [weber]

        The loop goes all the way around the machine, so it is a `poloidal_flux` loop with one `position`.
        """
        ...
    def add_rogowski_coil(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        gaps_name: list[str] = ...,
        gaps_r_start: npt.NDArray[np.float64] | None = None,
        gaps_r_end: npt.NDArray[np.float64] | None = None,
        gaps_z_start: npt.NDArray[np.float64] | None = None,
        gaps_z_end: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Append a Rogowski coil to `magnetics/rogowski_coil`.

        :param name: short identifier for the Rogowski coil, unique within the Rogowski coils, e.g. `"INIVC000"`
        :param r: major radius of each point of the coil's contour, at least 3 points [metre]
        :param z: height of each point of the coil's contour [metre]
        :param time: the experimental timebase [second]
        :param current: the measured current [ampere]
        :param gaps_name: (optional) short identifier for each gap in the coil's winding, unique within the coil
        :param gaps_r_start: (optional) major radius of the start of each gap [metre]
        :param gaps_r_end: (optional) major radius of the end of each gap [metre]
        :param gaps_z_start: (optional) height of the start of each gap [metre]
        :param gaps_z_end: (optional) height of the end of each gap [metre]

        Each gap is the straight line from its start to its end; the documentation of the Rust `RogowskiCoilGap::r_start` says which
        way it runs. A coil with no gaps leaves the gap arguments out.
        """
        ...
    def add_diamagnetic_flux(
        self,
        method_name: str,
        time: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
    ) -> None:
        """
        Add the diamagnetic flux to `magnetics/diamagnetic_flux`, as `magnetics/diamagnetic_flux(0)`.

        GSFit accepts only one diamagnetic flux, as the equilibrium IDS has a single diamagnetic flux constraint,
        `time_slice(itime)/constraints/diamagnetic_flux`. A machine with several diamagnetic loops averages their signals into one before
        adding it.

        :param method_name: name of the method the diamagnetic flux is calculated with, e.g. `"DIALOOP"`
        :param time: the experimental timebase [second]
        :param data: the diamagnetic flux, positive counter-clockwise viewed from above [weber]
        :raises ValueError: When a diamagnetic flux has been added already, `method_name` is empty, or `data` does not have one value per
            time
        """
        ...
    def add_ip(
        self,
        method_name: str,
        time: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a plasma current to `magnetics/ip`, one for each method of calculating it.

        :param method_name: name of the method the plasma current is calculated with, unique within the plasma currents
        :param time: the experimental timebase [second]
        :param data: the plasma current, positive counter-clockwise viewed from above [ampere]

        The data dictionary asks for the generally recommended method to be added first.
        """
        ...
    def add_isoflux(
        self,
        name: str,
        description: str,
        time: npt.NDArray[np.float64],
        position_1_r: npt.NDArray[np.float64],
        position_1_z: npt.NDArray[np.float64],
        position_2_r: npt.NDArray[np.float64],
        position_2_z: npt.NDArray[np.float64],
        validity_timed: npt.NDArray[np.int32] | None = None,
        validity: int | None = None,
    ) -> None:
        """
        Append an isoflux constraint to `magnetics/isoflux`, a custom key: two points which lie on the same flux surface, so that the
        poloidal flux at the first minus the poloidal flux at the second is zero.

        The points can move, so each is a time trace, on the constraint's own timebase, which need not be the reconstruction times:
        `Equilibrium.copy_from_magnetics_ids` interpolates it onto them. A time where either point is NaN, or which is not valid, is not
        fitted.

        :param name: short identifier for the constraint, unique within the isoflux constraints, e.g. `"ISOFLUX01"`, matching its fit
            settings, the equilibrium settings' `constraints/isoflux/isoflux(i1)/name`
        :param description: description, e.g. where the two points are read from
        :param time: the constraint's timebase, increasing [second]
        :param position_1_r: the first point's major radius, positive, or NaN where it is not known [metre]
        :param position_1_z: the first point's height [metre]
        :param position_2_r: the second point's major radius, positive, or NaN where it is not known [metre]
        :param position_2_z: the second point's height [metre]
        :param validity_timed: (optional) the validity at each time, as a magnetic sensor's `validity_timed`: negative is not valid
        :param validity: (optional) the validity of the whole time trace: negative is not valid
        :raises ValueError: When the name is empty or repeated, the timebase is empty, not finite or not increasing, a point or
            `validity_timed` does not have one value per time, or a major radius is finite but not positive
        """
        ...
    def add_isoflux_boundary(
        self,
        name: str,
        description: str,
        time: npt.NDArray[np.float64],
        position_r: npt.NDArray[np.float64],
        position_z: npt.NDArray[np.float64],
        validity_timed: npt.NDArray[np.int32] | None = None,
        validity: int | None = None,
    ) -> None:
        """
        Append an isoflux boundary constraint to `magnetics/isoflux_boundary`, a custom key: a point which lies on the plasma boundary,
        so that the poloidal flux there minus the poloidal flux at the boundary is zero.

        The point can move, so it is a time trace, on the constraint's own timebase, which need not be the reconstruction times:
        `Equilibrium.copy_from_magnetics_ids` interpolates it onto them. A time where the point is NaN, or which is not valid, is not
        fitted.

        :param name: short identifier for the constraint, unique within the isoflux boundary constraints, e.g. `"BOUNDARY01"`, matching its
            fit settings, the equilibrium settings' `constraints/isoflux_boundary/isoflux_boundary(i1)/name`
        :param description: description, e.g. where the point is read from
        :param time: the constraint's timebase, increasing [second]
        :param position_r: the point's major radius, positive, or NaN where it is not known [metre]
        :param position_z: the point's height [metre]
        :param validity_timed: (optional) the validity at each time, as a magnetic sensor's `validity_timed`: negative is not valid
        :param validity: (optional) the validity of the whole time trace: negative is not valid
        :raises ValueError: When the name is empty or repeated, the timebase is empty, not finite or not increasing, the point or
            `validity_timed` does not have one value per time, or the major radius is finite but not positive
        """
        ...
    def add_stationary_point(
        self,
        name: str,
        description: str,
        time: npt.NDArray[np.float64],
        position_r: npt.NDArray[np.float64],
        position_z: npt.NDArray[np.float64],
        validity_timed: npt.NDArray[np.int32] | None = None,
        validity: int | None = None,
    ) -> None:
        """
        Append a stationary point constraint to `magnetics/stationary_point`, a custom key: a point where both derivatives of the
        poloidal flux, `d(psi)/d(r)` and `d(psi)/d(z)`, are zero, an X-point or the magnetic axis.

        The point can move, so it is a time trace, on the constraint's own timebase, which need not be the reconstruction times:
        `Equilibrium.copy_from_magnetics_ids` interpolates it onto them. A time where the point is NaN, or which is not valid, is not
        fitted.

        :param name: short identifier for the constraint, unique within the stationary point constraints, e.g. `"MAGNETIC_AXIS"` or
            `"X_POINT_LOWER"`, matching its fit settings, the equilibrium settings' `constraints/stationary_point/stationary_point(i1)/name`
        :param description: description, e.g. whether the point is the magnetic axis or an X-point, and where it is read from
        :param time: the constraint's timebase, increasing [second]
        :param position_r: the point's major radius, positive, or NaN where it is not known [metre]
        :param position_z: the point's height [metre]
        :param validity_timed: (optional) the validity at each time, as a magnetic sensor's `validity_timed`: negative is not valid
        :param validity: (optional) the validity of the whole time trace: negative is not valid
        :raises ValueError: When the name is empty or repeated, the timebase is empty, not finite or not increasing, the point or
            `validity_timed` does not have one value per time, or the major radius is finite but not positive
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.magnetics_paths`, straight out of the magnetics IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.Magnetics.get`.
        """
        ...
    @property
    def magnetics_ids(self) -> MagneticsIds:
        """A copy of the whole magnetics IDS, read with `gsfit_rs.imas.magnetics_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Symmetry:
    """Up-down symmetry of the equilibrium configuration.

    `Symmetric` selects the 7-term cos(h_n y) expansion (limiter or double-null);
    `Asymmetric` adds the sin(h_n y) family for the 12-term single-null solve.
    """

    Symmetric: Symmetry
    Asymmetric: Symmetry

class Configuration:
    """Plasma boundary topology and the shape parameters it requires.

    Each variant carries exactly its own shape inputs; up-down symmetry
    (7- vs 12-term expansion) is implied by the variant.
    """
    @staticmethod
    def SymmetricLimited(kappa: float, delta: float) -> Configuration:
        """Up-down symmetric smooth limiter (7-term).

        :param kappa: Elongation [dimensionless]
        :param delta: Triangularity [dimensionless]
        """
        ...
    @staticmethod
    def AntisymmetricLimited(kappa_upper: float, delta_upper: float, kappa_lower: float, delta_lower: float) -> Configuration:
        """Up-down asymmetric smooth limiter (12-term). NOT YET IMPLEMENTED.

        :param kappa_upper: Upper elongation [dimensionless]
        :param delta_upper: Upper triangularity [dimensionless]
        :param kappa_lower: Lower elongation [dimensionless]
        :param delta_lower: Lower triangularity [dimensionless]
        """
        ...
    @staticmethod
    def DoubleNull(kappa_x: float, delta_x: float) -> Configuration:
        """Up-down symmetric double-null divertor (7-term).

        :param kappa_x: X-point elongation [dimensionless]
        :param delta_x: X-point triangularity [dimensionless]
        """
        ...
    @staticmethod
    def SingleNull(kappa: float, delta: float, kappa_x: float, delta_x: float) -> Configuration:
        """Up-down asymmetric lower single-null divertor (12-term). The upper half is a smooth Miller
        profile (`kappa`, `delta`) and the lower half has the X-point (`kappa_x`, `delta_x`).

        :param kappa: Upper (smooth) elongation [dimensionless]
        :param delta: Upper (smooth) triangularity [dimensionless]
        :param kappa_x: X-point elongation [dimensionless]
        :param delta_x: X-point triangularity [dimensionless]
        """
        ...

class GuazzottoFreidberg:
    """Guazzotto-Freidberg analytic Grad-Shafranov equilibrium.

    Solves the analytic equilibrium and produces a `PfActive` object whose PF-coil
    currents reproduce the vacuum field consistent with the solution, i.e. a
    self-consistent free-boundary test case for GSFit to reconstruct.

    Flux follows the IMAS convention, the total poloidal flux `psi = 2 * pi * R * A_phi` [weber],
    so `psi_0` and the equilibrium IDS are `2 * pi` times GF's per-radian values.

    Reference: J. Plasma Phys. 87 (2021) 905870303.
    """
    def __new__(
        cls,
        configuration: Configuration,
        eps: float,
        nu: float,
        r_geo: float,
        bt_vac_at_r_geo: float,
        p_axis: float,
        n_r: int,
        n_z: int,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
        n_radial_expansion: int,
    ) -> GuazzottoFreidberg:
        """Construct the equilibrium; it is not solved until `solve` is called.

        :param configuration: Boundary topology + shape (SymmetricLimited, AntisymmetricLimited, DoubleNull, or SingleNull)
        :param eps: Inverse aspect ratio, a / r_geo [dimensionless]
        :param nu: Profile parameter, approximately the poloidal beta [dimensionless]
        :param r_geo: Geometric major radius [metre]
        :param bt_vac_at_r_geo: Vacuum toroidal field at r_geo [tesla]
        :param p_axis: Pressure on the magnetic axis [pascal]
        :param n_r: Number of radial grid points [dimensionless]
        :param n_z: Number of vertical grid points [dimensionless]
        :param r_min: Minimum radius [metre]
        :param r_max: Maximum radius [metre]
        :param z_min: Minimum vertical position [metre]
        :param z_max: Maximum vertical position [metre]
        :param n_radial_expansion: Number of terms in the C_n / S_n series (the paper's M) [dimensionless]
        """
        ...
    def solve(
        self,
    ) -> None:
        """Solve the eigenvalue `alpha` and evaluate the analytic flux and plasma mask over the (R, Z) grid."""
        ...
    def get_pf_active(
        self,
        coil_regularisation_weight: float,
        control_points: npt.NDArray[np.float64],
        wall: Wall,
    ) -> PfActive:
        """
        Fit the PF-coil currents on the (R, Z) grid boundary so that, with the plasma's own field, they reproduce
        the analytic flux at `control_points`. Then store the free-boundary equilibrium in `equilibrium_ids` and
        post-process it. Requires `solve` first.

        The IDS holds the total flux, plasma plus coils, which is valid in the vacuum as well as inside the plasma.
        For a diverted plasma the magnetic axis, the X-points, the boundary flux and the plasma mask are found in
        that flux, as the Grad-Shafranov solver finds them; for a limited one they are the analytic ones.

        :param coil_regularisation_weight: Tikhonov weight on the coil-current magnitude [dimensionless]
        :param control_points: (n_control_point, 2) array of (r, z) points where the flux is matched [metre]
        :param wall: The wall: its limiter is copied into `equilibrium_ids`, where its limiter and vacuum vessel bound a
            diverted plasma, and the post-processor traces the scrape-off layer legs up to it
        :return: One single-filament PF coil per boundary node, carrying the fitted current, at the single time 0
        """
        ...
    def get_sensor_values(
        self,
        magnetics: Magnetics,
        pf_active: PfActive,
    ) -> None:
        """
        Fill the magnetic sensors' measurements with what the analytic plasma current plus `pf_active` produce at each sensor:
        `magnetics/flux_loop(i1)/flux` (weber) and `magnetics/b_field_pol_probe(i1)/field` (tesla). Requires `solve` first. The
        sensors must already be in `magnetics`, e.g. added with `Magnetics.add_flux_loop` and `Magnetics.add_b_field_pol_probe`.

        Each coil filament carries its coil's current at the first time of its experimental timebase, so each measurement is a
        signal with that one time, the first coil's, or 0 without any coils.

        :param magnetics: The magnetic sensors, with their geometry already set; their measurements are overwritten
        :param pf_active: The PF coils, typically the ones `get_pf_active` returned
        """
        ...
    def model_surface(
        self,
        n_point_per_half: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        GF's model surface, the shape the analytic boundary is matched to at the matching points (the red curve in
        GF's figures). Does not need `solve`.

        :param n_point_per_half: Number of points on each of the upper and lower halves; must be even
        :return: `(r, z)`, the closed outline, anticlockwise from the outer midplane; the last point repeats the first,
            so each is `2 * n_point_per_half + 1` long [metre]
        """
        ...
    @property
    def configuration(self) -> Configuration:
        """Boundary topology and its shape parameters"""
        ...
    @property
    def eps(self) -> float:
        """Inverse aspect ratio, a / r_geo [dimensionless]"""
        ...
    @property
    def nu(self) -> float:
        """Profile parameter, approximately the poloidal beta [dimensionless]"""
        ...
    @property
    def r_geo(self) -> float:
        """Geometric major radius [metre]"""
        ...
    @property
    def bt_vac_at_r_geo(self) -> float:
        """Vacuum toroidal field at r_geo [tesla]"""
        ...
    @property
    def p_axis(self) -> float:
        """Pressure normalisation, the pressure where psi_hat = 1 [pascal]"""
        ...
    @property
    def n_radial_expansion(self) -> int:
        """Number of terms in the C_n / S_n series (the paper's M) [dimensionless]"""
        ...
    @property
    def alpha(self) -> float:
        """The eigenvalue; NaN until `solve` has run [dimensionless]"""
        ...
    @property
    def beta_0(self) -> float:
        """Beta at psi_hat = 1, 2 * mu_0 * p_axis / bt_vac_at_r_geo ** 2 (GF Eq. 6.4); NaN until `solve` has run [dimensionless]"""
        ...
    @property
    def bt_diamagnetic_shift(self) -> float:
        """Diamagnetic change in the toroidal field (GF Eq. 6.4); NaN until `solve` has run [tesla]"""
        ...
    @property
    def psi_0(self) -> float:
        """Flux normalisation, psi = psi_0 * psi_hat: GF's Psi_0 (Eq. 6.5) times 2 * pi; NaN until `solve` has run [weber]"""
        ...
    @property
    def psi_analytic(self) -> npt.NDArray[np.float64]:
        """The analytic flux, psi_0 * psi_hat, on every grid cell, shape (n_z, n_r) [weber].

        Outside the model box the truncated series is unphysical and can diverge; use `mask` to select the plasma.
        """
        ...
    @property
    def mask(self) -> npt.NDArray[np.float64]:
        """1 inside the analytic plasma and 0 elsewhere, shape (n_z, n_r) [dimensionless]"""
        ...
    @property
    def equilibrium_ids(self) -> EquilibriumIds:
        """The free-boundary equilibrium IDS, read with `gsfit_rs.imas.equilibrium_paths`. Empty until `get_pf_active` has run.

        The IDS is copied into the returned object, so it is a snapshot: changes made on the Rust side afterwards
        are not seen by it.
        """
        ...

class BuxtonProkopyszyn:
    """Analytic Grad-Shafranov equilibrium with the paper's EFIT-polynomial p' and ff' of any order, each with a constant term:
    Guazzotto and Freidberg's solution generalised (P. F. Buxton and A. P. K. Prokopyszyn).

    Each profile is c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1): c_0 is the constant (Solov'ev) term, its value on the LCFS,
    c_1 the coefficient of (1 - psi_N), c_2 that of (1 - psi_N) psi_N, and so on. The coefficients, in physical units, are the inputs:
    [c_0_p_prime, c_1_p_prime, ...] and [c_0_ff_prime, c_1_ff_prime, ...]; [0, c_1] for both is Guazzotto and Freidberg's solution.

    The flux is a cosine series in Z at n_wave equally spaced wavenumbers k_j, whose coefficients are power series in R**2 - R_c**2
    (or ln(R**2 / R_c**2)); the profile nonlinearity is summed as a Taylor series in lambda about Guazzotto and Freidberg's solution;
    the shape is fitted to the LCFS by least squares. `solve` also fills `equilibrium_ids` on the (R, Z) grid and post-processes it.

    The solution is analytic only where the Taylor series converges. When it does not, `solve` keeps the partial sum with the
    smallest GS residual and warns (`gs_residual` above 1e-6): such an equilibrium is one for a numerical Grad-Shafranov solver.

    The equilibrium is a fixed-boundary one: outside the LCFS the flux is the analytic continuation of the series (not a vacuum
    field), and NaN where the radial series cannot be evaluated. The flux is the total poloidal flux, psi = 2 pi R A_phi [weber], with
    psi_b = 0, and psi_N = (psi_a - psi) / (psi_a - psi_b) is 0 on the magnetic axis and 1 on the LCFS (the paper's notation).

    It is constructed from its settings, as GSFit's `Equilibrium` is, which the Python driver, `analytic_grad_shafranov.buxton_prokopyszyn`,
    reads from XML: they are written to `equilibrium_ids`' `code/parameters`, and read into structs generated from their XML schema,
    `python/analytic_grad_shafranov/buxton_prokopyszyn/settings_schema/xsd/equilibrium/equilibrium.xsd`, which describes every setting.
    """
    def __new__(cls, parameters_input: ParametersInput) -> BuxtonProkopyszyn:
        """Construct the analytic equilibrium from its settings; it is not solved until `solve` is called.

        The coefficients fix the equilibrium: psi_a - psi_b, the plasma current and p_axis follow from them. With `global_quantities/ip`,
        both lists are multiplied by the one factor that gives that current instead (`coefficient_scale`, the paper's s), which leaves the
        flux surfaces unchanged. The linear part's value at R_c = (r_geo**2 + a_minor**2)**0.5, mu0 R_c**2 c_bar_p_prime +
        c_bar_ff_prime (the bracket of the paper's k_max**2), must not vanish and must have the sign of its p' part: an ff' diamagnetic
        enough to cancel or reverse the toroidal current density of p' at R_c is a ValueError.

        :param parameters_input: The settings, whose XML is `equilibrium.xml` with each file it links to written in its link's place, as
            `serialise_python_dataclass_to_imas_parameters_input` writes it; it is written to `equilibrium_ids`' `code/parameters`
        :raises ValueError: The XML cannot be read into the settings' structs, or the settings cannot be solved
        """
        ...
    def solve(self) -> None:
        """Solve the normalised problem, then the physical scalars and the diagnostics; then fill `equilibrium_ids` on the grid and
        post-process it. With n_boundary_lawson_iteration_max > 0 the solve is repeated with the LCFS points reweighted, and the
        solution with the smallest boundary misfit kept (options_used["weighting_misfits"] and ["weighting_stop"] say how it went)."""
        ...
    def __repr__(self) -> str:
        """The wavenumbers of the series in units of k_max, to two decimals, in order of k**2: the cosh terms' imaginary ones,
        k_j = i m k_spacing_cosh k_max (k_j**2 < 0, cosh(|k_j| Z) = cos(k_j Z)), m = n_wave_cosh..1, then the paper's
        k_j = (j - 1) k_spacing k_max, j = 1..n_wave, e.g. `k = [1.0i, 0.5i, 0.0, 0.33, 0.67, 1.0, 1.33] * k_max`"""
        ...
    @staticmethod
    def valid_r_range(parameters_input: ParametersInput) -> tuple[float, float]:
        """The radial extent where the radial power series can be evaluated for the settings' shape, which the IDS grid must lie in
        (`psi(r, z)` and the other methods are NaN outside it): |R**2 - R_c**2| <= 0.98 R_c**2 with "r_squared"; with "log_r_squared",
        ln(R**2 / R_c**2) within 1.1 half-widths of the middle of the plasma's range of it. Only the settings' `configuration` and
        `numerics/radial_series/radial_variable` are used, so the grid can be chosen from it.

        :param parameters_input: The settings, as the constructor takes them
        :return: `(r_min, r_max)` [metre]
        :raises ValueError: The XML cannot be read into the settings' structs, or the configuration rejects the shape
        """
        ...
    def psi(self, r: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The total poloidal flux, psi_a on the magnetic axis and psi_b = 0 on the LCFS [weber]"""
        ...
    def psi_n(self, r: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The paper's psi_N, 0 on the magnetic axis and 1 on the LCFS [dimensionless]"""
        ...
    def psi_derivatives(self, r: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        """`psi` [weber]; `d_psi_d_r`, `d_psi_d_z` [weber / metre]; `d2_psi_d_r2`, `d2_psi_d_r_d_z`, `d2_psi_d_z2` [weber / metre ** 2]"""
        ...
    def b_field(self, r: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """`(b_r, b_z)` of the plasma's analytic flux, B_R = -(d psi / d Z) / (2 pi R), B_Z = (d psi / d R) / (2 pi R) [tesla]"""
        ...
    def b_field_phi(self, r: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The toroidal field B_phi = f(psi_N) / R inside the LCFS and r0 b0 / R outside [tesla]"""
        ...
    def j_phi(self, r: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The toroidal current density J_phi = 2 pi (R p' + ff' / (mu0 R)) inside the LCFS, 0 outside [ampere / metre ** 2]"""
        ...
    def is_inside(self, r: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Inside the analytic LCFS"""
        ...
    def pressure(self, psi_n: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """p(psi_N) = (psi_a - psi_b) int_{psi_N}^1 p' dpsi_N, zero on the LCFS [pascal]"""
        ...
    def f(self, psi_n: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """f = R B_phi at psi_N, f**2 = (r0 b0)**2 + 2 (psi_a - psi_b) int_{psi_N}^1 ff' dpsi_N, with the sign of b0 [tesla * metre]"""
        ...
    def p_prime(self, psi_n: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """p' = dp / dpsi = c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1) at psi_N, with `p_prime_coefficients` [pascal / weber]"""
        ...
    def ff_prime(self, psi_n: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ff' = f df / dpsi = c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1) at psi_N, with `ff_prime_coefficients`
        [tesla ** 2 * metre ** 2 / weber]"""
        ...
    def boundary(self, n_point: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The analytic LCFS, closed, `(boundary_r, boundary_z)`, about `n_point` points [metre]"""
        ...
    def model_surface(self, n_point: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The shape the LCFS was fitted to (the model surface, or the user's points), closed, `(r, z)`, about `n_point` points
        [metre]"""
        ...
    def x_points(self) -> list[tuple[float, float]]:
        """`(x_point_r, x_point_z)` of every X-point [metre]"""
        ...
    def x_point(self) -> tuple[float, float] | None:
        """`(x_point_r, x_point_z)` of the first X-point, or None [metre]"""
        ...
    def plasma_current(self, n_node: int) -> float:
        """The plasma current, I_p = int J_phi dR dZ over the LCFS, by Gauss-Legendre quadrature with `n_node` nodes in each
        direction of each half (128 gives about 1e-14) [ampere]"""
        ...
    @property
    def nu(self) -> float:
        """Guazzotto and Freidberg's nu of the profiles, an output only (the solution does not use it): the share of the toroidal current density at R_c = (r_geo**2 +
        a_minor**2)**0.5 carried by p', with both profiles at their values on the magnetic axis (c_0 + c_1),
        mu0 R_c**2 p'(axis) / (mu0 R_c**2 p'(axis) + ff'(axis)); 0 without p', 1 without ff', above 1 for a diamagnetic ff', infinite
        where the two cancel, NaN where both vanish on the axis [dimensionless]"""
        ...
    @property
    def r_geo(self) -> float:
        """The paper's R_geo: the model surface's, or the middle of the radial extent of the LCFS points and the X-point [metre]"""
        ...
    @property
    def z_geo(self) -> float:
        """The height about which the series is built: the model surface's z_geo, or the middle of the vertical extent of the LCFS points
        and the X-point [metre]"""
        ...
    @property
    def a_minor(self) -> float:
        """The paper's a_minor: the model surface's, or half the radial extent of the LCFS points and the X-point [metre]"""
        ...
    @property
    def k_max(self) -> float:
        """The paper's k_max, k_max**2 = (2 pi)**2 (mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime) / (psi_a - psi_b) for the linear part's
        c_bar (options_used["c_bar_method"]): the sum of the orders' k_max,l**2 kept [1 / metre]"""
        ...
    @property
    def psi_a_minus_psi_b(self) -> float:
        """The paper's psi_a - psi_b, with psi_b = 0 [weber]"""
        ...
    @property
    def p_axis(self) -> float:
        """The pressure on the magnetic axis [pascal]"""
        ...
    @property
    def coefficient_scale(self) -> float:
        """The factor the given coefficients were multiplied by: 1, or the one that gives `global_quantities/ip` [dimensionless]"""
        ...
    @property
    def mag_r(self) -> float:
        """Magnetic axis [metre]"""
        ...
    @property
    def mag_z(self) -> float:
        """Magnetic axis [metre]"""
        ...
    @property
    def gs_residual(self) -> float:
        """The paper's GS residual: the largest difference between the two sides of the Grad-Shafranov equation inside the LCFS,
        over the largest value of its right-hand side [dimensionless]"""
        ...
    @property
    def boundary_misfit(self) -> float:
        """Largest distance from the given shape to the analytic LCFS [metre]"""
        ...
    @property
    def boundary_weights_used(self) -> npt.NDArray[np.float64]:
        """The relative weight of each given LCFS point's row in the kept solution's fits: all 1 with
        n_boundary_lawson_iteration_max = 0, else Lawson's, normalised to mean(w**2) = 1; empty for a model surface [dimensionless]"""
        ...
    @property
    def boundary_deviations(self) -> npt.NDArray[np.float64]:
        """The distance of each given LCFS point from the analytic LCFS, |psi| / |grad psi|, in the given order; NaN next to an
        X-point; empty for a model surface [metre]"""
        ...
    @property
    def x_point_flux(self) -> float:
        """1 - psi_N at the X-point, 0 when the separatrix is the LCFS [dimensionless]"""
        ...
    @property
    def is_closed(self) -> bool:
        """The LCFS is a closed set of nested surfaces"""
        ...
    @property
    def is_unstable(self) -> bool:
        """The series' coefficient sums grow"""
        ...
    @property
    def n_taylor(self) -> int:
        """The number of Taylor terms kept, orders 0..n_taylor - 1: the partial sum with the smallest GS residual [dimensionless]"""
        ...
    @property
    def fit_used(self) -> str:
        """The fit of orders >= 1 kept: `none`, `tikhonov` or `tsvd`"""
        ...
    @property
    def order_gs_residuals(self) -> npt.NDArray[np.float64]:
        """The GS residual of every partial sum [dimensionless]"""
        ...
    @property
    def coefficient_sizes(self) -> npt.NDArray[np.float64]:
        """sum |d_{j,m}| of every order of the series, the paper's (1 - psi_N)_l [dimensionless]"""
        ...
    @property
    def k_max_squared_orders(self) -> npt.NDArray[np.float64]:
        """The paper's k_max,l**2 of every order [1 / metre ** 2]"""
        ...
    @property
    def axis_radial_orders(self) -> npt.NDArray[np.float64]:
        """The series' coefficients in lambda of the magnetic axis' shift from order 0's, in the scaled radial variable,
        (R**2 - R_c**2) / (2 R_geo a_minor) (r_squared) or ln(R**2 / R_c**2) scaled to -1 and 1 at the plasma's innermost and
        outermost R (log_r_squared) [dimensionless]"""
        ...
    @property
    def axis_z_orders(self) -> npt.NDArray[np.float64]:
        """... and in Z [metre]"""
        ...
    @property
    def n_wavenumbers(self) -> list[int]:
        """The number of distinct non-zero wavenumbers in every order of the series [dimensionless]"""
        ...
    @property
    def order_half_widths(self) -> list[list[int]]:
        """The largest multiple of the wavenumber spacing (and of the cosh terms' spacing) in every order of the series
        [dimensionless]"""
        ...
    @property
    def dropped_source(self) -> float:
        """The largest source the growth cut-offs dropped (wavenumbers far above k_max), relative to k_max,0**2 times the size of
        order 0, the size of the terms of its equation [dimensionless]"""
        ...
    def order_terms(self, i_taylor: int, r: float, z: float) -> dict[str, npt.NDArray[np.complex128] | npt.NDArray[np.float64] | float]:
        """The terms of one order of the Taylor series at one point, for looking at its sums; with verbose=True only.

        Order i_taylor's (1 - psi_N)_l at (R, Z) is sum_j sum_m terms[j, m]: terms[j, m] is the radial power m of the wavenumber
        k_j = k_over_k_max[j] k_max, with k_j and -k_j together, cos(k_j (Z - z_geo)) and sin(k_j (Z - z_geo)) (cosh and sinh for an
        imaginary k_j). The sum of the orders 0..n - 1 (lambda = 1), divided by its value on its own magnetic axis, is the partial sum
        whose GS residual is order_gs_residuals[n - 1].

        The terms are also split as the paper's sum over the wavenumbers,
        cos(k_j Z') [A_j C_j + B_j S_j + P_j] + sin(k_j Z') [A~_j C_j + B~_j S_j + P~_j], Z' = Z - z_geo: C_j is 1 with zero slope at
        R_c and S_j 0 with unit slope there, as order 0's terms are built; A_j and B_j are the value and slope at R_c of the radial
        function of cos(k_j Z'), and P_j, zero with zero slope at R_c, is the rest: the particular part of orders >= 1, rounding at
        order 0. Likewise for sin(k_j Z'); `terms` is the sum of the six parts.

        :param i_taylor: The order, below len(k_max_squared_orders); 0 is Guazzotto and Freidberg's solution of the profiles' linear part
            [dimensionless]
        :param r: Major radius [metre]
        :param z: Height [metre]
        :return: `k_over_k_max`, the wavenumbers in units of order 0's k_max, complex, imaginary for the cosh terms [dimensionless];
            `terms`, (n_k, n_radial) [dimensionless]; its parts `terms_a` (cos(k_j Z') A_j C_j), `terms_b` (cos(k_j Z') B_j S_j),
            `terms_a_tilde` (sin(k_j Z') A~_j C_j), `terms_b_tilde` (sin(k_j Z') B~_j S_j), `terms_particular` (cos(k_j Z') P_j) and
            `terms_particular_tilde` (sin(k_j Z') P~_j), each (n_k, n_radial) [dimensionless]; `radial`, the radial variable at R, -1
            and +1 at the plasma's innermost and outermost R [dimensionless]; and `k_max`, order 0's [1 / metre]
        """
        ...
    @property
    def p_prime_coefficients(self) -> npt.NDArray[np.float64]:
        """[c_0_p_prime, c_1_p_prime, ...] of the solution: the given ones (padded to two) times `coefficient_scale`; GSFit's
        EfitPolynomial DOFs are [c_1_p_prime, ...] [pascal / weber]"""
        ...
    @property
    def ff_prime_coefficients(self) -> npt.NDArray[np.float64]:
        """[c_0_ff_prime, c_1_ff_prime, ...] of the solution, likewise [tesla ** 2 * metre ** 2 / weber]"""
        ...
    @property
    def options_used(self) -> dict[str, object]:
        """Every option as `solve` resolved it, the fit used, and the gsfit_rs version"""
        ...
    @property
    def warnings(self) -> list[str]:
        """The warnings `solve` raised"""
        ...
    @property
    def equilibrium_ids(self) -> EquilibriumIds:
        """The post-processed equilibrium IDS, read with `gsfit_rs.imas.equilibrium_paths`; empty until `solve` has run. A snapshot."""
        ...

class EfitPolynomial:
    def __new__(
        cls,
        n_dof: int,
        regularisations: npt.NDArray[np.float64],
        exact: bool = False,
        coefficients: npt.NDArray[np.float64] | None = None,
    ) -> EfitPolynomial:
        """
        :param n_dof: Number of degrees of freedom
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values [dimensionless]; not used when `exact=True`
        :param exact: When `True` the coefficients are fixed to `coefficients` rather than fitted. With both `p_prime` and `ff_prime` exact, this is a forward solve:
            their shapes are fixed, and if there is a constraint besides the magnetic axis (normally the plasma current from a Rogowski coil) they share one fitted amplitude,
            which holds the plasma radially. The coefficients written to the equilibrium are the given ones times this amplitude
        :param coefficients: A 1D array of size [n_dof] with the fixed coefficients; required when `exact=True`
        """
        ...

class TensionedCubicBSpline:
    def __new__(
        cls,
        regularisations: npt.NDArray[np.float64],
        interior_knots: npt.NDArray[np.float64],
        interval_tensions: npt.NDArray[np.float64],
    ) -> TensionedCubicBSpline:
        """
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values [dimensionless]
        :param interior_knots: A 1D array of size [n_interior_knots] with the interior knots [dimensionless]
        :param interval_tensions: A 1D array of size [n_intervals] with the interval tensions [dimensionless]
        """
        ...

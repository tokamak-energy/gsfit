from .gsfit import Gsfit


class Gsfit_Ts(Gsfit):
    """Pressure-constrained ("Thomson scattering") GSFit, writing to the ``GSFIT_TS`` tree.

    This bakes in the settings-driven workflow of
    ``examples/example_13__st40__real_data__pressure_constrained_simplified.py``:
      1. only reconstruct time-slices where the pressure (TS) sensors are "good",
      2. turn the pressure (Thomson scattering) sensors on, and
      3. use a tensioned cubic B-spline for p'.

    It also forces ``analysis_name="GSFIT_TS"`` so the results are written to the
    ``GSFIT_TS`` MDSplus tree rather than the default ``GSFIT`` tree (the tree name is
    derived from ``analysis_name`` in ``DiagnosticAndSimulationBase``).

    Concrete subclasses only differ in the p' tensioned-cubic-B-spline regularisation
    scan point, which they set via the ``REGULARISATION_SCALE`` and
    ``RIGHT_BOUNDARY_CONDITION`` class attributes.
    """

    # Set by the concrete Gsfit_Ts_N subclasses below.
    REGULARISATION_SCALE: float
    RIGHT_BOUNDARY_CONDITION: str

    def __init__(
        self,
        pulseNo: int,
        run_name: str,
        run_description: str = "Pressure-constrained (Thomson scattering) GSFit",
        write_to_mds: bool = True,
        pulseNo_write: int | None = None,
        link_run_to_best: bool = False,
    ) -> None:
        super().__init__(
            pulseNo=pulseNo,
            run_name=run_name,
            run_description=run_description,
            write_to_mds=write_to_mds,
            pulseNo_write=pulseNo_write,
            analysis_name="GSFIT_TS",
            link_run_to_best=link_run_to_best,
        )

        # 1. Only reconstruct the time-slices where the pressure (TS) sensors are "good".
        self.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "good_pressure_sensors"
        # 2. Turn on the pressure (Thomson scattering) sensors.
        self.settings["sensor_weights_pressure.json"]["include"] = True
        # 3. Use a tensioned cubic B-spline for p'.
        self.settings["source_function_p_prime.json"]["method"] = "tensioned_cubic_b_spline"

        # Scan point: the p' regularisation scale and the right boundary condition.
        regularisation_builder = self.settings["source_function_p_prime.json"]["tensioned_cubic_b_spline"]["regularisation_builder"]
        regularisation_builder["regularisation_scale"] = self.REGULARISATION_SCALE
        regularisation_builder["right_boundary_condition"] = self.RIGHT_BOUNDARY_CONDITION


class Gsfit_Ts_1(Gsfit_Ts):
    """Pressure-constrained GSFit: reg_scale=1e-6, right boundary condition = free."""

    REGULARISATION_SCALE = 1e-6
    RIGHT_BOUNDARY_CONDITION = "free"


class Gsfit_Ts_2(Gsfit_Ts):
    """Pressure-constrained GSFit: reg_scale=1e-6, right boundary condition = dirichlet."""

    REGULARISATION_SCALE = 1e-6
    RIGHT_BOUNDARY_CONDITION = "dirichlet"


class Gsfit_Ts_3(Gsfit_Ts):
    """Pressure-constrained GSFit: reg_scale=1e-5, right boundary condition = free."""

    REGULARISATION_SCALE = 1e-5
    RIGHT_BOUNDARY_CONDITION = "free"


class Gsfit_Ts_4(Gsfit_Ts):
    """Pressure-constrained GSFit: reg_scale=1e-5, right boundary condition = dirichlet."""

    REGULARISATION_SCALE = 1e-5
    RIGHT_BOUNDARY_CONDITION = "dirichlet"

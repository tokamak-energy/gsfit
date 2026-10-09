from .gsfit import Gsfit


class Gsfit_Ts(Gsfit):
    """Pressure-constrained ("Thomson scattering") GSFit, writing to the ``GSFIT_TS`` tree.

    This configures the simple EFIT-polynomial pressure workflow:
      1. only reconstruct time-slices where the pressure (TS) sensors are "good",
      2. turn the pressure (Thomson scattering) sensors on, and
      3. use the EFIT polynomial for p', with either fixed or free edge pressure.

    It also forces ``analysis_name="GSFIT_TS"`` so the results are written to the
    ``GSFIT_TS`` MDSplus tree rather than the default ``GSFIT`` tree (the tree name is
    derived from ``analysis_name`` in ``DiagnosticAndSimulationBase``).
    """

    PRESSURE_EDGE_FREE: bool
    PRESSURE_SENSORS_ENABLED: bool = True
    ISOFLUX_SENSORS_ENABLED: bool = False

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
        # 2. Configure which Thomson-derived constraints are included.
        self.settings["sensor_weights_pressure.json"]["include"] = self.PRESSURE_SENSORS_ENABLED
        self.settings["sensor_weights_isoflux.json"]["include"] = self.ISOFLUX_SENSORS_ENABLED
        # 3. Use the EFIT polynomial for p' and select the edge-pressure constraint.
        p_prime_settings = self.settings["source_function_p_prime.json"]
        p_prime_settings["method"] = "efit_polynomial"
        p_prime_settings["pressure_edge"]["free"] = self.PRESSURE_EDGE_FREE


class Gsfit_Ts_1(Gsfit_Ts):
    """Pressure-constrained GSFit with the edge pressure fixed to zero."""

    PRESSURE_EDGE_FREE = False


class Gsfit_Ts_2(Gsfit_Ts):
    """Pressure-constrained GSFit with a free edge pressure."""

    PRESSURE_EDGE_FREE = True


class Gsfit_Ts_3(Gsfit_Ts_1):
    """Isoflux-constrained GSFit with EFIT-polynomial p' and zero edge pressure."""

    PRESSURE_SENSORS_ENABLED = False
    ISOFLUX_SENSORS_ENABLED = True

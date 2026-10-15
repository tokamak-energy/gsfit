from typing import TYPE_CHECKING

from gsfit_rs import Pressure

from ...settings_schema.python.equilibrium.constraints_pressure import ConstraintsPressure as ConstraintsPressureSettings
from ...settings_schema.python.thomson_scattering.thomson_scattering import ThomsonScatteringSettings

# The Thomson scattering (TS) tree reading and the PPTS `BAD_MA` "good time-slice" logic live in the
# `st40_mdsplus_with_digital_filter` reader. It is reused here so that the pressure-sensor set-up is
# defined in a single place (it depends only on `constraints_pressure.xml` and `thomson_scattering.xml`,
# not on the digital filter), which keeps both ST40 readers consistent and avoids duplicated boiler-plate.
from ..st40_mdsplus_with_digital_filter.setup_pressure_sensors import get_good_pressure_sensor_times as get_good_pressure_sensor_times
from ..st40_mdsplus_with_digital_filter.setup_pressure_sensors import make_pressure_sensors

if TYPE_CHECKING:
    from ..st40_mdsplus_with_digital_filter import DatabaseReader


def setup_pressure_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: ThomsonScatteringSettings | None,
    settings_pressure_sensors: ConstraintsPressureSettings,
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class using ST40's Thomson scattering (TS) data.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `ThomsonScatteringSettings` object, read from `thomson_scattering/thomson_scattering.xml`, whose `st40_mdsplus` element says
        which PPTS and TS runs to read; None when `GSFIT_code_settings.xml` does not link one. Only needed when the pressure sensors are included
    :param settings_pressure_sensors: `ConstraintsPressure` object, read from `equilibrium/constraints_pressure.xml`, which says whether the pressure
        sensors are included, and holds their fitting settings

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Pressure sensors are opt-in, so the Thomson scattering is only read, and its runs only needed, when they are included
    if not settings_pressure_sensors.pressure.include:
        return Pressure()

    # This reader reads only its own element of `thomson_scattering.xml`
    settings_for_this_reader = settings.st40_mdsplus if settings is not None else None
    if settings_for_this_reader is None:
        raise ValueError(
            "`constraints_pressure.xml` includes the pressure sensors, but `thomson_scattering.xml` has no `st40_mdsplus` element, "
            "which says which PPTS and TS runs to read"
        )

    return make_pressure_sensors(
        ppts_pulseNo=pulseNo,
        ppts_run_name=settings_for_this_reader.ppts.run_name,
        ts_pulseNo=pulseNo,
        ts_run_name=settings_for_this_reader.ts.run_name,
        settings_pressure_sensors=settings_pressure_sensors,
    )

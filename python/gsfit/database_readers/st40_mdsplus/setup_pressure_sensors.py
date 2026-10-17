# The pressure-sensor set-up lives in the `st40_mdsplus_with_digital_filter` reader: the Thomson scattering (TS) tree reading,
# the PPTS `BAD_MA` "good time-slice" logic, and which PPTS and TS runs to read, from the `workflow.xml` which `thomson_scattering.xml` links to.
# It is reused here so that it is defined in a single place (it depends only on `constraints_pressure.xml` and `thomson_scattering.xml`,
# not on the digital filter), which keeps both ST40 readers consistent and avoids duplicated boiler-plate.
from ..st40_mdsplus_with_digital_filter.setup_pressure_sensors import get_good_pressure_sensor_times
from ..st40_mdsplus_with_digital_filter.setup_pressure_sensors import setup_pressure_sensors

# Define interfaces
__all__ = [
    "get_good_pressure_sensor_times",
    "setup_pressure_sensors",
]

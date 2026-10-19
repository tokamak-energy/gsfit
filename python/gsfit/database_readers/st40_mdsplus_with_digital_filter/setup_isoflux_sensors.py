from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from gsfit_rs import Isoflux

from ...settings_schema.python.equilibrium.constraints_isoflux import Isoflux as IsofluxSettings
from ...settings_schema.python.magnetics.magnetics import MagneticsSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_isoflux_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
    settings_isoflux_sensors: KeyedList[IsofluxSettings],
    times_to_reconstruct: npt.NDArray[np.float64],
) -> Isoflux:
    """
    This method initialises the Rust `Isoflux` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `MagneticsSettings` object, read from `magnetics/magnetics.xml`; not used
    :param settings_isoflux_sensors: The isoflux constraints, one `Isoflux` settings object each, read from `equilibrium/constraints_isoflux.xml`
    :param times_to_reconstruct: Times the equilibrium will be solved at [second]

    No isoflux constraints are made at present, as no settings say where to read their points from.
    So `constraints_isoflux.xml` should be empty, and any constraints it lists are not added.

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Isoflux Rust class
    isoflux = Isoflux()

    if len(settings_isoflux_sensors) == 0:
        return isoflux

    # TODO: need to check that this is working!
    # It also needs settings for the code and run to read the points from, `code_name` and `run_name`, which the XML settings do not have yet.
    # The JSON settings had them as `isoflux_code_name`, and its `run_name` in `workflow`, in the `database_reader` of `GSFIT_code_settings.json`
    # for isoflux_name in isoflux_settings.keys():
    #     # We need a try/except here because there is sometimes no TS data
    #     try:
    #         results = GetData(pulseNo, f"{code_name}#{run_name}")

    #         time = typing.cast(npt.NDArray[np.float32], results.get("TIME")).astype(np.float64)
    #         n_time = len(time)

    #         r = typing.cast(npt.NDArray[np.float32], results.get("R_MID_PROFILES.R")).astype(np.float64)
    #         te = typing.cast(npt.NDArray[np.float32], results.get("R_MID_PROFILES.TE")).astype(np.float64)

    #         location_1_r = np.full(n_time, np.nan)
    #         location_1_z = np.full(n_time, np.nan)
    #         location_2_r = np.full(n_time, np.nan)
    #         location_2_z = np.full(n_time, np.nan)

    #         for i_time in range(0, n_time):
    #             # Create LineString objects for the two lines
    #             te_profile_line = LineString(np.column_stack((r[i_time, :], te[i_time, :])))

    #             # Read in a LFS coordinate we want to use as the constant
    #             location_1_r_now = isoflux_settings[isoflux_name]["location_1"]["r"]
    #             location_1_z_now = 0.0  # Assumed for TS

    #             # Interpolate the Te profile to find Te at the LFS
    #             index_sorted = np.argsort(r[i_time, :])
    #             te_interpolated = np.interp(location_1_r_now, r[i_time, index_sorted], te[i_time, index_sorted])

    #             r_target_line = np.array([location_1_r_now, 200.0])
    #             te_target_line = np.array([te_interpolated, te_interpolated])
    #             line_to_interpolate = LineString(np.column_stack((r_target_line, te_target_line)))

    #             # Find the intersection
    #             intersections = te_profile_line.intersection(line_to_interpolate)

    #             # Convert `intersections` to list
    #             geoms = []
    #             if intersections.geom_type == "MultiPoint":
    #                 geoms = [geom for geom in intersections.geoms]  # TODO: causes mypy error??
    #             # if intersections.geom_type == 'Point' # there is only one intersection, so we haven't found both the LFS and HFS
    #             # if intersections.geom_type == 'MultiLineString':  # this only happens when the entire line is 0.0

    #             # Loop over all intersections
    #             location_2_r_now = np.nan
    #             location_2_z_now = np.nan
    #             if len(geoms) > 0:
    #                 for geom in geoms:
    #                     # Look for HFS intersection
    #                     if geom.x > location_1_r_now:
    #                         location_2_r_now = geom.x
    #                         location_2_z_now = 0.0

    #             # Store the isoflux coordinates
    #             if not np.isnan(location_2_r_now):
    #                 location_1_r[i_time] = location_1_r_now
    #                 location_1_z[i_time] = location_1_z_now
    #                 location_2_r[i_time] = location_2_r_now
    #                 location_2_z[i_time] = location_2_z_now

    #         isoflux.add_sensor(
    #             name=isoflux_name,
    #             fit_settings_comment="",
    #             fit_settings_include=True,
    #             fit_settings_weight=1.0e3,
    #             time=time,
    #             location_1_r=location_1_r,
    #             location_1_z=location_1_z,
    #             location_2_r=location_2_r,
    #             location_2_z=location_2_z,
    #             times_to_reconstruct=times_to_reconstruct,
    #         )
    #     except Exception as e:
    #         print(f"Error reading {isoflux_name}; exception={e}")

    return isoflux

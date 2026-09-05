"""Initialise a `Wall` object and read the limiter back out.

The `Wall` object holds an IMAS `wall` IDS. Only the limiter is filled so far:

    wall/description_2d(0)/limiter/unit(i)/outline/r    [metre]
    wall/description_2d(0)/limiter/unit(i)/outline/z    [metre]

This is the contour the data dictionary designates for equilibrium codes: "Description of
the immobile limiting surface(s) or plasma facing components for defining the Last Closed
Flux Surface".

Each unit is one plasma facing component, and every point of every unit is a candidate limit
point. **The order matters**: `unit(0)` is the vacuum vessel contour, and the solver uses
that one, and only that one, as the region the plasma is allowed to occupy.

The outlines below are illustrative, not a machine description. A real one is read from the
ELMAG tree; see `python/gsfit/database_readers/*/setup_wall.py`.
"""

import numpy as np
from gsfit_rs import Wall
from gsfit_rs.imas import wall_paths

# A simple closed D-shaped contour, standing in for a real vacuum vessel.
# The data dictionary asks for the first point to be repeated to close the contour.
n_points = 60
theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

major_radius = 0.5  # [metre]
minor_radius = 0.3  # [metre]
elongation = 1.8  # [dimensionless]
triangularity = 0.4  # [dimensionless]

vacuum_vessel_r = major_radius + minor_radius * np.cos(theta + triangularity * np.sin(theta))
vacuum_vessel_z = minor_radius * elongation * np.sin(theta)

# Repeat the first point, so the contour is closed
vacuum_vessel_r = np.append(vacuum_vessel_r, vacuum_vessel_r[0])
vacuum_vessel_z = np.append(vacuum_vessel_z, vacuum_vessel_z[0])

# Tiles which protrude inside the vessel contour. These are single points, not contours,
# which is why they are separate units rather than extra points on the vessel outline
mct_tiles_r = np.array([0.7103])  # [metre]
mct_tiles_z = np.array([0.3031])  # [metre]
mcb_tiles_r = np.array([0.7103])  # [metre]
mcb_tiles_z = np.array([-0.3131])  # [metre]

# Construct the wall. The vacuum vessel must be added first
wall = Wall()
wall.add_limiter_unit(name="vacuum_vessel", r=vacuum_vessel_r, z=vacuum_vessel_z)
wall.add_limiter_unit(name="mct_tiles", r=mct_tiles_r, z=mct_tiles_z)
wall.add_limiter_unit(name="mcb_tiles", r=mcb_tiles_r, z=mcb_tiles_z)

print(wall)

# Data is read back only through its data dictionary path, exactly as the equilibrium IDS
# is read. A path holds no data, so it can be built once, stored, and reused.
lim_r = wall.wall_ids.get(wall_paths.description_2d[0].limiter.unit[0].outline.r)
lim_z = wall.wall_ids.get(wall_paths.description_2d[0].limiter.unit[0].outline.z)

print()
print(f"{wall_paths.description_2d[0].limiter.unit[0].outline.r!r}  shape={lim_r.shape}")
print(f"{wall_paths.description_2d[0].limiter.unit[0].outline.z!r}  shape={lim_z.shape}")
print()
print(f"r range = [{lim_r.min():.3f}, {lim_r.max():.3f}] metre")
print(f"z range = [{lim_z.min():.3f}, {lim_z.max():.3f}] metre")
print(f"contour is closed = {bool(lim_r[0] == lim_r[-1] and lim_z[0] == lim_z[-1])}")

# The outline is stored, not merely echoed: what comes back is what went in
assert np.array_equal(lim_r, vacuum_vessel_r)
assert np.array_equal(lim_z, vacuum_vessel_z)

# A slice reads every unit at once. The units have different numbers of points, so the
# result is padded with NaN out to the longest one
unit_names = wall.wall_ids.get(wall_paths.description_2d[0].limiter.unit[:].name)
all_units_r = wall.wall_ids.get(wall_paths.description_2d[0].limiter.unit[:].outline.r)

print()
print(f"unit names = {unit_names}")
print(f"all units, outline r; shape = {all_units_r.shape}")
print(f"points per unit = {[int(np.count_nonzero(~np.isnan(all_units_r[i_unit, :]))) for i_unit in range(all_units_r.shape[0])]}")

# A path carries the data dictionary's own description and units
outline_r_path = wall_paths.description_2d[0].limiter.unit[0].outline.r
print()
print(f"units        : {outline_r_path.units}")
print(f"data_type    : {outline_r_path.data_type}")
print(f"documentation: {outline_r_path.documentation}")

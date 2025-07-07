from ..interface import DatabaseReaderProtocol
from .setup_bp_probes import setup_bp_probes
from .setup_coils import setup_coils
from .setup_dialoop import setup_dialoop
from .setup_flux_loops import setup_flux_loops
from .setup_isoflux_boundary_sensors import setup_isoflux_boundary_sensors
from .setup_isoflux_sensors import setup_isoflux_sensors
from .setup_passives import setup_passives
from .setup_plasma import setup_plasma
from .setup_rogowski_coils import setup_rogowski_coils


class DatabaseReaderST40AstraMDSplus(DatabaseReaderProtocol):
    setup_bp_probes = lambda *args, **kwargs: setup_bp_probes(*args, **kwargs)
    setup_coils = lambda *args, **kwargs: setup_coils(*args, **kwargs)
    setup_dialoop = lambda *args, **kwargs: setup_dialoop(*args, **kwargs)
    setup_flux_loops = lambda *args, **kwargs: setup_flux_loops(*args, **kwargs)
    setup_isoflux_boundary_sensors = lambda *args, **kwargs: setup_isoflux_boundary_sensors(*args, **kwargs)
    setup_isoflux_sensors = lambda *args, **kwargs: setup_isoflux_sensors(*args, **kwargs)
    setup_passives = lambda *args, **kwargs: setup_passives(*args, **kwargs)
    setup_plasma = lambda *args, **kwargs: setup_plasma(*args, **kwargs)
    setup_rogowski_coils = lambda *args, **kwargs: setup_rogowski_coils(*args, **kwargs)

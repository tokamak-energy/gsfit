import numpy as np

from . import BpProbes
from . import Coils

def run():
    # Make a Helmholtz PF coil
    coils = Coils()
    coils.add_pf_coil(
        name='helmholtz_01',
        r=np.array([1.123, 1.223]),
        z=np.array([-1.123, 1.223]),
        d_r=np.array([0.0, 0.0]),
        d_z=np.array([0.0, 0.0]),
        time=np.array([0.0, 1.0]),
        measured=np.array([1.0e3, 1.0e3]),
    )

    # Add some Bp probes
    bp_probes = BpProbes()
    # bp_probes.add_sensor(
    #     name='bp_01',
    #     r=1.0,
    #     z=0.0,
    # )
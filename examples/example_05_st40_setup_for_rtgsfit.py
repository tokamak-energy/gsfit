import numpy as np
from gsfit import Gsfit

pulseNo = 13343  # sed_tag:pulse_num_replay
pulseNo_write = 99_000_230
run_name = "RUN01"  # sed_tag:run_name_replay

# Construct the GSFit object; using the "st40_setup_for_rtgsfit" settings
gsfit_controller = Gsfit(
    pulseNo=pulseNo,
    run_name=run_name,
    run_description="Using a single degree of freedom for p_prime and ff_prime; and 15 eigenmodes for the IVC",
    write_to_mds=True,
    pulseNo_write=pulseNo_write,
    settings_path="st40_setup_for_rtgsfit",
)

# Change the analysis_name, so that GSFit writes into RT-GSFit MDSplus tree
gsfit_controller.analysis_name = "RTGSFIT"

# Add a list of signals to be read using PCS formatting
gsfit_controller.results["PRESHOT"]["COIL_SIGNALS"] = np.array(
    [
        "I_BVL_PSU",
        "I_BVUB_PSU",
        "I_BVUT_PSU",
        "I_DIV_PSU",
        "I_MCVC_PSU",
        "I_PSH_PSU",
        "I_ROG_MCWIRE",
        "I_SOL_PSU",
    ]
)
# fmt: off
coil_matrix = np.array(
    [
        # BVL_PSU, BVUB_PSU, BVUT_PSU, DIV_PSU, MCVC_PSU, PSH_PSU, ROG_MCWIRE, SOL_PSU
        [0.0,      0.0,      0.0,      0.0,     0.0,      0.0,     0.0,        1.0],  # SOL coil
        [0.0,      0.0,      0.0,      0.0,     0.0,      0.0,     1.0,        0.0],  # MCT coil
        [0.0,      0.0,      0.0,      0.0,     1.0,      0.0,     1.0,        0.0],  # MCB coil
        [0.0,      0.0,      0.0,      1.0,     0.0,      0.0,     0.0,        0.0],  # DIV coil
        [1.0,      0.0,      0.0,      0.0,     0.0,      0.0,     0.0,        0.0],  # BVL coil
        [0.0,      0.0,      1.0,      0.0,     0.0,      0.0,     0.0,        0.0],  # BVUT coil
        [0.0,      1.0,      0.0,      0.0,     0.0,      0.0,     0.0,        0.0],  # BVUB coil
        [0.0,      0.0,      0.0,      0.0,     0.0,      0.0,     0.0,        0.0],  # PSH coil
    ]
)
# fmt: on
# Transpose as this is used by Matlab/Simulink which is column-major
coil_matrix = coil_matrix.T
gsfit_controller.results["PRESHOT"]["COIL_MATRIX"] = coil_matrix

# Run
gsfit_controller.run()

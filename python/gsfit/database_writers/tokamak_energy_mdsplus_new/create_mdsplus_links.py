import mdsthin
import numpy as np


def create_mdsplus_links(pulseNo_write: int, run_name: str) -> None:
    """
    Add the "ALL" nodes for the "CONSTRAINTS"

    :param pulseNo_write: Pulse number to write to
    :param run_name: Run name, e.g. "TEST07"
    """

    conn = mdsthin.Connection("smaug")
    conn.openTree("gsfit", pulseNo_write)

    # Add signals
    sensor_types = ["BP_PROBE", "FLUX_LOOP", "ROGOWSKI", "PF_CURRENT", "PRESSURE"]
    signal_names = ["MEASURED", "RECONSTRUCT"]
    for sensor_type in sensor_types:
        tdi_command = f"getnci('\\\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.*', 'node_name')"
        sensor_names_on_mds_np = conn.get(tdi_command).data()
        sensor_names_on_mds = np.char.strip(sensor_names_on_mds_np).astype(str, copy=False).tolist()
        sensor_names_on_mds.remove("ALL")

        for signal_name in signal_names:
            tdi_command = "Build_Signal(TRANSPOSE(["
            for sensor_name in sensor_names_on_mds:
                tdi_command += f'DATA(GETNCI("\\\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.{sensor_name}:{signal_name}", "record")), '
            tdi_command = (
                tdi_command[:-2]  # remove the last comma and space
                + "]), "  # closing TRANSPOSE
                + "*, "  # no data in "raw"
                + f'DATA(GETNCI("\\\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.ALL:NAMES", "record")), DATA(GETNCI("\\\\GSFIT::TOP.{run_name}:TIME", "record"))'
                + ")"
            )
            conn.put(f"\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.ALL:{signal_name}", tdi_command)

        conn.put(f"\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.ALL:NAMES", "$1", np.array(sensor_names_on_mds))

    sensor_types = ["BP_PROBE", "FLUX_LOOP", "ROGOWSKI", "PF_CURRENT", "PRESSURE"]
    signal_names = ["INCLUDE", "WEIGHT"]
    for sensor_type in sensor_types:
        tdi_command = f"getnci('\\\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.*', 'node_name')"
        sensor_names_on_mds_np = conn.get(tdi_command).data()
        sensor_names_on_mds = np.char.strip(sensor_names_on_mds_np).astype(str, copy=False).tolist()
        sensor_names_on_mds.remove("ALL")

        for signal_name in signal_names:
            tdi_command = "Build_Signal(["
            for sensor_name in sensor_names_on_mds:
                tdi_command += f'DATA(GETNCI("\\\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.{sensor_name}:{signal_name}", "record")), '
            tdi_command = (
                tdi_command[:-2]  # remove the last comma and space
                + "], "  # closing TRANSPOSE
                + "*, "  # no data in "raw"
                + f'DATA(GETNCI("\\\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.ALL:NAMES", "record")), DATA(GETNCI("\\\\GSFIT::TOP.{run_name}:TIME", "record"))'
                + ")"
            )
            conn.put(f"\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.ALL:{signal_name}", tdi_command)

    # Add NAMES
    sensor_types = ["BP_PROBE", "FLUX_LOOP", "ROGOWSKI", "PF_CURRENT", "PRESSURE"]
    for sensor_type in sensor_types:
        tdi_command = f"getnci('\\\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.*', 'node_name')"
        sensor_names_on_mds_np = conn.get(tdi_command).data()
        sensor_names_on_mds = np.char.strip(sensor_names_on_mds_np).astype(str, copy=False).tolist()
        sensor_names_on_mds.remove("ALL")

        conn.put(f"\\GSFIT::TOP.{run_name}.CONSTRAINTS.{sensor_type}.ALL:NAMES", "$1", np.array(sensor_names_on_mds))

    # Close connection
    conn.closeAllTrees()

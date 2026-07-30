import json
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Files are named `{FILE_PREFIX}_{tree_name}_{pulseNo}.npz`, with the tree name lowercased
FILE_PREFIX = "mdsplus_mock"

# Archive member holding the capture provenance; not an MDSplus node
METADATA_KEY = "__metadata__"


class MockGetData:
    """
    Drop-in replacement for `st40_database.GetData`, backed by mocked MDSplus tree files.

    Each (tree, pulse) which the `st40_mdsplus` reader would have fetched over the network is
    mocked by a single `.npz`, mirroring how MDSplus itself stores one tree per shot:
        `mdsplus_mock_mag_12050.npz`

    Each archive member is named for the bare node path (e.g. `BPPROBE.P101.B`), because the
    tree and pulse are already carried by the filename. Capture provenance is stored inside the
    same archive under `__metadata__`, so it cannot drift away from the data it describes; read
    it with `get_metadata()`, or from a shell with:
        python -c "import numpy as np; print(np.load('<file>.npz')['__metadata__'])"

    Which tree, pulse and run each read maps to comes from the `workflow` settings, exactly as
    it does for the live `st40_mdsplus` reader; use `from_workflow(...)` rather than naming a
    tree directly. The requested run is checked against the run the mock was captured from, so
    asking for a run which was never captured is an error rather than silently wrong data.

    The files are produced by `investigation/capture_snapshot.py`.
    """

    # Each tree file is read once and shared by all readers
    _cache: dict[tuple[str, int, str], typing.Any] = {}

    def __init__(self, mock_dir: str, pulseNo: int, tree_run: str) -> None:
        """
        :param mock_dir: Directory containing the `mdsplus_mock_*.npz` files
        :param pulseNo: Pulse number, used to select which mocked tree file to read
        :param tree_run: MDSplus tree name, optionally with a `#run_name` suffix
        """

        self.pulseNo = pulseNo
        self.tree_name = tree_run.split("#")[0]
        self.run_name = tree_run.split("#")[1] if "#" in tree_run else ""
        self._nodes = MockGetData._load(mock_dir, pulseNo, self.tree_name)

        # Guard against reading a mock which was captured from a different run
        captured_run_name = self.get_metadata()["run_name"]
        if self.run_name != "" and captured_run_name != "" and self.run_name != captured_run_name:
            raise ValueError(
                f"mock_st40_mdsplus: {self.tree_name}#{self.run_name} was requested for pulseNo={pulseNo}, "
                f"but the mock was captured from {self.tree_name}#{captured_run_name}"
            )

    @classmethod
    def from_workflow(cls, settings: dict[str, typing.Any], pulseNo: int, workflow_name: str) -> "MockGetData":
        """
        Open the mocked tree which `workflow_name` maps to.

        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param pulseNo: The shot's pulse number, used when the workflow entry does not pin its own
        :param workflow_name: Key into `["database_reader"]["mock_st40_mdsplus"]["workflow"]`
        """

        reader_settings = settings["GSFIT_code_settings.json"]["database_reader"]["mock_st40_mdsplus"]
        workflow = reader_settings["workflow"]

        if workflow_name not in workflow:
            raise KeyError(f"mock_st40_mdsplus: no `workflow` entry named '{workflow_name}'; the settings define {sorted(workflow)}")
        workflow_entry = workflow[workflow_name]

        # `pulseNo = null` means "use the shot's pulse"; the machine-description reads pin their own
        tree_pulse_no = pulseNo if workflow_entry["pulseNo"] is None else workflow_entry["pulseNo"]

        return cls(str(reader_settings["mock_dir"]), tree_pulse_no, f"{workflow_entry['tree_name']}#{workflow_entry['run_name']}")

    @classmethod
    def _load(cls, mock_dir: str, pulseNo: int, tree_name: str) -> typing.Any:
        cache_key = (mock_dir, pulseNo, tree_name)
        if cache_key not in cls._cache:
            npz_path = Path(mock_dir) / f"{FILE_PREFIX}_{tree_name.lower()}_{pulseNo}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"mock_st40_mdsplus: no mocked MDSplus tree for pulseNo={pulseNo}, tree={tree_name}; expected {npz_path}")
            cls._cache[cache_key] = np.load(npz_path)
        return cls._cache[cache_key]

    def get(self, node: str) -> typing.Any:
        """
        :param node: MDSplus node path, relative to the top of the tree, e.g. `"BPPROBE.ALL.NAMES"`
        """

        if node == METADATA_KEY or node not in self._nodes:
            raise KeyError(f"mock_st40_mdsplus: node not found in mocked tree {self.tree_name}#{self.pulseNo}: {node}")

        value = typing.cast(npt.NDArray[typing.Any], self._nodes[node])

        # `st40_database.GetData` hands back Python lists of `str` for string nodes, never numpy
        # arrays, so match that here. `.tolist()` also un-nests, e.g. for `PF.ALL.COILS`
        if value.dtype.kind == "U":
            return value.tolist()

        return value

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return the provenance recorded when this tree was captured."""

        return typing.cast(dict[str, typing.Any], json.loads(str(self._nodes[METADATA_KEY])))

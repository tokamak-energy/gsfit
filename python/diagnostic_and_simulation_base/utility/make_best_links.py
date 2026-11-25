import numpy as np
import numpy.typing as npt

# The node usages which `BEST` can copy: MDSplus's usage number -> the name TCL's `add node /usage=` takes
_USAGE_NAMES: dict[int, str] = {1: "STRUCTURE", 5: "NUMERIC", 6: "SIGNAL", 8: "TEXT", 11: "SUBTREE"}

# The usages whose nodes hold data, so are linked; a structure or subtree only holds other nodes
_USAGE_NAMES_WITH_DATA: list[str] = ["NUMERIC", "SIGNAL", "TEXT"]


def make_best_links(server: str, tree_name: str, pulseNo: int, run_name: str) -> None:
    """Make `BEST` of `tree_name` at `pulseNo` a link to its run `run_name`, deleting the `BEST` there was.

    `BEST` is made as `make_BEST_links2` in `st40_mds_trees` makes it. It has every node the run has, at the same path
    below it. Each data node holds `LINK_TO_BEST(GETNCI("\\<tree>::TOP.BEST:BEST_RUN", "NID_NUMBER"), "<path below the
    run>")`, and `BEST:HELP` holds `LINK_TO_BEST_HELP(...)`: TDI functions on the server which read the node of
    whichever run the text node `BEST:BEST_RUN` names. So changing `BEST:BEST_RUN` re-points `BEST`, with no linking.

    An existing `BEST` is deleted first, because MDSplus nodes are only ever added: re-linking it in place would keep
    the nodes of every run it was linked to before which this run does not have.

    The run's nodes must exist, but need not hold data yet: `BEST` reads them only when it is itself read.

    :param server: MDSplus server, e.g. "smaug"
    :param tree_name: e.g. "GSFIT"
    :param pulseNo: pulse number
    :param run_name: the run to link `BEST` to, e.g. "RUN04"
    :raises RuntimeError: if `BEST` was not made as it should have been
    """

    # Lazy loading of `mdsthin`, because only a code writing to MDSplus needs it
    import mdsthin

    tree_root: str = f"\\{tree_name}::TOP"
    run_root: str = f"{tree_root}.{run_name}"
    best_root: str = f"{tree_root}.BEST"
    tree_root_in_tdi_string: str = tree_root.replace("\\", "\\\\")
    run_root_in_tdi_string: str = run_root.replace("\\", "\\\\")
    best_root_in_tdi_string: str = best_root.replace("\\", "\\\\")
    best_run_nid: str = f'GETNCI("{best_root_in_tdi_string}:BEST_RUN", "NID_NUMBER")'

    with mdsthin.Connection(server) as conn:
        conn.openTree(tree_name, pulseNo)

        # Every node of the run, by its path below the run, e.g. ".GLOBAL:IP" or ":TIME", in tree order, so that each
        # node comes after the node it is below. The first is the run itself
        run_fullpaths: npt.NDArray[np.str_] = np.atleast_1d(conn.get(f"getnci('{run_root_in_tdi_string}***', 'FULLPATH')").data())
        run_usages: npt.NDArray[np.int32] = np.atleast_1d(conn.get(f"getnci('{run_root_in_tdi_string}***', 'USAGE')").data())
        n_node_run: int = len(run_fullpaths)
        node_paths: list[str] = []
        node_usage_names: list[str] = []
        for i_node in range(1, n_node_run):
            node_paths.append(str(run_fullpaths[i_node]).strip()[len(run_root) :])
            node_usage: int = int(run_usages[i_node])
            if node_usage not in _USAGE_NAMES:
                raise RuntimeError(f"{run_root}{node_paths[-1]} has usage {node_usage}, which BEST cannot copy")
            node_usage_names.append(_USAGE_NAMES[node_usage])
        n_node: int = len(node_paths)

        # The old `BEST`, deleted in an edit of its own, as a node cannot be added in the edit which deletes it
        tree_node_names: list[str] = np.char.strip(np.atleast_1d(conn.get(f"getnci('{tree_root_in_tdi_string}.*', 'NODE_NAME')").data()).astype(str)).tolist()
        if "BEST" in tree_node_names:
            conn.tcl(f"edit {tree_name}/shot={pulseNo}")
            conn.tcl(f"delete node {best_root} /CONFIRM")
            conn.tcl("write")
            conn.tcl("close")

        # The nodes of the new `BEST`, in one edit. TCL paths are relative to `TOP`
        conn.tcl(f"edit {tree_name}/shot={pulseNo}")
        conn.tcl("add node .BEST /usage=STRUCTURE")
        conn.tcl("add node .BEST:BEST_RUN /usage=TEXT")
        conn.tcl("add node .BEST:BEST_RUN:HELP /usage=TEXT")
        for i_node in range(n_node):
            conn.tcl(f"add node .BEST{node_paths[i_node]} /usage={node_usage_names[i_node]}")
        conn.tcl("write")
        conn.tcl("close")

    # The links, in the tree opened afresh, to see the new nodes
    with mdsthin.Connection(server) as conn:
        conn.openTree(tree_name, pulseNo)
        conn.put(f"{best_root}:BEST_RUN", "$", run_name)
        conn.put(f"{best_root}:BEST_RUN:HELP", "$", 'Link to the "BEST" run')
        for i_node in range(n_node):
            if node_usage_names[i_node] not in _USAGE_NAMES_WITH_DATA:
                continue
            if node_paths[i_node] == ":HELP":
                conn.put(f"{best_root}:HELP", f'LINK_TO_BEST_HELP({best_run_nid}, ":HELP")')
            else:
                conn.put(f"{best_root}{node_paths[i_node]}", f'LINK_TO_BEST({best_run_nid}, "{node_paths[i_node]}")')

    # TCL does not raise when a command fails, so check that `BEST` has a node for every node of the run, names the
    # run, and that every one of its data nodes holds a link
    with mdsthin.Connection(server) as conn:
        conn.openTree(tree_name, pulseNo)
        best_usages: npt.NDArray[np.int32] = np.atleast_1d(conn.get(f"getnci('{best_root_in_tdi_string}***', 'USAGE')").data())
        best_lengths: npt.NDArray[np.int32] = np.atleast_1d(conn.get(f"getnci('{best_root_in_tdi_string}***', 'LENGTH')").data())
        best_run: str = str(conn.get(f"{best_root}:BEST_RUN").data()).strip()
    # `BEST` has the run's nodes, apart from the run itself, plus itself, `BEST_RUN` and its `HELP`
    n_node_best_expected: int = n_node + 3
    if len(best_usages) != n_node_best_expected:
        raise RuntimeError(f"{best_root} of {pulseNo} has {len(best_usages)} nodes, but should have {n_node_best_expected}")
    if best_run != run_name:
        raise RuntimeError(f"{best_root}:BEST_RUN of {pulseNo} is {best_run!r}, but should be {run_name!r}")
    n_node_best: int = len(best_usages)
    n_node_without_link: int = 0
    for i_node_best in range(n_node_best):
        if _USAGE_NAMES.get(int(best_usages[i_node_best])) in _USAGE_NAMES_WITH_DATA and int(best_lengths[i_node_best]) == 0:
            n_node_without_link += 1
    if n_node_without_link > 0:
        raise RuntimeError(f"{n_node_without_link} data nodes of {best_root} of {pulseNo} hold no link")

from ..settings_schema.python.gsfit_code_settings import GsfitCodeSettings
from ..settings_schema.python.workflow import Workflow
from ..settings_schema.python.workflow import WorkflowTree


def workflow_from_settings(settings: GsfitCodeSettings) -> dict[str, WorkflowTree]:
    """
    The MDSplus trees ("workflow") which the `database_reader` reads from, with which run of each, by tree name,
    for example `{"ELMAG": WorkflowTree(...), "MAG": WorkflowTree(...), "PSU2COIL": WorkflowTree(...)}`.

    They are the trees of the `workflow.xml` which each IDS's file links to, e.g. `pf_active/pf_active.xml`.
    The IDSs' files normally all link to the same `workflow.xml`. If they link to different ones, a tree in several must have the same
    run in each, as `INPUT.WORKFLOW` holds one run per tree. An IDS's file which links to no `workflow.xml` adds no trees.

    These are what `map_results_to_database` stores under `INPUT.WORKFLOW`, and what the MDSplus `INPUT.WORKFLOW` nodes have to be
    created for. They come from the settings alone, so they are known before GSFit has read a single byte from the database.

    :param settings: `GsfitCodeSettings` object containing the XML settings read from the `settings` directory
    :return: Each tree's run, by its `tree_name`, e.g. "ELMAG", which also names its node under `INPUT.WORKFLOW`
    """

    # The `workflow.xml` which each IDS's file links to, by IDS; None for an IDS's file which links to none, or an IDS which is not built
    workflows_by_ids_name: dict[str, Workflow | None] = {
        "magnetics": settings.ids.magnetics.workflow,
        "pf_active": settings.ids.pf_active.workflow,
        "pf_passive": settings.ids.pf_passive.workflow,
        "pulse_schedule": settings.ids.pulse_schedule.workflow if settings.ids.pulse_schedule is not None else None,
        "tf": settings.ids.tf.workflow,
        "thomson_scattering": settings.ids.thomson_scattering.workflow if settings.ids.thomson_scattering is not None else None,
        "wall": settings.ids.wall.workflow,
    }

    workflow_trees_by_tree_name: dict[str, WorkflowTree] = {}
    ids_names_by_tree_name: dict[str, str] = {}  # the first IDS whose `workflow.xml` holds each tree, for the error message
    for ids_name, workflow in workflows_by_ids_name.items():
        if workflow is None:
            continue
        for workflow_tree in workflow.tree:
            tree_name: str = workflow_tree.tree_name
            if tree_name not in workflow_trees_by_tree_name:
                workflow_trees_by_tree_name[tree_name] = workflow_tree
                ids_names_by_tree_name[tree_name] = ids_name
            elif workflow_trees_by_tree_name[tree_name] != workflow_tree:
                raise ValueError(
                    f"`{ids_names_by_tree_name[tree_name]}.xml` and `{ids_name}.xml` link to `workflow.xml`s with different {tree_name} `tree`s, "
                    f"{workflow_trees_by_tree_name[tree_name]} and {workflow_tree}, but `INPUT.WORKFLOW` holds one run per tree"
                )

    return workflow_trees_by_tree_name


def workflow_names_from_settings(settings: GsfitCodeSettings) -> list[str]:
    """
    The names of the MDSplus trees ("workflow") which the `database_reader` reads from, for example `["ELMAG", "MAG", "PSU2COIL"]`.

    These are the names which `map_results_to_database` stores under `INPUT.WORKFLOW`, and which
    the MDSplus `INPUT.WORKFLOW` nodes have to be created for; see `workflow_from_settings`.

    :param settings: `GsfitCodeSettings` object containing the XML settings read from the `settings` directory
    """

    return list(workflow_from_settings(settings).keys())

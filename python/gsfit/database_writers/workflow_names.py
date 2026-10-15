import dataclasses

from ..settings_schema.python.gsfit_code_settings import GsfitCodeSettings
from ..settings_schema.python.workflow import Workflow
from ..settings_schema.python.workflow import WorkflowEntry


def workflow_from_settings(settings: GsfitCodeSettings) -> dict[str, WorkflowEntry]:
    """
    The input codes ("workflow") which the `database_reader` reads from, with which run of each, by name,
    for example `{"ELMAG": WorkflowEntry(...), "MAG": WorkflowEntry(...), "PSU2COIL": WorkflowEntry(...)}`.

    They are the codes of the `workflow.xml` which each IDS's file links to, e.g. `pf_active/pf_active.xml`.
    The IDSs' files normally all link to the same `workflow.xml`. If they link to different ones, a code in several must have the same
    run in each, as `INPUT.WORKFLOW` holds one run per code. An IDS's file which links to no `workflow.xml` adds no codes.

    These are what `map_results_to_database` stores under `INPUT.WORKFLOW`, and what the MDSplus `INPUT.WORKFLOW` nodes have to be
    created for. They come from the settings alone, so they are known before GSFit has read a single byte from the database.

    :param settings: `GsfitCodeSettings` object containing the XML settings read from the `settings` directory
    :return: Each code's run, by the code's name in upper case, e.g. "ELMAG", which names its MDSplus tree, and its node under `INPUT.WORKFLOW`
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

    workflow_entries_by_code_name: dict[str, WorkflowEntry] = {}
    ids_names_by_code_name: dict[str, str] = {}  # the first IDS whose `workflow.xml` holds each code, for the error message
    for ids_name, workflow in workflows_by_ids_name.items():
        if workflow is None:
            continue
        # `Workflow` has one field per code, e.g. `elmag`, which is None when the code is not in `workflow.xml`
        for code_field in dataclasses.fields(workflow):
            workflow_entry: WorkflowEntry | None = getattr(workflow, code_field.name)
            if workflow_entry is None:
                continue
            code_name = code_field.name.upper()
            if code_name not in workflow_entries_by_code_name:
                workflow_entries_by_code_name[code_name] = workflow_entry
                ids_names_by_code_name[code_name] = ids_name
            elif workflow_entries_by_code_name[code_name] != workflow_entry:
                raise ValueError(
                    f"`{ids_names_by_code_name[code_name]}.xml` and `{ids_name}.xml` link to `workflow.xml`s with different `{code_field.name}`s, "
                    f"{workflow_entries_by_code_name[code_name]} and {workflow_entry}, but `INPUT.WORKFLOW` holds one run per code"
                )

    return workflow_entries_by_code_name


def workflow_names_from_settings(settings: GsfitCodeSettings) -> list[str]:
    """
    The names of the input codes ("workflow") which the `database_reader` reads from, for example `["ELMAG", "MAG", "PSU2COIL"]`.

    These are the names which `map_results_to_database` stores under `INPUT.WORKFLOW`, and which
    the MDSplus `INPUT.WORKFLOW` nodes have to be created for; see `workflow_from_settings`.

    :param settings: `GsfitCodeSettings` object containing the XML settings read from the `settings` directory
    """

    return list(workflow_from_settings(settings).keys())

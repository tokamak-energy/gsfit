"""
The Buxton-Prokopyszyn analytic solution's Python driver.

It reads the settings, `BUXTON_PROKOPYSZYN_code_settings.xml` and the XML files it links to, into dataclasses, which can be changed in
Python, e.g. for a scan, and then solves the analytic equilibrium with the Rust code, `gsfit_rs.analytic_grad_shafranov.buxton_prokopyszyn`,
as GSFit's driver does with `gsfit_rs.Equilibrium`:

    from analytic_grad_shafranov.buxton_prokopyszyn import BuxtonProkopyszyn

    buxton_prokopyszyn = BuxtonProkopyszyn(pulseNo=0, run_name="", settings_path="default", write_to_mds=False)
    buxton_prokopyszyn.settings_xml.ids.equilibrium.global_quantities.ip = 3.0e5
    equilibrium = buxton_prokopyszyn.run()
    print(equilibrium.gs_residual)

The settings' XML schemas, in `settings_schema/xsd/`, describe every setting; the dataclasses are generated from them.
"""

from diagnostic_and_simulation_base import DiagnosticAndSimulationBase
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs.analytic_grad_shafranov import buxton_prokopyszyn as buxton_prokopyszyn_rs
from gsfit_rs.dd_support import ParametersInput

from .settings_schema import XSD_DIRECTORY_PATH
from .settings_schema.python.buxton_prokopyszyn_code_settings import BuxtonProkopyszynCodeSettings


class BuxtonProkopyszyn(DiagnosticAndSimulationBase[BuxtonProkopyszynCodeSettings]):
    """
    The Buxton-Prokopyszyn analytic solution: its settings, and the analytic equilibrium solved from them.

    It is constructed as every `DiagnosticAndSimulationBase` code is; the analytic solution has no pulse, so `pulseNo=0, run_name=""`.
    `settings_path` is the settings directory, holding `BUXTON_PROKOPYSZYN_code_settings.xml`: the path of a directory, or the name of
    one of the directories in `settings/`, next to this module, e.g. "default".

    The settings are read when it is constructed, and can be changed in `settings_xml` until `run` is called, which solves the analytic
    equilibrium with the settings as they stand then. `run` can be called again after changing them, e.g. for a scan.
    The analytic solution has no numerical fallback: when its series does not converge, `run` keeps the partial sum with the smallest
    Grad-Shafranov residual and warns (`gs_residual` above 1e-6), and the equilibrium is one for a numerical Grad-Shafranov solver.
    """

    # The dataclass which `BUXTON_PROKOPYSZYN_code_settings.xml`, and the XML files it links to, are read into, as `self.settings_xml`.
    # They can be changed in Python before `run`, e.g. `buxton_prokopyszyn.settings_xml.ids.equilibrium.global_quantities.ip = 3.0e5`
    settings_xml_class = BuxtonProkopyszynCodeSettings

    # The analytic equilibrium solved by the last `run`, `None` until `run` has succeeded
    equilibrium: buxton_prokopyszyn_rs.BuxtonProkopyszyn | None = None

    def parameters_input(self) -> ParametersInput:
        """
        The equilibrium's settings, as they stand, as the Rust code takes them.

        :return: The IMAS `parameters_input` of `settings_xml.ids.equilibrium`: `equilibrium.xml` with each file it links to written in
            its link's place, checked against their XML schema, which the Equilibrium IDS keeps in `code/parameters`
        """

        parameters_input: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
            settings=self.settings_xml.ids.equilibrium,
            xsd_directory_path=XSD_DIRECTORY_PATH,
        )

        return parameters_input

    def valid_r_range(self) -> tuple[float, float]:
        """
        The radial extent where the analytic solution can be evaluated for the settings' shape, as they stand, which the grid,
        `settings_xml.ids.equilibrium.grid`, must lie in. Only the shape and the radial series' variable are used, so the grid can be
        chosen from it.

        :return: `(r_min, r_max)` [metre]
        """

        r_min: float
        r_max: float
        r_min, r_max = buxton_prokopyszyn_rs.BuxtonProkopyszyn.valid_r_range(self.parameters_input())

        return r_min, r_max

    def run(self) -> buxton_prokopyszyn_rs.BuxtonProkopyszyn:
        """
        Solve the analytic equilibrium with the settings as they stand, and post-process its Equilibrium IDS.

        :return: The analytic equilibrium, also kept in `equilibrium`
        :raises ValueError: The settings cannot be solved, e.g. a shape the configuration rejects
        :raises RuntimeError: The solve fails, e.g. the plasma current cannot set the coefficients' scale
        """

        # A failed run leaves no equilibrium, rather than the last one
        self.equilibrium = None

        equilibrium: buxton_prokopyszyn_rs.BuxtonProkopyszyn = buxton_prokopyszyn_rs.BuxtonProkopyszyn(self.parameters_input())
        equilibrium.solve()
        self.equilibrium = equilibrium

        return equilibrium

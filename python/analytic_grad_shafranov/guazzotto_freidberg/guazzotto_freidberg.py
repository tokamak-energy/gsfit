"""
The Guazzotto-Freidberg analytic solution's Python driver.

It reads the settings, `GUAZZOTTO_FREIDBERG_code_settings.xml` and the XML files it links to, into dataclasses, which can be changed in
Python, e.g. for a scan, and then solves the analytic equilibrium with the Rust code, `gsfit_rs.analytic_grad_shafranov.guazzotto_freidberg`,
as GSFit's driver does with `gsfit_rs.Equilibrium`:

    from analytic_grad_shafranov.guazzotto_freidberg import GuazzottoFreidberg

    guazzotto_freidberg = GuazzottoFreidberg(pulseNo=0, run_name="", settings_path="default", write_to_mds=False)
    guazzotto_freidberg.settings_xml.ids.equilibrium.profiles.nu = 1.0
    equilibrium = guazzotto_freidberg.run()
    print(equilibrium.alpha)

The free-boundary equilibrium, with the PF-coil currents which reproduce the analytic solution's vacuum field, then follows from
`equilibrium.get_pf_active`.

The settings' XML schemas, in `settings_schema/xsd/`, describe every setting; the dataclasses are generated from them.
"""

from diagnostic_and_simulation_base import DiagnosticAndSimulationBase
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs.analytic_grad_shafranov import guazzotto_freidberg as guazzotto_freidberg_rs
from gsfit_rs.dd_support import ParametersInput

from .settings_schema import XSD_DIRECTORY_PATH
from .settings_schema.python.guazzotto_freidberg_code_settings import GuazzottoFreidbergCodeSettings


class GuazzottoFreidberg(DiagnosticAndSimulationBase[GuazzottoFreidbergCodeSettings]):
    """
    The Guazzotto-Freidberg analytic solution: its settings, and the analytic equilibrium solved from them.

    It is constructed as every `DiagnosticAndSimulationBase` code is; the analytic solution has no pulse, so `pulseNo=0, run_name=""`.
    `settings_path` is the settings directory, holding `GUAZZOTTO_FREIDBERG_code_settings.xml`: the path of a directory, or the name of
    one of the directories in `settings/`, next to this module, e.g. "default".

    The settings are read when it is constructed, and can be changed in `settings_xml` until `run` is called, which solves the analytic
    equilibrium with the settings as they stand then. `run` can be called again after changing them, e.g. for a scan.
    """

    # The dataclass which `GUAZZOTTO_FREIDBERG_code_settings.xml`, and the XML files it links to, are read into, as `self.settings_xml`.
    # They can be changed in Python before `run`, e.g. `guazzotto_freidberg.settings_xml.ids.equilibrium.profiles.nu = 1.0`
    settings_xml_class = GuazzottoFreidbergCodeSettings

    # The analytic equilibrium solved by the last `run`, `None` until `run` has succeeded
    equilibrium: guazzotto_freidberg_rs.GuazzottoFreidberg | None = None

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

    def run(self) -> guazzotto_freidberg_rs.GuazzottoFreidberg:
        """
        Solve the analytic equilibrium with the settings as they stand: its eigenvalue, and its flux and plasma mask on the grid.
        The Equilibrium IDS is filled once the PF coils are fitted, by `get_pf_active` on the returned equilibrium.

        :return: The analytic equilibrium, also kept in `equilibrium`
        :raises ValueError: The settings cannot be solved, e.g. a model surface Guazzotto and Freidberg's construction rejects, or a grid
            which does not contain it
        :raises RuntimeError: The solve fails: the matching constraints are singular at the eigenvalue found
        """

        # A failed run leaves no equilibrium, rather than the last one
        self.equilibrium = None

        equilibrium: guazzotto_freidberg_rs.GuazzottoFreidberg = guazzotto_freidberg_rs.GuazzottoFreidberg(self.parameters_input())
        equilibrium.solve()
        self.equilibrium = equilibrium

        return equilibrium

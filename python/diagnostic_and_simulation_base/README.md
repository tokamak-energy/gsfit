# Diagnostic and simulation base
The `DiagnosticAndSimulationBase` abstract class provides a structured and consistent framework for developing diagnostic analysis and simulation codes.
All codes at Tokamak Energy that are part of the Post Pulse Analysis Chain (PPAC) use this standard.

This approach serves two main purposes:
1. **Traceability:** It ensures that all settings used for an analysis or simulation are automatically recorded, improving reliability and traceability.
2. **Consistency:** It standardizes how codes are initialised and executed, making it easier for new users to run any code. Additionally, it also simplifies the automation of the PPAC workflow.

Core to ensuring traceability and consistency is the principle of minimizing the number of input arguments required to initialize a child `DiagnosticAndSimulationBase` object.
Rather than passing many parameters directly, all settings are stored in configuration files—preferably `*.json`.
This approach reduces errors when running an analysis and makes it easier for new users to run any code.

# 1. Example usage

Below we show some examples of how to run codes which use the `DiagnosticAndSimulationBase`:

<table>
    <tr>
        <td style="width: 50%; vertical-align: top;">

### GSFit (equilibrium reconstruction)

```python
from gsfit import Gsfit

gsfit_controller = Gsfit(
        pulseNo=12050,
        run_name="RUN01",
        run_description="Default settings"
)

gsfit_controller.run()
```

</td>
        <td style="width: 50%; vertical-align: top;">

### ZeffBrems (Zeff from Bremsstrahlung radiation)

```python
from bolometry import BlomXY1

blom_xy1_controller = BlomXY1(
        pulseNo=12050,
        run_name="RUN01",
        run_description="Default settings"
)

blom_xy1_controller.run()
```

</td>
    </tr>
    <tr>
        <td style="width: 50%; vertical-align: top;">

### XRCS (X-ray crystal spectrometer analysis)

```python
from xrcs import Xrcs

xrcs_controller = Xrcs(
        pulseNo=12050,
        run_name="RUN01",
        run_description="Default settings"
)

xrcs_controller.run()
```

</td>
        <td style="width: 50%; vertical-align: top;">

### Dialoop (diamagnetic loop analysis)

```python
from dialoop import Dialoop

dialoop_controller = Dialoop(
        pulseNo=12050,
        run_name="RUN01",
        run_description="Default settings"
)

dialoop_controller.run()
```

</td>
    </tr>
</table>


Below is the code needed at a minimum to create a **new code** using `DiagnosticAndSimulationBase` child:
```python
from diagnostic_and_simulation_base import DiagnosticAndSimulationBase

class NewCodeName(DiagnosticAndSimulationBase):
    # Note: the `__init__()` method is inherited from `DiagnosticAndSimulationBase`

    def run(self):
        # Do some analysis
        # Write to database
        ...
```
# 2. All options to initialize a child

The following is the complete list of supported input arguments.

> **No additional options should be added to the class initializer.**

```python
new_code_name_controller = NewCodeName(
    pulseNo=12050,
    run_name="RUN01",
    run_description="Default settings",
    settings_path="default",
    write_to_database=True,
    pulseNo_write=12050,
    link_run_to_best=True,
)
```

## 2.1 `pulseNo`

<table>
    <tr>
        <td><strong>Type</strong></td>
        <td><code>int</code></td>
    </tr>
    <tr>
        <td><strong>Required/Optional</strong></td>
        <td>Required</td>
    </tr>
    <tr>
        <td><strong>Default value</strong></td>
        <td>N/A</td>
    </tr>
    <tr>
        <td><strong>Description</strong></td>
        <td>Pulse number to analyse or simulate.</td>
    </tr>
</table>

## 2.2 `run_name`

<table>
    <tr>
        <td><strong>Type</strong></td>
        <td><code>str</code></td>
    </tr>
    <tr>
        <td><strong>Required/Optional</strong></td>
        <td>Required</td>
    </tr>
    <tr>
        <td><strong>Default value</strong></td>
        <td>N/A</td>
    </tr>
    <tr>
        <td><strong>Description</strong></td>
        <td>Name of the run to save in the database.</td>
    </tr>
</table>

## 2.3 `run_description`

<table>
    <tr>
        <td><strong>Type</strong></td>
        <td><code>str</code></td>
    </tr>
    <tr>
        <td><strong>Required/Optional</strong></td>
        <td>Required</td>
    </tr>
    <tr>
        <td><strong>Default value</strong></td>
        <td>N/A</td>
    </tr>
    <tr>
        <td><strong>Description</strong></td>
        <td>Short description of the run. Used for documentation and traceability.</td>
    </tr>
</table>

## 2.4 `settings_path`

<table>
    <tr>
        <td><strong>Type</strong></td>
        <td><code>str</code></td>
    </tr>
    <tr>
        <td><strong>Required/Optional</strong></td>
        <td>Optional</td>
    </tr>
    <tr>
        <td><strong>Default value</strong></td>
        <td><code>"default"</code></td>
    </tr>
    <tr>
        <td><strong>Description</strong></td>
        <td>Path to the settings directory or file containing code inputs (e.g., a JSON file). This can either be a relative path or an absolute path. There should always be a directory called "default" which contains the settings which should be used in the Post Pulse Analysis Chain (PPAC). We can use multiple directories to store other machines, or different types of runs.</td>
    </tr>
</table>

## 2.5 `write_to_database`

<table>
    <tr>
        <td><strong>Type</strong></td>
        <td><code>bool</code></td>
    </tr>
    <tr>
        <td><strong>Required/Optional</strong></td>
        <td>Optional</td>
    </tr>
    <tr>
        <td><strong>Default value</strong></td>
        <td><code>True</code></td>
    </tr>
    <tr>
        <td><strong>Description</strong></td>
        <td>Whether to write results to the database. Set to <code>False</code> to skip database writing.</td>
    </tr>
</table>

## 2.6 `pulseNo_write`

<table>
    <tr>
        <td><strong>Type</strong></td>
        <td><code>None</code> &#124; <code>int</code></td>
    </tr>
    <tr>
        <td><strong>Required/Optional</strong></td>
        <td>Optional</td>
    </tr>
    <tr>
        <td><strong>Default value</strong></td>
        <td><code>None</code></td>
    </tr>
    <tr>
        <td><strong>Description</strong></td>
        <td>Pulse number to write data to, if different from <code>pulseNo</code>. Useful for modeling or scenario runs.</td>
    </tr>
</table>

## 2.7 `link_run_to_best`

<table>
    <tr>
        <td><strong>Type</strong></td>
        <td><code>None</code> &#124; <code>bool</code></td>
    </tr>
    <tr>
        <td><strong>Required/Optional</strong></td>
        <td>Optional</td>
    </tr>
    <tr>
        <td><strong>Default value</strong></td>
        <td><code>False</code></td>
    </tr>
    <tr>
        <td><strong>Description</strong></td>
        <td>Whether to link the current run to the "best" run in the database for traceability.</td>
    </tr>
</table>

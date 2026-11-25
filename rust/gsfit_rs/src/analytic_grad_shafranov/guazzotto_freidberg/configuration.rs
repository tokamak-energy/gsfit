use super::Symmetry;
use super::settings_schema::{
    GuazzottoFreidbergSettingsConfiguration, GuazzottoFreidbergSettingsDoubleNull, GuazzottoFreidbergSettingsSingleNull,
    GuazzottoFreidbergSettingsSymmetricLimited,
};

/// Plasma boundary topology, and the model surface's shape, size and position.
///
/// Each variant carries exactly its own inputs, so illegal combinations are unrepresentable. Every model surface is centred on
/// `(r_geo, z_geo)`, its innermost and outermost points being at `r_geo - a_minor` and `r_geo + a_minor`, on `Z = z_geo`. The up-down
/// symmetry (7- vs 12-term expansion) is implied by the variant and can be read via [`Configuration::symmetry`].
///
/// References are to L. Guazzotto & J. P. Freidberg, J. Plasma Phys. 87 (2021) 905870303.
#[derive(Clone, Debug)]
pub enum Configuration {
    /// Up–down symmetric smooth limiter (GF §3): 7-term `cos(h_n y)` series.
    SymmetricLimited {
        r_geo: f64,
        z_geo: f64,
        a_minor: f64,
        kappa: f64,
        delta: f64,
    },
    /// Up–down symmetric double-null divertor (GF §4): 7-term series with an X-point.
    DoubleNull {
        r_geo: f64,
        z_geo: f64,
        a_minor: f64,
        kappa_x: f64,
        delta_x: f64,
    },
    /// Up–down asymmetric lower single-null divertor (GF §5): 12-term series. The upper half is
    /// a smooth Miller profile (`kappa`, `delta`) and the lower half has the X-point (`kappa_x`, `delta_x`).
    SingleNull {
        r_geo: f64,
        z_geo: f64,
        a_minor: f64,
        kappa: f64,
        delta: f64,
        kappa_x: f64,
        delta_x: f64,
    },
}

impl Configuration {
    /// The configuration in the settings, `equilibrium_settings/configuration`: the model surface its `method` names, with the settings
    /// of the element of the same name.
    pub fn from_settings(settings: &GuazzottoFreidbergSettingsConfiguration) -> Result<Configuration, String> {
        let missing = |method: &str| -> String {
            format!(
                "`equilibrium_settings/configuration/method` is \"{method}\", but `equilibrium_settings/configuration/{method}`, which holds its shape, is missing"
            )
        };
        match settings.method.as_str() {
            "symmetric_limited" => {
                let shape: &GuazzottoFreidbergSettingsSymmetricLimited = settings.symmetric_limited.as_ref().ok_or_else(|| missing("symmetric_limited"))?;
                Ok(Configuration::SymmetricLimited {
                    r_geo: shape.r_geo,
                    z_geo: shape.z_geo,
                    a_minor: shape.a_minor,
                    kappa: shape.kappa,
                    delta: shape.delta,
                })
            }
            "double_null" => {
                let shape: &GuazzottoFreidbergSettingsDoubleNull = settings.double_null.as_ref().ok_or_else(|| missing("double_null"))?;
                Ok(Configuration::DoubleNull {
                    r_geo: shape.r_geo,
                    z_geo: shape.z_geo,
                    a_minor: shape.a_minor,
                    kappa_x: shape.kappa_x,
                    delta_x: shape.delta_x,
                })
            }
            "single_null" => {
                let shape: &GuazzottoFreidbergSettingsSingleNull = settings.single_null.as_ref().ok_or_else(|| missing("single_null"))?;
                Ok(Configuration::SingleNull {
                    r_geo: shape.r_geo,
                    z_geo: shape.z_geo,
                    a_minor: shape.a_minor,
                    kappa: shape.kappa,
                    delta: shape.delta,
                    kappa_x: shape.kappa_x,
                    delta_x: shape.delta_x,
                })
            }
            method => Err(format!("`equilibrium_settings/configuration/method` = \"{method}\" is not recognized")),
        }
    }

    /// Up–down symmetry implied by the topology (selects the 7- vs 12-term expansion).
    pub fn symmetry(&self) -> Symmetry {
        match self {
            Configuration::SymmetricLimited { .. } | Configuration::DoubleNull { .. } => Symmetry::Symmetric,
            Configuration::SingleNull { .. } => Symmetry::Asymmetric,
        }
    }

    /// Major radius midway between the model surface's innermost and outermost points, GF's `R_0` [metre]
    pub fn r_geo(&self) -> f64 {
        match *self {
            Configuration::SymmetricLimited { r_geo, .. } | Configuration::DoubleNull { r_geo, .. } | Configuration::SingleNull { r_geo, .. } => r_geo,
        }
    }

    /// Height of the model surface's innermost and outermost points, GF's `y = 0` [metre]
    pub fn z_geo(&self) -> f64 {
        match *self {
            Configuration::SymmetricLimited { z_geo, .. } | Configuration::DoubleNull { z_geo, .. } | Configuration::SingleNull { z_geo, .. } => z_geo,
        }
    }

    /// Minor radius, half the distance between the model surface's innermost and outermost points [metre]
    pub fn a_minor(&self) -> f64 {
        match *self {
            Configuration::SymmetricLimited { a_minor, .. } | Configuration::DoubleNull { a_minor, .. } | Configuration::SingleNull { a_minor, .. } => a_minor,
        }
    }

    /// GF's inverse aspect ratio, `eps = a_minor / r_geo` [dimensionless]
    pub fn eps(&self) -> f64 {
        self.a_minor() / self.r_geo()
    }
}

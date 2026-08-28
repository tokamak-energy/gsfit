from . import gsfit_rs
from .gsfit_rs import *

__doc__ = gsfit_rs.__doc__
if hasattr(gsfit_rs, "__all__"):
    __all__ = list(gsfit_rs.__all__)

# Analytic Grad-Shafranov equilibria are namespaced under
# gsfit_rs.analytic_grad_shafranov.<solver>, not the flat top level.
for _name in ("Configuration", "GuazzottoFreidberg", "Symmetry"):
    globals().pop(_name, None)
    if "__all__" in globals() and _name in __all__:
        __all__.remove(_name)

"""Buxton-Prokopyszyn analytic Grad-Shafranov equilibrium.

Re-exports the Rust-backed class under a namespace that mirrors the Rust module
tree (`analytic_grad_shafranov::buxton_prokopyszyn`). It is constructed from its
settings, which the Python driver, `analytic_grad_shafranov.buxton_prokopyszyn`,
reads from XML.
"""

from gsfit_rs.gsfit_rs import BuxtonProkopyszyn

__all__ = ["BuxtonProkopyszyn"]

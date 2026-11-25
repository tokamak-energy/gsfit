"""Guazzotto-Freidberg analytic Grad-Shafranov equilibrium.

Re-exports the Rust-backed class under a namespace that mirrors the Rust module
tree (`analytic_grad_shafranov::guazzotto_freidberg`). It is constructed from its
settings, which the Python driver, `analytic_grad_shafranov.guazzotto_freidberg`,
reads from XML.
"""

from gsfit_rs.gsfit_rs import GuazzottoFreidberg

__all__ = ["GuazzottoFreidberg"]

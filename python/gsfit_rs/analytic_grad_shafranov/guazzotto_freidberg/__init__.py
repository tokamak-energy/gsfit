"""Guazzotto-Freidberg analytic Grad-Shafranov equilibrium.

Re-exports the Rust-backed classes under a namespace that mirrors the Rust module
tree (`analytic_grad_shafranov::guazzotto_freidberg`), so the solver-specific
`Configuration` does not sit on the flat `gsfit_rs` namespace.
"""

from gsfit_rs.gsfit_rs import Configuration, GuazzottoFreidberg, Symmetry

__all__ = ["Configuration", "GuazzottoFreidberg", "Symmetry"]

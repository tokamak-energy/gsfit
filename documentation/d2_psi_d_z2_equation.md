# The equation for $\partial^2\psi/\partial z^2$ of a toroidal current source

This is the complete analytic expression implemented in
[`Greens::d2_psi_d_z2`](../rust/gsfit_rs/src/greens/greens.rs), per unit source
current $I$, for a sensor at $(r, z)$ and a toroidal current source at
$(r', z')$ with rectangular cross-section $\Delta r \times \Delta z$.
The derivation is in
[`jump_condition_dbr_dz.md`](jump_condition_dbr_dz.md).

$$
\frac{1}{I}\,\frac{\partial^2 \psi}{\partial z^2}
=
\begin{cases}
\;-\dfrac{\mu_0}{d^2\, u^3}
\left\{
\left[\,v^2 u^2 \;-\; h^2\!\left(d^2 + \dfrac{4\, r\, r'\, u^2}{d^2}\right)\right] E(m)
\;+\;
\left(h^2 v^2 \;-\; u^2 d^2\right) K(m)
\right\},
& (r, z) \neq (r', z'),
\\[3ex]
\;-\dfrac{2\pi \mu_0\, r\, F(\lambda)}{\Delta r\, \Delta z},
& (r, z) = (r', z'),
\end{cases}
$$

with

$$
h = z - z', \qquad
d^2 = (r - r')^2 + h^2, \qquad
u^2 = (r + r')^2 + h^2, \qquad
v^2 = r^2 + r'^2 + h^2,
$$

$$
m = k^2 = \frac{4\, r\, r'}{u^2}, \qquad
\lambda = \frac{\Delta r}{\Delta z},
$$

$$
F(\lambda) = 1 - \frac{\pi}{6\lambda}
+ \frac{\pi}{\lambda}\sum_{n=1}^{\infty}\operatorname{csch}^2\!\left(\frac{\pi n}{\lambda}\right),
\qquad
F(\lambda) = 1 - F(1/\lambda)\;\;\text{for }\lambda > 1 .
$$

Notes:

* $K(m)$ and $E(m)$ are the complete elliptic integrals of the first and second
  kind **with parameter $m = k^2$** (the SciPy convention,
  `scipy.special.ellipk(m)` / `ellipe(m)`; in the Rust code
  `ellpk(1 - k_sq)` / `ellpe(k_sq)`).
* The off-source branch follows from differentiating the loop flux function
  twice; it is equivalent to Garrett's elliptic-integral field-gradient
  formulas via $\partial^2\psi/\partial z^2 = -2\pi r\,\partial B_r/\partial z$.
* The coincident branch is the delta-function source of the Grad–Shafranov
  operator, $\Delta^*\psi = -2\pi\mu_0 r J_\phi$, shared between
  $\psi_{zz}$ and $\psi_{rr}$ by the cell aspect ratio. $F(\lambda)$ contains
  the isolated-rectangle splitting $f_z = \tfrac{2}{\pi}\arctan\lambda$ **plus**
  the lattice (midpoint-quadrature) correction appropriate for a source cell
  embedded in a regular filament grid; the series converges in 3–4 terms.
  The companion self-term of $\psi_{rr}$ uses $1 - F(\lambda)$, so the pair
  integrates the delta function exactly.
* $F(1) = 1/2$ exactly. $F$ can lie outside $[0, 1]$ for elongated cells
  (e.g. $F(2) = 1.04711$) — this is correct, see
  [`jump_condition_dbr_dz.md`](jump_condition_dbr_dz.md) §6.
* Validation: summed over a discretised conductor, this expression matches a
  Richardson-extrapolated second difference of $\psi$ to $\sim 3\times10^{-6}$
  relative (see `examples/d2_psi_d_z2_investigation.ipynb`).

## References

1. M. W. Garrett, *Calculation of Fields, Forces, and Mutual Inductances of
   Current Systems by Elliptic Integrals*, J. Appl. Phys. **34**, 2567 (1963) —
   elliptic-integral expressions for axisymmetric loop fields and their
   gradients, including $\partial B_\rho/\partial z$ (the off-source branch).
2. J. D. Jackson, *Classical Electrodynamics* (3rd ed.), §5.8 — the tangential
   jump condition $[\mathbf{B}_\parallel] = \mu_0 \mathbf{K}$ across a current
   sheet (the $\lambda \to \infty$ limit of the coincident branch).
3. L. Hurwitz and M. Landreman, *Efficient calculation of self magnetic field,
   self-force, and self-inductance for electromagnetic coils with rectangular
   cross-section*, arXiv:2310.12087 / Nucl. Fusion (2025) — self-field of an
   isolated rectangular-cross-section circular conductor (the
   $\tfrac{2}{\pi}\arctan\lambda$ part of $F$).
4. J. M. Borwein, M. L. Glasser, R. C. McPhedran, J. G. Wan, I. J. Zucker,
   *Lattice Sums Then and Now*, Cambridge University Press (2013) — the
   Eisenstein-series / $\operatorname{csch}^2$ row-summation identities used to
   evaluate the dipole lattice sum in closed form (e.g.
   $\sum_{n\ge1}\operatorname{csch}^2(\pi n) = \tfrac16 - \tfrac{1}{2\pi}$,
   which gives $F(1) = \tfrac12$).

The combined formula $F(\lambda)$ — the lattice-corrected self-term for the
*diagonal of a discretised Green's table* — was derived for GSFit in
[`jump_condition_dbr_dz.md`](jump_condition_dbr_dz.md) §5–6 from these
classical ingredients; we are not aware of it appearing in published form, and
it is validated numerically in `examples/d2_psi_d_z2_investigation.ipynb` and
in the Rust tests (`test_d2_psi_d_z2_self_fraction`,
`test_d2_psi_d_z2_self_term`).

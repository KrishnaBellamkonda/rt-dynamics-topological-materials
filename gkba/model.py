"""
Two band Chern insulator model, non interacting pieces.
Interactions live in interaction.py and gkba.py.

Reference
M. Schueler, J. C. Budich and P. Werner, Phys. Rev. B 97, 245129 (2018).

h0(k) = dx sigma_x + dy sigma_y + dz sigma_z
  dx = lam sin(kx)
  dy = lam sin(ky)
  dz = M + cos(kx) + cos(ky)
Two orbitals E and H (index 0 and 1). M tunes the topology, lam mixes the bands.
"""
import os
import numpy as _np

# numpy on CPU, cupy on GPU. every array goes through xp
_BACKEND = os.environ.get("GKBA_BACKEND", "numpy").lower()
if _BACKEND == "cupy":
    import cupy as xp
else:
    import numpy as xp


def _asnumpy(a):
    """bring a device array back to host numpy"""
    return xp.asnumpy(a) if _BACKEND == "cupy" else _np.asarray(a)


# BZ grid
def kgrid(Nk):
    """Nk points per direction on [0, 2 pi)"""
    return xp.linspace(0.0, 2.0 * _np.pi, Nk, endpoint=False)


def _kflat(Nk):
    """flat (kx, ky) for an Nk by Nk grid, index ik = a Nk + b"""
    ks = kgrid(Nk)
    KX, KY = xp.meshgrid(ks, ks, indexing="ij")
    return KX.ravel(), KY.ravel()


# Bloch Hamiltonian, one 2x2 per k
def h0_grid(kx, ky, M, lam):
    """stack of 2x2 Bloch Hamiltonians h0(k) = d(k) . sigma"""
    dx = lam * xp.sin(kx)
    dy = lam * xp.sin(ky)
    dz = M + xp.cos(kx) + xp.cos(ky)          # +cos+cos puts the inversion at Gamma
    H = xp.zeros((kx.size, 2, 2), dtype=xp.complex128)
    H[:, 0, 0] = dz
    H[:, 1, 1] = -dz
    H[:, 0, 1] = dx - 1j * dy
    H[:, 1, 0] = dx + 1j * dy
    return H


def _dagger(A):
    """hermitian conjugate of a stack of matrices"""
    return xp.conj(xp.swapaxes(A, -1, -2))


# quenched initial state
def init_rho(kx, ky, M_pre, lam):
    """rho(k) for the filled lower band of h0(M_pre), a projector |psi><psi|"""
    H = h0_grid(kx, ky, M_pre, lam)
    w, v = xp.linalg.eigh(H)                   # ascending, column 0 is the lower band
    lo = v[:, :, 0]
    return xp.einsum("ka,kb->kab", lo, lo.conj())


# propagator
I2 = xp.eye(2, dtype=xp.complex128)


def _expm_batch(H, dt):
    """exp(-i H dt) for a batch of 2x2 H, closed form via H = a I + d . sigma"""
    a = 0.5 * (H[:, 0, 0] + H[:, 1, 1]).real   # identity part
    dz = 0.5 * (H[:, 0, 0] - H[:, 1, 1]).real
    dx = H[:, 1, 0].real
    dy = H[:, 1, 0].imag
    D = xp.sqrt(dx * dx + dy * dy + dz * dz)
    D = xp.where(D < 1e-15, 1e-15, D)           # guard the degenerate D near zero case
    c = xp.cos(D * dt)
    s = xp.sin(D * dt)
    ph = xp.exp(-1j * a * dt)                    # overall phase from the identity part
    U = xp.empty_like(H)
    U[:, 0, 0] = c - 1j * s * dz / D
    U[:, 1, 1] = c + 1j * s * dz / D
    U[:, 0, 1] = -1j * s * (dx - 1j * dy) / D
    U[:, 1, 0] = -1j * s * (dx + 1j * dy) / D
    return ph[:, None, None] * U


# observables

# path for the occupation figure in the paper
def _highsym_path(Nk):
    """flat k indices tracing M to Gamma to X to M"""
    h = Nk // 2

    def ik(a, b):
        return a * Nk + b

    seg = [ik(i, i) for i in range(h, -1, -1)]   # M to Gamma
    seg += [ik(a, 0) for a in range(1, h + 1)]   # Gamma to X
    seg += [ik(h, b) for b in range(1, h + 1)]   # X to M
    return xp.asarray(_np.array(seg))


def _chern_fh(U):
    """Fukui Hatsugai lattice Chern number of an eigenvector field U (Nk, Nk, 2)"""
    # link variables along x and y, put on the unit circle
    Ux = _np.sum(_np.conj(U) * _np.roll(U, -1, axis=0), axis=2)
    Uy = _np.sum(_np.conj(U) * _np.roll(U, -1, axis=1), axis=2)
    Ux /= _np.abs(Ux)
    Uy /= _np.abs(Uy)
    # plaquette flux summed over the BZ, divided by 2 pi
    F = _np.angle(Ux * _np.roll(Uy, -1, axis=0)
                  * _np.conj(_np.roll(Ux, -1, axis=1)) * _np.conj(Uy))
    return float(_np.sum(F) / (2.0 * _np.pi))

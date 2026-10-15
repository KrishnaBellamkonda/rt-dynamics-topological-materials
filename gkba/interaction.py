"""
k resolved second born self energy for the interacting Chern insulator.

Interaction in momentum space, q is the momentum transfer
  W_intra(q) = 2 V1 (cos qx + cos qy)        same orbital
  W_inter(q) = V0 + 2 V1 (cos qx + cos qy)    different orbital
V0 is the on site repulsion, V1 the nearest neighbour one. V1 = 0 is the local
model (W flat in q), V1 > 0 is the nonlocal model.

Second born self energy, diagonal in orbital, built from the diagonal G
  Pi_a(q)   = (1/Nk^2) sum_k'  G_nm_aa(k') G_mn_aa(k' - q)
  Sig_EE(k) = (1/Nk^2) sum_q [ W_intra(q)^2 Pi_E(q) + W_inter(q)^2 Pi_H(q) ] G_nm_EE(k - q)
  Sig_HH(k) = (1/Nk^2) sum_q [ W_intra(q)^2 Pi_H(q) + W_inter(q)^2 Pi_E(q) ] G_nm_HH(k - q)
Pi is a correlation over the BZ, Sig a convolution. Both are periodic so a 2D
FFT does them in O(Nk^2 log Nk). sigma_fft is production, sigma_bruteforce is
the plain O(Nk^4) definition kept to check it.
"""
import os
import numpy as _np

# numpy on CPU, cupy on GPU
_BACKEND = os.environ.get("GKBA_BACKEND", "numpy").lower()
if _BACKEND == "cupy":
    import cupy as xp
else:
    import numpy as xp


# local and nonlocal interactions. q is the momentum transfer, every transfer is
# allowed and the simulation sums the contributions from all of them
def Wq(Nk, V0, V1):
    """interaction in momentum space, returns (W_intra, W_inter), each (Nk, Nk)"""
    ks = xp.linspace(0.0, 2.0 * _np.pi, Nk, endpoint=False)
    QX, QY = xp.meshgrid(ks, ks, indexing="ij")
    nn = 2.0 * V1 * (xp.cos(QX) + xp.cos(QY))   # nearest neighbour part
    return nn, V0 + nn                            # intra, inter


# plain definition, used to check sigma_fft on a small grid
def sigma_bruteforce(Gnm, Gmn, Wintra, Winter):
    """explicit O(Nk^4) momentum sums, the literal definition"""
    Nk = Gnm.shape[0]
    N2 = Nk * Nk
    gE_nm = Gnm[:, :, 0, 0].copy()
    gH_nm = Gnm[:, :, 1, 1].copy()
    gE_mn = Gmn[:, :, 0, 0].copy()
    gH_mn = Gmn[:, :, 1, 1].copy()

    # bubbles Pi_a(q) by shift and sum
    PiE = xp.zeros((Nk, Nk), dtype=xp.complex128)
    PiH = xp.zeros((Nk, Nk), dtype=xp.complex128)
    for qx in range(Nk):
        for qy in range(Nk):
            PiE[qx, qy] = xp.sum(gE_nm * xp.roll(xp.roll(gE_mn, qx, 0), qy, 1)) / N2
            PiH[qx, qy] = xp.sum(gH_nm * xp.roll(xp.roll(gH_mn, qx, 0), qy, 1)) / N2

    # weight the bubbles by the interaction
    FE = Wintra**2 * PiE + Winter**2 * PiH
    FH = Wintra**2 * PiH + Winter**2 * PiE

    # self energy convolution Sig_a(k) = sum_q F_a(q) g_nm(k - q)
    SigE = xp.zeros((Nk, Nk), dtype=xp.complex128)
    SigH = xp.zeros((Nk, Nk), dtype=xp.complex128)
    for kx in range(Nk):
        for ky in range(Nk):
            gE_shift = xp.roll(xp.roll(gE_nm[::-1, ::-1], kx + 1, 0), ky + 1, 1)
            gH_shift = xp.roll(xp.roll(gH_nm[::-1, ::-1], kx + 1, 0), ky + 1, 1)
            SigE[kx, ky] = xp.sum(FE * gE_shift) / N2
            SigH[kx, ky] = xp.sum(FH * gH_shift) / N2

    Sig = xp.zeros((Nk, Nk, 2, 2), dtype=xp.complex128)
    Sig[:, :, 0, 0] = SigE
    Sig[:, :, 1, 1] = SigH
    return Sig


# FFT helpers on the periodic grid
def _conv(f, g):
    """periodic convolution C(k) = sum_q f(q) g(k - q) by 2D FFT"""
    return xp.fft.ifft2(xp.fft.fft2(f) * xp.fft.fft2(g))


def _reverse(b):
    """periodic index reversal b_rev[i] = b[(-i) mod N] on each axis"""
    return xp.roll(xp.flip(b), shift=(1, 1), axis=(0, 1))


def _corr(a, b):
    """periodic correlation C(q) = sum_k' a(k') b(k' - q) = conv(a, reverse(b))"""
    return _conv(a, _reverse(b))


def sigma_fft(Gnm, Gmn, Wintra, Winter):
    """production self energy, same result as sigma_bruteforce but O(Nk^2 log Nk)"""
    Nk = Gnm.shape[0]
    gE_nm = Gnm[:, :, 0, 0]
    gH_nm = Gnm[:, :, 1, 1]
    gE_mn = Gmn[:, :, 0, 0]
    gH_mn = Gmn[:, :, 1, 1]

    PiE = _corr(gE_nm, gE_mn) / (Nk * Nk)     # orbital bubbles
    PiH = _corr(gH_nm, gH_mn) / (Nk * Nk)

    FE = Wintra**2 * PiE + Winter**2 * PiH
    FH = Wintra**2 * PiH + Winter**2 * PiE

    SigE = _conv(FE, gE_nm) / (Nk * Nk)       # self energy
    SigH = _conv(FH, gH_nm) / (Nk * Nk)

    Sig = xp.zeros((Nk, Nk, 2, 2), dtype=xp.complex128)
    Sig[:, :, 0, 0] = SigE
    Sig[:, :, 1, 1] = SigH
    return Sig

"""
GKBA second born time propagation of the interacting Chern insulator quench.

Reference
M. Schueler, J. C. Budich and P. Werner, Phys. Rev. B 97, 245129 (2018).

We propagate the single particle density matrix rho(k, t), a 2x2 orbital matrix
at every k, under the equation of motion
    d rho(k,t)/dt = -i [ h_MF(k,t), rho(k,t) ] - ( I(k,t) + I(k,t)^dagger )
h_MF = h0 + Sigma_HF is the mean field part (built in _hf_nonlocal) and I is the
collision integral that carries the second born scattering
    I(k,t_n) = dt sum_m w_m [ Sigma_>(k;n,m) G_<(k;m,n) - Sigma_<(k;n,m) G_>(k;m,n) ].
The GKBA closes this by rebuilding the two time Green functions from rho and the
mean field propagator U(k;n,m) = P_n P_m^dagger
    G_<(k;n,m) =  i U rho(k,m)
    G_>(k;n,m) = -i U ( 1 - rho(k,m) )
    G(k;m,n)   = - [ G(k;n,m) ]^dagger.
Quench. rho starts as the filled lower band of h0(M_pre). At t = 0 the mass is
switched to M_post and the interaction is turned on. The same routine runs the
local model (V1 = 0) and the nonlocal model (V1 > 0).
"""
import os
import numpy as _np

# numpy on CPU, cupy on GPU
_BACKEND = os.environ.get("GKBA_BACKEND", "numpy").lower()
if _BACKEND == "cupy":
    import cupy as xp
else:
    import numpy as xp

import model
import interaction

I2 = xp.eye(2, dtype=xp.complex128)


# probe free instantaneous Hall conductance from one density matrix
def _sigma_xy_step(rho_n, Nk):
    """sigma_xy = (1/2pi) integral |r| Omega dk, Omega the Berry curvature of r_hat"""
    rx = xp.real(rho_n[:, 0, 1] + rho_n[:, 1, 0])
    ry = xp.real(1j * (rho_n[:, 0, 1] - rho_n[:, 1, 0]))
    rz = xp.real(rho_n[:, 0, 0] - rho_n[:, 1, 1])
    mag = xp.sqrt(rx * rx + ry * ry + rz * rz)
    r = xp.stack([rx, ry, rz], axis=-1).reshape(Nk, Nk, 3)
    m = mag.reshape(Nk, Nk)
    rhat = r / xp.clip(m[..., None], 1e-12, None)
    dk = 2.0 * _np.pi / Nk
    # central differences of r_hat on the periodic grid
    dX = (xp.roll(rhat, -1, 0) - xp.roll(rhat, 1, 0)) / (2 * dk)
    dY = (xp.roll(rhat, -1, 1) - xp.roll(rhat, 1, 1)) / (2 * dk)
    S = xp.sum(rhat * xp.cross(dX, dY), axis=-1)       # skyrmion density
    Omega = -0.5 * S
    return float((1.0 / (2 * _np.pi)) * xp.sum(m * Omega) * dk * dk)


# mean field (Hartree Fock) self energy to add to h0
def _hf_nonlocal(rho_grid, V0, V1, Wi, We, Nk):
    """Hartree part is the q = 0 interaction times the local density. Fock part is
    the exchange convolution -conv(W, rho)/Nk^2. V1 = 0 gives the on site case."""
    N2 = Nk * Nk
    rl = rho_grid.reshape(N2, 2, 2).mean(axis=0)       # BZ averaged rho
    S = xp.zeros((Nk, Nk, 2, 2), dtype=xp.complex128)

    # Hartree, diagonal, q = 0 component of the interaction
    w0_intra = 4.0 * V1
    w0_inter = V0 + 4.0 * V1
    S[:, :, 0, 0] += w0_intra * rl[0, 0] + w0_inter * rl[1, 1]
    S[:, :, 1, 1] += w0_inter * rl[0, 0] + w0_intra * rl[1, 1]

    # Fock exchange, k dependent, by FFT
    S[:, :, 0, 0] += -interaction._conv(Wi, rho_grid[:, :, 0, 0]) / N2
    S[:, :, 1, 1] += -interaction._conv(Wi, rho_grid[:, :, 1, 1]) / N2
    S[:, :, 0, 1] += -interaction._conv(We, rho_grid[:, :, 0, 1]) / N2
    S[:, :, 1, 0] += -interaction._conv(We, rho_grid[:, :, 1, 0]) / N2
    return S


# master routine that runs the whole quench and returns the observables
def gkba_run_kresolved(M_pre, M_post, lam, V0, V1, Nk, Nt, dt, n_mem=400,
                       save_every=10, verbose=True):
    """run the GKBA quench. V1 = 0 is local, V1 > 0 is nonlocal"""
    kx, ky = model._kflat(Nk)
    Nk2 = kx.size
    Wi, We = interaction.Wq(Nk, V0, V1)                # intra and inter interaction

    # post quench Hamiltonian and its upper band, used for f_plus
    Hp = model.h0_grid(kx, ky, M_post, lam)            # one 2x2 per k
    wv, vv = xp.linalg.eigh(Hp)                         # two bands per k
    up = vv[:, :, 1]
    H0 = Hp

    rho = model.init_rho(kx, ky, M_pre, lam)           # quenched initial state

    # ring buffers holding the last n_mem+2 times of rho, the propagator P and
    # P^dagger. indexing is modulo cap
    cap = n_mem + 2
    rho_h = xp.empty((cap, Nk2, 2, 2), dtype=xp.complex128)
    P_h = xp.empty((cap, Nk2, 2, 2), dtype=xp.complex128)
    Pd_h = xp.empty((cap, Nk2, 2, 2), dtype=xp.complex128)

    def put(b, i, v):
        b[i % cap] = v

    def get(b, i):
        return b[i % cap]

    put(rho_h, 0, rho)
    P0 = xp.broadcast_to(I2, (Nk2, 2, 2)).copy()       # P(0) = identity
    put(P_h, 0, P0)
    put(Pd_h, 0, model._dagger(P0))

    # mean field Hamiltonian h0 + Sigma_HF at a given rho
    def hmf_of(rho_n):
        add = _hf_nonlocal(rho_n.reshape(Nk, Nk, 2, 2), V0, V1, Wi, We, Nk).reshape(Nk2, 2, 2)
        return H0 + add

    # second born self energy Sigma(k;n,m) by FFT
    def sig_pair(Gnm, Gmn_partner, les):
        g1 = Gnm.reshape(Nk, Nk, 2, 2)
        g2 = Gmn_partner.reshape(Nk, Nk, 2, 2)
        return interaction.sigma_fft(g1, g2, Wi, We).reshape(Nk2, 2, 2)

    # Collision integral, second born scattering. Method references
    #   Lipavsky, Spicka and Velicky, Phys. Rev. B 34, 6933 (1986), the GKBA
    #   Stefanucci and van Leeuwen, Nonequilibrium Many Body Theory of Quantum
    #     Systems, Cambridge Univ. Press (2013), the collision integral
    #   Schueler, Budich and Werner, Phys. Rev. B 97, 245129 (2018), the quench
    # Independent open source codes for the same scheme
    #   NESSi (libcntr)  https://github.com/nessi-cntr/nessi
    #   G12KBA           https://github.com/VlcekGroup/G12KBA
    def collision(n, Pn, rho_n):
        """sum over the memory window. rebuild G(n,m) by GKBA, build Sigma(n,m),
        accumulate the integrand with trapezoidal weights"""
        if V0 == 0.0 and V1 == 0.0:
            return xp.zeros((Nk2, 2, 2), dtype=xp.complex128)
        m0 = max(0, n - n_mem)
        hist = list(range(m0, n))
        Mtot = len(hist) + 1
        wts = xp.ones(Mtot)
        wts[0] = 0.5
        wts[-1] = 0.5                                   # trapezoidal rule
        acc = xp.zeros((Nk2, 2, 2), dtype=xp.complex128)
        Pnd = model._dagger(Pn)
        for p in range(Mtot):
            if p < len(hist):
                m = hist[p]
                Pm_d = get(Pd_h, m)
                rm = get(rho_h, m)
            else:                                        # the current time n
                Pm_d = Pnd
                rm = rho_n
            U_nm = Pn @ Pm_d                             # U(k;n,m) = P_n P_m^dagger
            Gl_nm = 1j * (U_nm @ rm)                     # G_<(n,m)
            Gg_nm = -1j * (U_nm @ (I2 - rm))             # G_>(n,m)
            Gl_mn = -model._dagger(Gl_nm)                # G_<(m,n)
            Gg_mn = -model._dagger(Gg_nm)                # G_>(m,n)
            Sig_g = sig_pair(Gg_nm, Gl_mn, False)        # Sigma_>(n,m)
            Sig_l = sig_pair(Gl_nm, Gg_mn, True)         # Sigma_<(n,m)
            term = (Sig_g @ Gl_mn) - (Sig_l @ Gg_mn)
            acc = acc + float(wts[p]) * term
        return dt * acc

    # observable buffers
    from_path = model._highsym_path(Nk)
    saved_t, E, Fp, Pur, Cin, Fp_path, Ecorr, Etot, Sxy = [], [], [], [], [], [], [], [], []

    def record(n, rho_n, I_coll=None):
        t = n * dt
        # energy, one body plus mean field plus optional correlation
        e1 = xp.real(xp.einsum("kab,kba->k", H0, rho_n)).mean()
        add = hmf_of(rho_n) - H0
        ehf = 0.5 * xp.real(xp.einsum("kab,kba->k", add, rho_n)).mean()
        emf = float(e1 + ehf)
        E.append(emf)
        ec = 0.0 if I_coll is None else float(0.5 * xp.real(xp.einsum("kaa->k", I_coll)).mean())
        Ecorr.append(ec)
        Etot.append(emf + ec)
        # upper band occupation, BZ mean and along the path
        ov = xp.real(xp.einsum("ka,kab,kb->k", up.conj(), rho_n, up))
        Fp.append(float(ov.mean()))
        Fp_path.append(model._asnumpy(ov[from_path]))
        # pseudospin purity, its minimum over k
        rx = xp.real(rho_n[:, 0, 1] + rho_n[:, 1, 0])
        ry = xp.real(1j * (rho_n[:, 0, 1] - rho_n[:, 1, 0]))
        rz = xp.real(rho_n[:, 0, 0] - rho_n[:, 1, 1])
        Pur.append(float(xp.sqrt(rx * rx + ry * ry + rz * rz).min()))
        # instantaneous Chern number of the occupied natural orbital
        w2, v2 = xp.linalg.eigh(rho_n)
        occ = model._asnumpy(v2[:, :, 1]).reshape(Nk, Nk, 2)
        Cin.append(model._chern_fh(occ))
        Sxy.append(_sigma_xy_step(rho_n, Nk))
        saved_t.append(t)

    record(0, rho, None)

    # time loop, midpoint mean field step plus predictor corrector collision
    for n in range(Nt):
        # unitary mean field substep, exponential midpoint rule
        hmf_n = hmf_of(rho)
        Uh = model._expm_batch(hmf_n, 0.5 * dt)
        rho_mid = Uh @ rho @ model._dagger(Uh)
        Umid = model._expm_batch(hmf_of(rho_mid), dt)
        rho_unit = Umid @ rho @ model._dagger(Umid)

        # advance the propagator, P_{n+1} = U_mid P_n
        Pn = get(P_h, n)
        Pnp1 = Umid @ Pn

        # predictor, collision at the current step
        I_n = collision(n, Pn, rho)
        coll_n = I_n + model._dagger(I_n)
        rho_P = rho_unit - dt * coll_n

        I_last = None
        if V0 != 0.0 or V1 != 0.0:
            # corrector, collision at the predicted next step, then average
            put(P_h, n + 1, Pnp1)
            put(Pd_h, n + 1, model._dagger(Pnp1))
            put(rho_h, n + 1, rho_P)
            I_np1 = collision(n + 1, Pnp1, rho_P)
            rho_new = rho_unit - 0.5 * dt * (coll_n + I_np1 + model._dagger(I_np1))
            I_last = I_np1
        else:
            rho_new = rho_unit                            # non interacting, unitary only

        rho = 0.5 * (rho_new + model._dagger(rho_new))    # keep rho hermitian
        put(rho_h, n + 1, rho)
        put(P_h, n + 1, Pnp1)
        put(Pd_h, n + 1, model._dagger(Pnp1))

        if (n + 1) % save_every == 0 or n + 1 == Nt:
            record(n + 1, rho, I_last)
        if verbose and (n + 1) % max(1, Nt // 10) == 0:
            print(f"  step {n+1}/{Nt} t={(n+1)*dt:.1f} E={E[-1]:.6f} "
                  f"f+={Fp[-1]:.4f} pur={Pur[-1]:.3f}", flush=True)

    return dict(times=_np.array(saved_t), energy=_np.array(E), e_corr=_np.array(Ecorr),
                e_total=_np.array(Etot), f_plus=_np.array(Fp),
                purity_min=_np.array(Pur), C_inst=_np.array(Cin),
                sigma_xy_inst=_np.array(Sxy),
                f_plus_path=_np.array(Fp_path), path_idx=model._asnumpy(from_path),
                V0=V0, V1=V1, M_pre=M_pre, M_post=M_post, lam=lam, Nk=Nk, dt=dt, n_mem=n_mem)

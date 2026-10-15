"""
Read the NESSi KBE driver output and build the observables.

The driver writes, per saved step, the 2x2 orbital density matrix rho(k) for
every k as 8 doubles (Re and Im of rho00, rho01, rho10, rho11), with k index
ik = a Nk + b and kx = 2 pi a / Nk, ky = 2 pi b / Nk. All observables are
computed here. The Hamiltonian convention matches the driver (M - cos - cos).
"""
import numpy as np


def kgrid(Nk):
    """Nk points per direction on [0, 2 pi)"""
    return np.linspace(0.0, 2.0 * np.pi, Nk, endpoint=False)


def chern_from_states(U):
    """Fukui Hatsugai lattice Chern number of an eigenvector field U (Nk, Nk, 2)"""
    Nk = U.shape[0]

    def link(a, b):
        ov = np.vdot(a, b)
        return ov / np.abs(ov)

    C = 0.0
    for i in range(Nk):
        ip = (i + 1) % Nk
        for j in range(Nk):
            jp = (j + 1) % Nk
            U1 = link(U[i, j], U[ip, j])
            U2 = link(U[ip, j], U[ip, jp])
            U3 = link(U[ip, jp], U[i, jp])
            U4 = link(U[i, jp], U[i, j])
            C += np.angle(U1 * U2 * U3 * U4)
    return C / (2.0 * np.pi)


def load_meta(prefix):
    """read prefix_meta.txt into a dict"""
    meta = {}
    with open(prefix + "_meta.txt") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                try:
                    meta[p[0]] = float(p[1]) if "." in p[1] or "e" in p[1].lower() else int(p[1])
                except ValueError:
                    pass
    return meta


def load_run(prefix):
    """return times, rho (nsaved, Nk2, 2, 2) complex, and the meta dict"""
    meta = load_meta(prefix)
    Nk, Nt, se, dt = meta["Nk"], meta["Nt"], meta["SaveEvery"], meta["dt"]
    Nk2 = Nk * Nk
    raw = np.fromfile(prefix + "_rho.bin", dtype=np.float64)
    nsaved = raw.size // (Nk2 * 8)
    raw = raw[: nsaved * Nk2 * 8].reshape(nsaved, Nk2, 4, 2)
    rho = raw[..., 0] + 1j * raw[..., 1]
    rho = rho.reshape(nsaved, Nk2, 2, 2)
    saved_steps = [s for s in range(0, Nt + 1) if s % se == 0 or s == Nt]
    times = np.array(saved_steps[:nsaved], dtype=float) * dt
    return times, rho, meta


def _kflat(Nk):
    ks = kgrid(Nk)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    return KX.ravel(), KY.ravel()


def _h0(kx, ky, M, lam):
    dx = lam * np.sin(kx)
    dy = lam * np.sin(ky)
    dz = M - np.cos(kx) - np.cos(ky)
    H = np.empty((kx.size, 2, 2), dtype=complex)
    H[:, 0, 0] = dz
    H[:, 1, 1] = -dz
    H[:, 0, 1] = dx - 1j * dy
    H[:, 1, 0] = dx + 1j * dy
    return H


def f_plus(rho, Nk, Mpost, lam):
    """upper band occupation f_plus(k,t) with respect to post quench h0, (nsaved, Nk2)"""
    kx, ky = _kflat(Nk)
    H = _h0(kx, ky, Mpost, lam)
    w, v = np.linalg.eigh(H)
    up = v[:, :, 1]
    return np.real(np.einsum("ka,tkab,kb->tk", up.conj(), rho, up))


def pseudospin(rho):
    rx = np.real(rho[..., 0, 1] + rho[..., 1, 0])
    ry = np.real(1j * (rho[..., 0, 1] - rho[..., 1, 0]))
    rz = np.real(rho[..., 0, 0] - rho[..., 1, 1])
    return rx, ry, rz


def purity_min(rho):
    """min over k of |r(k)| per time"""
    rx, ry, rz = pseudospin(rho)
    return np.sqrt(rx * rx + ry * ry + rz * rz).min(axis=1)


def c_inst(rho, Nk):
    """instantaneous Chern number of the occupied natural orbital, per time"""
    out = np.empty(rho.shape[0])
    for t in range(rho.shape[0]):
        w, v = np.linalg.eigh(rho[t])
        occ = v[:, :, 1].reshape(Nk, Nk, 2)
        out[t] = chern_from_states(occ)
    return out


def sigma_xy_inst(rho, Nk):
    """instantaneous Hall conductance and Chern from the pseudospin texture"""
    rx = np.real(rho[..., 0, 1] + rho[..., 1, 0])
    ry = np.real(1j * (rho[..., 0, 1] - rho[..., 1, 0]))
    rz = np.real(rho[..., 0, 0] - rho[..., 1, 1])
    rmag = np.sqrt(rx * rx + ry * ry + rz * rz)
    dk = 2.0 * np.pi / Nk
    sxy, cin = [], []
    for it in range(rho.shape[0]):
        r = np.stack([rx[it], ry[it], rz[it]], axis=-1).reshape(Nk, Nk, 3)
        mag = rmag[it].reshape(Nk, Nk)
        rhat = r / np.clip(mag[..., None], 1e-12, None)
        dX = (np.roll(rhat, -1, 0) - np.roll(rhat, 1, 0)) / (2 * dk)
        dY = (np.roll(rhat, -1, 1) - np.roll(rhat, 1, 1)) / (2 * dk)
        S = np.sum(rhat * np.cross(dX, dY), axis=-1)
        Omega = -0.5 * S
        cin.append((1.0 / (2 * np.pi)) * np.sum(Omega) * dk * dk)
        sxy.append((1.0 / (2 * np.pi)) * np.sum(mag * Omega) * dk * dk)
    return np.array(sxy), np.array(cin)


def high_symmetry_path(Nk):
    """flat k indices tracing M to Gamma to X to M, plus tick positions"""
    h = Nk // 2

    def ik(a, b):
        return a * Nk + b

    seg = []
    for i in range(h, -1, -1):
        seg.append(ik(i, i))
    for a in range(1, h + 1):
        seg.append(ik(a, 0))
    for b in range(1, h + 1):
        seg.append(ik(h, b))
    ticks = {"M": 0, "G": h, "X": 2 * h, "M2": 3 * h}
    return np.array(seg), ticks

"""Checks for Lemma FK-log (strategy extraction with an O(log T) constant) and the clock-factorization bound.

Rewound referee cell on the legal clock space (Alice clock t_A, Bob clock t_B in {0..T}):
    H' = L_grid (x) I + |00><00| (x) (I - P) + |TT><TT| (x) Q.
Claims checked:
  (1) G(T) := L^+_{cc} at a corner c of the (T+1)x(T+1) grid grows like (2/pi) ln T (exact eigen-sum formula).
  (2) For the ground state AND for random low-energy states Psi of energy lambda:
        L_ext(Psi) <= C_G * N_tau * lambda,   C_G = 2 (3 + 4 sqrt(G))^2 + 1/2,
      where L_ext is the loss of the extracted strategy (fresh questions, clocks and old registers ignored).
  (3) Clock factorization: || Psi' - |unif> (x) u ||^2 <= N_tau * lambda / 4 with u = sqrt(N_tau) * mean_tau psi_tau.
  (4) An adversarial low-energy family (state morphs from a bad strategy far from the corner to the optimal one
      at the corner) to see how close the O(log T) constant is to necessary for arbitrary low-energy states.
Single-threaded numpy/scipy, small dimensions.
"""
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

rng = np.random.default_rng(20261029)
OUT = os.path.dirname(os.path.abspath(__file__))


def corner_green(T):
    """L^+_{cc} for the corner of the (T+1)x(T+1) grid (free/Neumann path Laplacians), exact eigen-sum."""
    n = T + 1
    a = np.arange(n)
    mu = 2.0 - 2.0 * np.cos(np.pi * a / n)
    phi0sq = np.where(a == 0, 1.0 / n, (2.0 / n) * np.cos(np.pi * a / (2 * n)) ** 2)
    MU = mu[:, None] + mu[None, :]
    W = phi0sq[:, None] * phi0sq[None, :]
    MU[0, 0] = 1.0
    W[0, 0] = 0.0
    return float((W / MU).sum())


def path_lap(n):
    main = np.full(n, 2.0)
    main[0] = main[-1] = 1.0
    off = -np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr")


def grid_lap(T):
    n = T + 1
    Lp = path_lap(n)
    I = sp.identity(n, format="csr")
    return (sp.kron(Lp, I) + sp.kron(I, Lp)).tocsr()


def random_instance(dreg, dsys, rankQ):
    reg = rng.normal(size=dreg)
    reg /= np.linalg.norm(reg)
    P = np.kron(np.outer(reg, reg), np.eye(dsys))
    d = dreg * dsys
    A = rng.normal(size=(d, rankQ))
    Qb, _ = np.linalg.qr(A)
    Q = Qb @ Qb.T
    V = np.kron(reg.reshape(dreg, 1), np.eye(dsys))
    Msys = V.T @ Q @ V
    return P, Q, Msys, dreg, dsys


def ext_loss(psi, P, Q, Msys, dreg, dsys):
    """Loss of the extracted strategy: sum_tau [<P psi|Q|P psi> + honest loss of (I-P) psi on the system]."""
    d = dreg * dsys
    IminusP = np.eye(d) - P
    L = 0.0
    for t in range(psi.shape[0]):
        a = P @ psi[t]
        b = IminusP @ psi[t]
        L += float(a @ Q @ a)
        B = b.reshape(dreg, dsys)
        L += float(np.trace(Msys @ (B.T @ B)))
    return L


def energy(psi, H):
    v = psi.reshape(-1)
    return float(v @ (H @ v))


def build_H(T, P, Q):
    n = T + 1
    Nt = n * n
    d = P.shape[0]
    Lg = grid_lap(T)
    E00 = sp.csr_matrix(([1.0], ([0], [0])), shape=(Nt, Nt))
    ETT = sp.csr_matrix(([1.0], ([Nt - 1], [Nt - 1])), shape=(Nt, Nt))
    Id = sp.identity(d, format="csr")
    H = sp.kron(Lg, Id) + sp.kron(E00, sp.csr_matrix(np.eye(d) - P)) + sp.kron(ETT, sp.csr_matrix(Q))
    return H.tocsr(), Nt, d


def main():
    rec = {"green": [], "instances": [], "adversarial": []}
    # (1) corner Green's function
    for T in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        G = corner_green(T)
        rec["green"].append(dict(T=T, G=G, G_over_lnT=(G / math.log(T)) if T > 1 else None,
                                 two_over_pi=2 / math.pi))
        print(f"G({T}) = {G:.5f}", f"G/lnT = {G / math.log(T):.4f}" if T > 1 else "", flush=True)
    # (2),(3) instances
    allok = True
    for inst in range(8):
        dreg, dsys, rankQ = 4, 4, int(rng.integers(2, 10))
        P, Q, Msys, dreg, dsys = random_instance(dreg, dsys, rankQ)
        eps_opt = float(np.linalg.eigvalsh(Msys)[0])
        for T in [2, 4, 8, 16, 24]:
            H, Nt, d = build_H(T, P, Q)
            k = 6
            w, v = spla.eigsh(H, k=k, which="SA", tol=1e-12)
            order = np.argsort(w)
            w, v = w[order], v[:, order]
            G = corner_green(T)
            CG = 2 * (3 + 4 * math.sqrt(G)) ** 2 + 0.5
            states = [("ground", v[:, 0])]
            for r in range(3):
                c = rng.normal(size=k) * np.exp(-np.arange(k))
                c /= np.linalg.norm(c)
                states.append((f"lowmix{r}", v @ c))
            for name, vec in states:
                psi = vec.reshape(Nt, d)
                lam = energy(psi, H)
                L = ext_loss(psi, P, Q, Msys, dreg, dsys)
                m = psi.mean(axis=0)
                dev2 = float(((psi - m[None, :]) ** 2).sum())
                ok2 = L <= CG * Nt * lam + 1e-10
                ok3 = dev2 <= Nt * lam / 4 + 1e-10
                allok &= ok2 and ok3
                rec["instances"].append(dict(inst=inst, T=T, state=name, lam=lam, NtLam=Nt * lam, eps_opt=eps_opt,
                                             L_ext=L, CG=CG, ratio_L_over_NtLam=L / (Nt * lam),
                                             ok_FKlog=bool(ok2), dev2=dev2, ok_factor=bool(ok3),
                                             gs_L_over_eps=(L / eps_opt) if name == "ground" else None))
            print(f"inst {inst} T={T:2d} eps={eps_opt:.3e} gs L/eps={rec['instances'][-4]['L_ext']/eps_opt:.4f} "
                  f"max L/(N lam)={max(r['ratio_L_over_NtLam'] for r in rec['instances'][-4:]):.3f} CG={CG:.1f}",
                  flush=True)
    # (4) adversarial hybrid states: chi_bad far from the corner, morph to chi_opt near (T,T) along the Green profile
    P, Q, Msys, dreg, dsys = random_instance(4, 4, 6)
    w_s, v_s = np.linalg.eigh(Msys)
    chi_opt, chi_bad = v_s[:, 0], v_s[:, -1]
    reg = None
    # reconstruct reg from P: P = |reg><reg| (x) I
    Preg = P[::dsys, ::dsys]
    wr, vr = np.linalg.eigh(Preg)
    reg = vr[:, -1]
    for T in [4, 8, 16, 32, 64]:
        n = T + 1
        Nt = n * n
        # harmonic profile h(tau) = G(tau, c) normalized: h(c) = 0 at corner, -> 1 far away (Green-function morph)
        Lg = grid_lap(T).toarray() if Nt <= 1100 else None
        if Lg is None:
            break
        Lp = np.linalg.pinv(Lg)
        c = Nt - 1
        g = Lp[:, c] - Lp[c, c]  # g(c) = 0, negative elsewhere
        prof = g / g.min()  # 0 at the corner, 1 at the farthest point
        best = None
        for theta in [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, math.pi / 2]:
            chi_far = math.cos(theta) * chi_opt + math.sin(theta) * chi_bad
            psi = np.zeros((Nt, dreg * dsys))
            for t in range(Nt):
                chi = (1 - prof[t]) * chi_opt + prof[t] * chi_far
                psi[t] = np.kron(reg, chi)
            psi /= np.linalg.norm(psi)
            H, _, _ = build_H(T, P, Q)
            lam = energy(psi, H)
            L = ext_loss(psi, P, Q, Msys, dreg, dsys)
            r = L / (Nt * lam)
            if best is None or r > best[0]:
                best = (r, theta, lam, L)
        G = corner_green(T)
        rec["adversarial"].append(dict(T=T, G=G, best_ratio=best[0], theta=best[1], lam=best[2], L_ext=best[3]))
        print(f"adversarial T={T}: max L_ext/(N lam) = {best[0]:.3f} (G = {G:.3f})", flush=True)
    rec["all_ok"] = bool(allok)
    print("ALL OK" if allok else "FAILURE", flush=True)
    with open(os.path.join(OUT, "check_fk_log.json"), "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()

"""Corner resistance-to-uniform G_k(T) = L^+_{cc} on the k-dimensional grid {0..T}^k (k = 1, 2, 3), exact eigen-sum.
k = 1: one clock (single path), k = 2: two-clock referee cell, k = 3: two clocks plus an idle third clock.
Also a direct 3-clock extraction check on random instances (small T): loss <= C_3 N_tau lambda with C_3 = 2(3+4 sqrt G_3)^2 + 1/2.
"""
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(29102026)


def spec(T):
    n = T + 1
    a = np.arange(n)
    mu = 2.0 - 2.0 * np.cos(np.pi * a / n)
    ph = np.where(a == 0, 1.0 / n, (2.0 / n) * np.cos(np.pi * a / (2 * n)) ** 2)
    return mu, ph


def G(T, k):
    mu, ph = spec(T)
    if k == 1:
        return float((ph[1:] / mu[1:]).sum())
    if k == 2:
        M = mu[:, None] + mu[None, :]
        W = ph[:, None] * ph[None, :]
    else:
        M = mu[:, None, None] + mu[None, :, None] + mu[None, None, :]
        W = ph[:, None, None] * ph[None, :, None] * ph[None, None, :]
    M.flat[0] = 1.0
    W.flat[0] = 0.0
    return float((W / M).sum())


def path_lap(n):
    main = np.full(n, 2.0)
    main[0] = main[-1] = 1.0
    off = -np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr")


def ext_check(T, dreg=2, dsys=4, rankQ=6):
    reg = rng.normal(size=dreg); reg /= np.linalg.norm(reg)
    P = np.kron(np.outer(reg, reg), np.eye(dsys))
    d = dreg * dsys
    A = rng.normal(size=(d, rankQ)); Qb, _ = np.linalg.qr(A); Q = Qb @ Qb.T
    V = np.kron(reg.reshape(dreg, 1), np.eye(dsys)); Msys = V.T @ Q @ V
    n = T + 1
    I = sp.identity(n, format="csr"); Lp = path_lap(n)
    L3 = sp.kron(sp.kron(Lp, I), I) + sp.kron(sp.kron(I, Lp), I) + sp.kron(sp.kron(I, I), Lp)
    Nt = n ** 3
    E0 = sp.csr_matrix(([1.0], ([0], [0])), shape=(Nt, Nt))
    ET = sp.csr_matrix(([1.0], ([Nt - 1], [Nt - 1])), shape=(Nt, Nt))
    H = sp.kron(L3, sp.identity(d)) + sp.kron(E0, sp.csr_matrix(np.eye(d) - P)) + sp.kron(ET, sp.csr_matrix(Q))
    w, v = spla.eigsh(H.tocsr(), k=4, which="SA", tol=1e-12)
    eps = float(np.linalg.eigvalsh(Msys)[0])
    g3 = G(T, 3)
    C3 = 2 * (3 + 4 * math.sqrt(g3)) ** 2 + 0.5
    out = []
    for j in range(4):
        psi = v[:, j].reshape(Nt, d)
        lam = float(v[:, j] @ (H @ v[:, j]))
        L = 0.0
        for t in range(Nt):
            a = P @ psi[t]; b = psi[t] - a
            L += float(a @ Q @ a)
            B = b.reshape(dreg, dsys); L += float(np.trace(Msys @ (B.T @ B)))
        out.append(dict(state=j, lam=lam, NtLam=Nt * lam, L=L, ratio=L / (Nt * lam), C3=C3, ok=bool(L <= C3 * Nt * lam + 1e-10),
                        L_over_eps=L / eps if eps > 1e-12 else None))
    return eps, out


def main():
    rec = {"G": [], "ext3": []}
    for T in [2, 4, 8, 16, 32, 64, 128, 256]:
        row = dict(T=T, G1=G(T, 1), G2=G(T, 2), G3=G(T, 3))
        rec["G"].append(row)
        print(f"T={T:4d}  G1={row['G1']:9.3f} (T/3={T/3:7.2f})  G2={row['G2']:6.3f}  G3={row['G3']:6.4f}", flush=True)
    ok = True
    for T in [2, 4, 6, 8]:
        for rep in range(3):
            eps, out = ext_check(T)
            ok &= all(o["ok"] for o in out)
            rec["ext3"].append(dict(T=T, rep=rep, eps=eps, states=out))
            print(f"3-clock T={T} rep={rep}: eps={eps:.3e} gs L/eps={(out[0]['L_over_eps'] or float('nan')):.4f} "
                  f"max L/(N lam)={max(o['ratio'] for o in out):.3f} C3={out[0]['C3']:.1f}", flush=True)
    rec["all_ok"] = bool(ok)
    print("ALL OK" if ok else "FAILURE", flush=True)
    json.dump(rec, open(os.path.join(OUT, "check_clock_dimension.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

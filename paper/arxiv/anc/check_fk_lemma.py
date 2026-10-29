"""Check of Lemma FK (strategy extraction from a two-clock Feynman-Kitaev game cell).

Rewound form on the legal clock space (Alice clock t_A, Bob clock t_B in {0..T}):
    H' = L_grid (x) I + |00><00| (x) (I - P) + |TT><TT| (x) Q,
P = |pi,0><pi,0| (x) I_sys   (questions in |pi>, answer registers in |0>),
Q = U^dag Pi_lose U          (U = U_A (x) U_B, the two players' measurement circuits).

Claims checked (exact diagonalization, double precision):
  (a) lambda_0 <= eps_opt / N_tau, N_tau = (T+1)^2, eps_opt = min loss over initialized inputs;
  (b) the extracted strategy (rewind locally, ignore clocks and old registers, insert fresh questions)
      has loss L_ext <= 5 (1 + sqrt(2T))^2 eps_opt;
  (c) L_ext >= eps_opt (sanity: it is a legitimate strategy with the same circuits).
Games: CHSH with qubit systems and several measurement-angle choices (random and optimal), and random
projector instances (P, Q) with a random 'loss operator' on the system for the (I-P) part.
Single-threaded numpy, small dimensions.
"""
import json
import math
import sys

import numpy as np

rng = np.random.default_rng(20261028)


def grid_laplacian(T):
    n = T + 1
    Lp = np.zeros((n, n))
    for t in range(T):
        Lp[t, t] += 1
        Lp[t + 1, t + 1] += 1
        Lp[t, t + 1] -= 1
        Lp[t + 1, t] -= 1
    I = np.eye(n)
    return np.kron(Lp, I) + np.kron(I, Lp)


def ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def chsh_instance(angA, angB):
    """Registers ordered (qA, qB, aA, aB, sA, sB), each a qubit; dims 2^6 = 64.
    Alice: on question x rotate sA by -angA[x], then CNOT sA -> aA (measure in rotated basis).
    Bob likewise. Lose iff aA xor aB != x AND y."""
    d = 64

    def idx(qA, qB, aA, aB, sA, sB):
        return (((((qA * 2 + qB) * 2 + aA) * 2 + aB) * 2 + sA) * 2 + sB)

    U = np.zeros((d, d))
    for qA in range(2):
        for qB in range(2):
            RA = ry(-angA[qA])
            RB = ry(-angB[qB])
            R = np.kron(RA, RB)  # on (sA, sB)
            for aA in range(2):
                for aB in range(2):
                    for sA in range(2):
                        for sB in range(2):
                            col = idx(qA, qB, aA, aB, sA, sB)
                            for sA2 in range(2):
                                for sB2 in range(2):
                                    amp = R[sA2 * 2 + sB2, sA * 2 + sB]
                                    if amp == 0:
                                        continue
                                    row = idx(qA, qB, aA ^ sA2, aB ^ sB2, sA2, sB2)
                                    U[row, col] += amp
    Pi = np.zeros((d, d))
    for qA in range(2):
        for qB in range(2):
            for aA in range(2):
                for aB in range(2):
                    for sA in range(2):
                        for sB in range(2):
                            if (aA ^ aB) != (qA & qB):
                                k = idx(qA, qB, aA, aB, sA, sB)
                                Pi[k, k] = 1.0
    Q = U.T @ Pi @ U
    # P: questions in |pi> = uniform |+>|+>, answers |0>, systems free
    pi = np.zeros(4)
    pi[:] = 0.5  # sqrt(1/4) on each (qA,qB)
    reg = np.zeros(16)  # (qA,qB,aA,aB)
    for qA in range(2):
        for qB in range(2):
            reg[((qA * 2 + qB) * 2 + 0) * 2 + 0] = 0.5
    Preg = np.outer(reg, reg)
    P = np.kron(Preg, np.eye(4))
    # honest loss operator on systems: M_sys with loss(chi) = <reg,chi|Q|reg,chi>
    V = np.kron(reg.reshape(16, 1), np.eye(4))  # 64 x 4 isometry chi -> reg (x) chi
    Msys = V.T @ Q @ V
    return P, Q, Msys, 16, 4


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


def run(P, Q, Msys, dreg, dsys, T):
    n = T + 1
    Nt = n * n
    d = dreg * dsys
    Lg = grid_laplacian(T)
    E00 = np.zeros((Nt, Nt)); E00[0, 0] = 1
    ETT = np.zeros((Nt, Nt)); ETT[Nt - 1, Nt - 1] = 1
    I_d = np.eye(d)
    H = np.kron(Lg, I_d) + np.kron(E00, I_d - P) + np.kron(ETT, Q)
    w, v = np.linalg.eigh(H)
    lam = w[0]
    psi = v[:, 0].reshape(Nt, d)
    # eps_opt: min over chi of loss = min eig of Msys
    eps_opt = float(np.linalg.eigvalsh(Msys)[0])
    # extracted strategy loss: sum_tau [ <P psi|Q|P psi> + honest loss of (I-P) psi part ]
    L_ext = 0.0
    IminusP = I_d - P
    for t in range(Nt):
        a = P @ psi[t]
        b = IminusP @ psi[t]
        L_ext += float(a @ Q @ a)
        # honest loss of the (I-P) component: reduce to the system (trace out old registers)
        B = b.reshape(dreg, dsys)
        rho_sys = B.T @ B
        L_ext += float(np.trace(Msys @ rho_sys))
    bound = 5 * (1 + math.sqrt(2 * T)) ** 2 * eps_opt
    return dict(T=T, lam=float(lam), lam_times_Nt=float(lam * Nt), eps_opt=eps_opt,
                L_ext=L_ext, bound=bound,
                ok_a=bool(lam <= eps_opt / Nt + 1e-12),
                ok_b=bool(L_ext <= bound + 1e-12),
                ok_c=bool(L_ext >= eps_opt - 1e-12),
                ratio_Lext_eps=(L_ext / eps_opt) if eps_opt > 1e-14 else None)


def main():
    records = []
    allok = True
    opt = ([0.0, math.pi / 2], [math.pi / 4, -math.pi / 4])
    angle_sets = [opt] + [([rng.uniform(0, math.pi), rng.uniform(0, math.pi)],
                           [rng.uniform(0, math.pi), rng.uniform(0, math.pi)]) for _ in range(4)]
    for k, (aA, aB) in enumerate(angle_sets):
        P, Q, Msys, dreg, dsys = chsh_instance(aA, aB)
        for T in (1, 2, 3, 4):
            r = run(P, Q, Msys, dreg, dsys, T)
            r["instance"] = "CHSH-opt" if k == 0 else f"CHSH-rand{k}"
            records.append(r)
            allok &= r["ok_a"] and r["ok_b"] and r["ok_c"]
            print(r, flush=True)
    for k in range(12):
        dreg = int(rng.integers(2, 5)); dsys = int(rng.integers(2, 5)); rankQ = int(rng.integers(1, dreg * dsys))
        P, Q, Msys, dreg, dsys = random_instance(dreg, dsys, rankQ)
        for T in (1, 2, 3, 5):
            r = run(P, Q, Msys, dreg, dsys, T)
            r["instance"] = f"random{k}(dreg={dreg},dsys={dsys},rankQ={rankQ})"
            records.append(r)
            allok &= r["ok_a"] and r["ok_b"] and r["ok_c"]
            print(r, flush=True)
    worst = max(r["L_ext"] / r["bound"] for r in records if r["bound"] > 1e-14)
    print("max L_ext / bound =", worst, flush=True)
    print("ALL PASS" if allok else "FAIL", flush=True)
    with open(sys.argv[1] if len(sys.argv) > 1 else "check_fk_lemma.json", "w") as f:
        json.dump(dict(records=records, all_pass=allok, max_ratio_to_bound=worst), f, indent=1)


if __name__ == "__main__":
    main()

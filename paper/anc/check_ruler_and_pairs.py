"""Two light checks.

(1) Frustrated-cell staircase (Lemma S'): F(p) = -g/p + (r + e(p-1))/p^2 with r = 1/16 and arbitrary
    cell energies 0 <= e(h) <= c h^-3.  Claim: the shifted ties g'_p = g_p + p(p+1)[e(p-1)/p^2 - e(p)/(p+1)^2]
    (g_p = r(2p+1)/(p(p+1))) are strictly decreasing for p >= p0 = ceil(4c/r) + 2, the minimizer of F over p >= 1
    equals p on (g'_p, g'_{p-1}) for small g, and p*(g) g -> 2r.  Checked on random e and adversarial e.
(2) Bell-pair cell robustness (Theorem R-E, cell level): the pair Hamiltonian (I - |Phi><Phi|) + v (x) I + I (x) v'
    with norm-bounded fields. Samples test finite gaps and ranks, not a uniform full-rank theorem.
"""
import json
import math
import sys

import numpy as np

rng = np.random.default_rng(29)
r = 1.0 / 16


def staircase_check(e, P=4000):
    def F(p, g):
        return -g / p + (r + e(p - 1)) / p ** 2
    gp = lambda p: r * (2 * p + 1) / (p * (p + 1))
    gprime = [None, None] + [gp(p) + p * (p + 1) * (e(p - 1) / p ** 2 - e(p) / (p + 1) ** 2) for p in range(2, P + 2)]
    # monotonicity from p0
    viol = [p for p in range(3, P + 1) if not (gprime[p] < gprime[p - 1])]
    p0 = (max(viol) + 1) if viol else 2
    # minimizer check on midpoints of shifted intervals
    bad = 0
    tested = 0
    for p in range(max(p0 + 1, 20), P, 7):
        g = 0.5 * (gprime[p] + gprime[p - 1])
        ps = np.arange(1, 3 * P)
        vals = np.array([F(int(q), g) if q >= 2 else -g + r + e(0) for q in ps])
        q = int(ps[np.argmin(vals)])
        tested += 1
        if q != p:
            bad += 1
    return dict(p0=p0, tested=tested, bad=bad, pg_at_P=(P - 1) * gprime[P - 1])


def pairs_check():
    out = []
    ok = True
    for beta in (1, 2, 3):
        D = 2 ** beta
        Phi = np.zeros(D * D);
        for z in range(D):
            Phi[z * D + z] = 1 / math.sqrt(D)
        H0 = np.eye(D * D) - np.outer(Phi, Phi)
        for eps in (0.01, 0.05, 0.1, 0.2):
            for _ in range(5):
                def rh():
                    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
                    A = (A + A.conj().T) / 2
                    return eps * A / np.linalg.norm(A, 2)
                v, vp = rh(), rh()
                H = H0 + np.kron(v, np.eye(D)) + np.kron(np.eye(D), vp)
                w, U = np.linalg.eigh(H)
                psi = U[:, 0].reshape(D, D)
                sv = np.linalg.svd(psi, compute_uv=False)
                rank = int(np.sum(sv > 1e-10))
                p = sv ** 2
                S = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
                gap = float(w[1] - w[0])
                rec = dict(beta=beta, eps=eps, schmidt_rank=rank, full=(rank == D), S_nats=S,
                           S_over_betaln2=S / (beta * math.log(2)), gap=gap)
                ok &= (rank == D) and gap > 1 - 4 * eps - 1e-12
                out.append(rec)
        # frozen site stays product trivially (single site); record ground energy shift only
    return out, ok


def main():
    res = {}
    allok = True
    for c in (0.0, 0.01, 0.05):
        for trial in range(3):
            tbl = rng.uniform(0, 1, size=20000)
            e = (lambda tbl, c: (lambda h: c * tbl[h] * max(h, 1) ** -3))(tbl, c)
            s = staircase_check(e)
            s["c"] = c
            s["p0_predicted"] = math.ceil(4 * c / r) + 2
            ok = s["bad"] == 0 and s["p0"] <= s["p0_predicted"]
            s["ok"] = ok
            allok &= ok
            res[f"random_c{c}_t{trial}"] = s
            print(s, flush=True)
    # adversarial: alternating e(h) = c h^-3 on odd h, 0 on even h
    c = 0.05
    e = lambda h: c * max(h, 1) ** -3 if h % 2 else 0.0
    s = staircase_check(e); s["c"] = c; s["p0_predicted"] = math.ceil(4 * c / r) + 2
    s["ok"] = s["bad"] == 0 and s["p0"] <= s["p0_predicted"]; allok &= s["ok"]
    res["alternating_c0.05"] = s
    print(s, flush=True)
    pr, ok2 = pairs_check()
    allok &= ok2
    print("pairs: all full Schmidt rank and gap > 1-4eps:", ok2, flush=True)
    for rec in pr[::5]:
        print(rec, flush=True)
    res["pairs"] = pr
    res["all_pass"] = allok
    print("ALL PASS" if allok else "FAIL", flush=True)
    with open(sys.argv[1] if len(sys.argv) > 1 else "check_ruler_and_pairs.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

"""
SCRIPT VERSION: v3-FINAL  (epsilon-constraint + intersection point + alpha sweep)
Run:  %run pharma_milp_epsilon_FINAL.py     (needs pharma_common.py and tradeoff_tools.py in the same folder)

For BOTH reliability measures (METRICS) it: sweeps epsilon, refines the interval where the normalised cost and
reliability objectives cross, solves exactly at the intersection (compromise point), saves CSV + PNG.
Then runs the alpha sweep. Outputs:
  epsilon_results_avg.csv / _worst.csv, cost_reliability_tradeoff_avg.png / _worst.png, alpha_results.csv, alpha_tradeoff.png
"""
import csv
from pharma_common import *
from tradeoff_tools import find_bracket, interpolate_crossing, normalise, plot_tradeoff, plot_alpha

print("SCRIPT VERSION: v3-FINAL (epsilon + intersection)")
METRICS = ["worst", "avg"]
N_COARSE, N_REFINE = 6, 8
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TOL = 1e-6

def solve(**kw):
    m, Z1, terms, V = build_model(**kw)
    m.Params.MIPGap = 1e-4
    m.Params.TimeLimit = 120
    m.optimize()
    return m, V

m_b, V_b = solve(name="Baseline_CostOnly")
a_b, w_b = reliability(V_b)[:2]
cost_base = m_b.objVal
print(f"Baseline (cost-minimal): cost {cost_base:,.0f} | R_avg {a_b:.4f} | R_worst {w_b:.4f} | {open_summary(V_b)}")

def run_eps(eps, metric, tag=""):
    m_e, V_e = solve(epsilon=max(eps, 0.0), metric=metric, name=f"Eps_{metric}_{eps:.5f}")
    if m_e.SolCount == 0:
        print(f"{eps:>9.5f} | {'--':>14} | INFEASIBLE (status {m_e.status}) {tag}"); return None
    ra, rw = reliability(V_e)[:2]
    p = {"eps": eps, "cost": m_e.objVal, "R": ra if metric == "avg" else rw, "R_avg": ra, "R_worst": rw, "net": open_summary(V_e)}
    print(f"{eps:>9.5f} | {p['cost']:>14,.0f} | {ra:>7.4f} | {rw:>7.4f} | {p['net']} {tag}")
    return p

def part_a(metric):
    print(f"\n=== PART A [{metric.upper()}]: cost vs {metric} reliability (epsilon-constraint) ===")
    m_r, _ = solve(maximize_R=True, metric=metric, name=f"Find_Rmax_{metric}")
    R_max = m_r.objVal
    R_base = a_b if metric == "avg" else w_b
    print(f"R_max ({metric}) = {R_max:.4f} | R at min-cost solution = {R_base:.4f}")
    print(f"{'epsilon':>9} | {'Cost':>14} | {'R_avg':>7} | {'R_worst':>7} | Network"); print("-" * 105)
    pts = []
    for i in range(N_COARSE):
        p = run_eps(R_base + i / (N_COARSE - 1) * (R_max - R_base) - TOL, metric)
        if p: pts.append(p)
    br = find_bracket(pts)
    if br:
        a, b = br
        print(f"\nIntersection of normalised objectives lies between R={a['R']:.4f} and R={b['R']:.4f} - refining ...")
        for j in range(1, N_REFINE + 1):
            p = run_eps(a["eps"] + j / (N_REFINE + 1) * (b["eps"] - a["eps"]), metric, "(refine)")
            if p: pts.append(p)
    cross = interpolate_crossing(pts); final = None
    if cross:
        print(f"\nInterpolated intersection: R = {cross['R']:.4f}, cost ~ {cross['cost']:,.0f}, both objectives ~{cross['mu']*100:.0f}% satisfied")
        final = run_eps(cross["R"], metric, "<-- COMPROMISE POINT")
        if final:
            _, Rmin, Rmx, Cmin, Cmax = normalise(pts + [final])
            print(f"  Compromise: cost {final['cost']:,.0f} (+{100*(final['cost']-cost_base)/cost_base:.2f}% vs min-cost) | "
                  f"R_avg {final['R_avg']:.4f} | R_worst {final['R_worst']:.4f} | network {final['net']}")
            print(f"  Reliability gain achieved: {100*(final['R']-Rmin)/(Rmx-Rmin):.0f}% of attainable | cost headroom unused: {100*(Cmax-final['cost'])/(Cmax-Cmin):.0f}%")
            pts.append(final)
    else:
        print("\nNo crossing found - inspect the table.")
    srt = normalise(pts)[0]
    with open(f"epsilon_results_{metric}.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["epsilon", "total_cost", "R_avg", "R_worst", "mu_rel", "mu_cost", "network"])
        for p in srt: wr.writerow([p["eps"], p["cost"], p["R_avg"], p["R_worst"], p["mu_rel"], p["mu_cost"], p["net"]])
    try:
        plot_tradeoff(pts, f"cost_reliability_tradeoff_{metric}.png",
                      crossing=({"R": final["R"], "cost": final["cost"], "mu": cross["mu"]} if final and cross else None),
                      title_suffix=f" [{metric} reliability]")
        print(f"Saved cost_reliability_tradeoff_{metric}.png")
    except Exception as ex:
        print("Plot skipped:", ex)

for mt in METRICS:
    part_a(mt)

print("\n=== PART B: alpha sweep ===")
print(f"{'alpha':>6} | {'Cost':>14} | {'R_avg':>7} | {'R_worst':>7} | {'Cost incr.':>10} | Network"); print("-" * 105)
rowsB = []
for a_ in ALPHAS:
    m_a, V_a = solve(alpha_val=a_, name=f"Alpha_{a_}")
    if m_a.SolCount > 0:
        ra, rw = reliability(V_a)[:2]; cost = m_a.objVal
        print(f"{a_:>6.2f} | {cost:>14,.0f} | {ra:>7.4f} | {rw:>7.4f} | {100*(cost-cost_base)/cost_base:>9.2f}% | {open_summary(V_a)}")
        rowsB.append((a_, cost, ra, rw, open_summary(V_a)))
    else:
        print(f"{a_:>6.2f} | INFEASIBLE (status {m_a.status})")
with open("alpha_results.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["alpha", "total_cost", "R_avg", "R_worst", "network"]); wr.writerows(rowsB)
try:
    plot_alpha(rowsB, "alpha_tradeoff.png"); print("Saved alpha_tradeoff.png")
except Exception as ex:
    print("Plot skipped:", ex)
print("DONE (v3-FINAL)")

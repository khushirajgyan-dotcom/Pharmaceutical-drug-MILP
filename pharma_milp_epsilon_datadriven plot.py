"""
Trade-off analysis (city scale). Run:  python pharma_milp_epsilon_datadriven.py
Needs pharma_common.py and tradeoff_tools.py in the same folder (matplotlib optional, for the PNG charts).

PART A - epsilon-constraint: min cost s.t. reliability >= epsilon (reliability = fill rate under disruption).
         Coarse sweep -> refine around the interval where the normalised cost and reliability objectives cross ->
         solve exactly at the interpolated intersection ("compromise point", max-min satisfaction).
PART B - alpha sweep (minimum backup coverage).
Outputs: epsilon_results.csv, alpha_results.csv, cost_reliability_tradeoff.png, alpha_tradeoff.png
"""
import csv
from pharma_common import *
from tradeoff_tools import find_bracket, interpolate_crossing, normalise, plot_tradeoff, plot_alpha

METRIC = "avg"            # "avg" or "worst": which reliability measure the epsilon constraint acts on
N_COARSE = 6
N_REFINE = 8
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TOL = 1e-6

def solve(**kw):
    m, Z1, terms, V = build_model(**kw)
    m.Params.MIPGap = 1e-4
    m.Params.TimeLimit = 120
    m.optimize()
    return m, V

def run_eps(eps, tag=""):
    m_e, V_e = solve(epsilon=max(eps, 0.0), metric=METRIC, name=f"Eps_{eps:.5f}")
    if m_e.SolCount == 0:
        print(f"{eps:>9.5f} | {'--':>14} | INFEASIBLE (status {m_e.status}) {tag}")
        return None
    ra, rw = reliability(V_e)[:2]
    p = {"eps": eps, "cost": m_e.objVal, "R": ra if METRIC == "avg" else rw, "R_avg": ra, "R_worst": rw, "net": open_summary(V_e)}
    print(f"{eps:>9.5f} | {p['cost']:>14,.0f} | {ra:>7.4f} | {rw:>7.4f} | {p['net']} {tag}")
    return p

# ---------------- PART A ----------------
print("=== PART A: epsilon-constraint (cost vs reliability) ===")
m_r, V_r = solve(maximize_R=True, metric=METRIC, name="Find_Rmax")
R_max = m_r.objVal
m_b, V_b = solve(name="Baseline_CostOnly")
a_b, w_b = reliability(V_b)[:2]
R_base = a_b if METRIC == "avg" else w_b
cost_base = m_b.objVal
print(f"R_max ({METRIC}) = {R_max:.4f} | baseline: cost {cost_base:,.0f}, R_avg {a_b:.4f}, R_worst {w_b:.4f}\n")
print(f"{'epsilon':>9} | {'Cost':>14} | {'R_avg':>7} | {'R_worst':>7} | Network")
print("-" * 105)

points = []
for i in range(N_COARSE):
    e = R_base + i / (N_COARSE - 1) * (R_max - R_base) - TOL
    p = run_eps(e)
    if p: points.append(p)

br = find_bracket(points)
if br:
    a, b = br
    print(f"\nIntersection of normalised objectives lies between R={a['R']:.4f} and R={b['R']:.4f} - refining ...")
    for j in range(1, N_REFINE + 1):
        e = a["eps"] + j / (N_REFINE + 1) * (b["eps"] - a["eps"])
        p = run_eps(e, "(refine)")
        if p: points.append(p)

cross = interpolate_crossing(points)
final = None
if cross:
    print(f"\nInterpolated intersection: R = {cross['R']:.4f}, cost ~ {cross['cost']:,.0f}, both objectives ~{cross['mu']*100:.0f}% satisfied")
    print("Solving exactly at the intersection:")
    final = run_eps(cross["R"], "<-- COMPROMISE POINT")
    if final:
        pts_n, Rmin, Rmax_, Cmin, Cmax = normalise(points)
        mu_rel = (final["R"] - Rmin) / (Rmax_ - Rmin); mu_cost = (Cmax - final["cost"]) / (Cmax - Cmin)
        print(f"  Compromise: cost {final['cost']:,.0f} (+{100*(final['cost']-cost_base)/cost_base:.2f}% vs min-cost), "
              f"R_avg {final['R_avg']:.4f}, R_worst {final['R_worst']:.4f}")
        print(f"  Satisfaction: reliability {mu_rel*100:.0f}% of attainable gain | cost headroom unused {mu_cost*100:.0f}%")
        points.append(final)
else:
    print("\nNo crossing found (one objective dominates) - inspect the table.")

pts_sorted = normalise(points)[0]
with open("epsilon_results.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["epsilon", "total_cost", "R_avg", "R_worst", "mu_rel", "mu_cost", "network"])
    for p in pts_sorted: wr.writerow([p["eps"], p["cost"], p["R_avg"], p["R_worst"], p["mu_rel"], p["mu_cost"], p["net"]])
try:
    plot_tradeoff(points, "cost_reliability_tradeoff.png",
                  crossing=({"R": final["R"], "cost": final["cost"], "mu": cross["mu"]} if final and cross else None))
    print("Saved cost_reliability_tradeoff.png")
except Exception as ex:
    print("Plot skipped (", ex, ") - install matplotlib: pip install matplotlib")

# ---------------- PART B ----------------
print("\n=== PART B: alpha sweep (minimum backup coverage of a disrupted DC's flow) ===")
print(f"{'alpha':>6} | {'Cost':>14} | {'R_avg':>7} | {'R_worst':>7} | {'Cost incr.':>10} | Network")
print("-" * 105)
rowsB = []
for a_ in ALPHAS:
    m_a, V_a = solve(alpha_val=a_, name=f"Alpha_{a_}")
    if m_a.SolCount > 0:
        ra, rw = reliability(V_a)[:2]; cost = m_a.objVal
        print(f"{a_:>6.2f} | {cost:>14,.0f} | {ra:>7.4f} | {rw:>7.4f} | {100*(cost-cost_base)/cost_base:>9.2f}% | {open_summary(V_a)}")
        rowsB.append((a_, cost, ra, rw, open_summary(V_a)))
    else:
        print(f"{a_:>6.2f} | {'--':>14} | INFEASIBLE (status {m_a.status})")
with open("alpha_results.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["alpha", "total_cost", "R_avg", "R_worst", "network"]); wr.writerows(rowsB)
try:
    plot_alpha(rowsB, "alpha_tradeoff.png"); print("\nSaved alpha_tradeoff.png")
except Exception as ex:
    print("Plot skipped (", ex, ")")
print("Saved epsilon_results.csv and alpha_results.csv")

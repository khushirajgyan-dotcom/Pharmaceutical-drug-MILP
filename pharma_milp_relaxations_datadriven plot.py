"""
Linear Relaxation and Lagrangian Relaxation (city scale, shared model).
Run:  python pharma_milp_relaxations_datadriven.py

Lagrangian (stabilised): resilience constraints are dualised with ONE multiplier per (failed DC, case, product, period)
(equal across customers) -> 300 multipliers instead of 6,600. Polyak step with UB = exact optimum; if an iteration is much
worse than the best bound, multipliers revert to the best point and the step is halved (prevents the divergence seen before).
Iteration 1 (mu = 0) is the model WITHOUT resilience constraints - itself a valid lower bound and a measure of the
price of resilience.
"""
from pharma_common import *

N_ITER = 40
theta_step = 1.0

m_exact, _, _, _ = build_model(name="Exact_MILP")
m_exact.Params.MIPGap = 0.0
m_exact.optimize()
exact_obj = m_exact.objVal
print(f"1. Exact MILP optimal objective              : {exact_obj:,.2f}")

m_lp, _, _, _ = build_model(relax_binaries=True, name="Linear_Relaxation")
m_lp.optimize()
lp_bound = m_lp.objVal
print(f"2. Linear Relaxation bound (binaries relaxed): {lp_bound:,.2f}  (gap {100*(exact_obj-lp_bound)/exact_obj:.2f}%)")

agg_keys = [(d, s, k, t) for (d, s) in fail_pairs for k in K for t in T]
mu = {key: 0.0 for key in agg_keys}
best_lb, best_mu, best_g = -float("inf"), dict(mu), None
no_improve, noresil_bound = 0, None

def cell_penalties(mu_agg):
    return {(d, s, c, k, t): mu_agg[(d, s, k, t)] for (d, s, c, k, t) in resilience_keys}

print(f"\n3. Lagrangian Relaxation ({len(agg_keys)} multipliers):")
for it in range(1, N_ITER + 1):
    m_lag, _, terms, _ = build_model(resilience_penalty=cell_penalties(mu), name=f"Lagrangian_{it}")
    m_lag.Params.MIPGap = 0.0
    m_lag.optimize()
    if m_lag.status != GRB.OPTIMAL:
        print(f"   Iteration {it}: solve status {m_lag.status}, stopping."); break
    lb = m_lag.ObjBound
    g = {key: 0.0 for key in agg_keys}
    for (d, s, c, k, t), (lhs, rhs) in terms.items():
        g[(d, s, k, t)] += rhs.getValue() - lhs.getValue()
    if it == 1:
        noresil_bound = lb
    note = ""
    if lb > best_lb + 1e-6:
        best_lb, best_mu, best_g, no_improve = lb, dict(mu), dict(g), 0
    else:
        no_improve += 1
        if no_improve >= 2:
            theta_step, no_improve = theta_step / 2.0, 0
    print(f"   Iteration {it:2d}: bound = {lb:,.2f}   (best: {best_lb:,.2f}, step factor {theta_step:.3f}){note}")
    diverged = lb < best_lb - 0.02 * exact_obj
    base_mu, base_g, base_lb = (best_mu, best_g, best_lb) if diverged else (mu, g, lb)
    if diverged:
        theta_step /= 2.0
        print("      -> too far below best bound: reverting to best multipliers, halving step")
    norm2 = sum(v * v for v in base_g.values())
    if norm2 < 1e-9:
        print("   Subgradient is zero - stopping."); break
    step = theta_step * max(exact_obj - base_lb, 0.0) / norm2
    mu = {key: max(0.0, base_mu[key] + step * base_g[key]) for key in agg_keys}

print("\nSummary:")
print(f"  Exact MILP optimum                          : {exact_obj:,.2f}")
print(f"  Linear Relaxation bound                     : {lp_bound:,.2f}  (gap {100*(exact_obj-lp_bound)/exact_obj:.2f}%)")
print(f"  Model without resilience constraints (mu=0) : {noresil_bound:,.2f}  -> price of resilience = {exact_obj-noresil_bound:,.0f} ({100*(exact_obj-noresil_bound)/noresil_bound:.1f}%)")
print(f"  Lagrangian bound (best)                     : {best_lb:,.2f}  (gap {100*(exact_obj-best_lb)/exact_obj:.2f}%)")

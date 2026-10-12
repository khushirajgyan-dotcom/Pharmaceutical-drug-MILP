"""
Base data-driven MILP (city scale). Uses the shared model in pharma_common.py.
Run:  python pharma_milp_datadriven_v2.py
Reports: optimum, facilities, DC throughput, cost/emission breakdown, expiry, and fill-rate reliability
under each disruption case.
"""
from pharma_common import *

peak = max(sum(D_ckt[c, k, t] for c in C for k in K) for t in T)
print(f"Sanity: peak monthly demand {peak:,.0f} | plant cap {sum(Cap_p.values()):,.0f} | "
      f"warehouse cap {sum(Cap_w.values()):,.0f} | DC cap {sum(Cap_d.values()):,.0f}")

m, Z1, terms, V = build_model(name="Pharma_MultiEchelon_DataDriven", quiet=False)
m.Params.MIPGap = 0.01
m.Params.TimeLimit = 600
m.optimize()

if m.SolCount > 0:
    print(f"\nObjective: {m.objVal:,.2f}  (MIP gap: {m.MIPGap*100:.2f}%)")
    print("Facilities opened:")
    for w in W:
        if V["Y_w"][w].X > 0.5: print(f"  Warehouse {w}: OPEN")
    for d in Dc:
        if V["Y_d"][d].X > 0.5:
            print(f"  DC {d}: OPEN" + (" [BACKUP]" if d in B and V["Z_d"][d].X > 0.5 else ""))
    tot = sum(V["X_dc"][d, c, k, t].X for d in Dc for c in C for k in K for t in T)
    print("Annual throughput by DC (units, share of total):")
    for d in Dc:
        f = sum(V["X_dc"][d, c, k, t].X for c in C for k in K for t in T)
        if f > 1e-6: print(f"  {d}: {f:>10,.0f}  ({100*f/tot:.1f}%)")
    print("\nEconomic cost components:")
    for n_, e_ in V["comps"].items(): print(f"  {n_:<16}{e_.getValue():>15,.0f}")
    print("Emission terms (units -> x lambda):")
    for n_, e_ in V["em"].items(): print(f"  {n_:<18}{e_.getValue():>12,.0f}  -> {lam*e_.getValue():>12,.0f}")
    print("Expired stock by product (units):", {k: round(sum(V["E_kt"][k, t].X for t in T), 1) for k in K})
    ravg, rworst, Rs = reliability(V)
    print(f"\nReliability (fill rate under disruption): average {ravg:.4f} | worst case {rworst:.4f}")
    for s, r in Rs.items(): print(f"  case {s:<8}: {r:.4f}")
elif m.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
    print("Model infeasible - writing IIS to infeasible.ilp. Likely cause: backup requirements vs DC capacity "
          "(try raising DC capacity in the Capacity sheet, lowering alpha, or FAILURE_MODE='listed' in pharma_common.py).")
    m.computeIIS(); m.write("infeasible.ilp")
else:
    print(f"Model status: {m.status}")

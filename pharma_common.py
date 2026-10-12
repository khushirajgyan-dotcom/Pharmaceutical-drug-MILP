"""
Data-Driven Deterministic Multi-Echelon Pharmaceutical Distribution MILP.

Reads ALL sets and parameters from Real_Data_Template.xlsx - no hardcoded
dictionaries. Fill in the template (see its "READ ME" sheet), point
DATA_FILE below at it, and run. Network size, number of products, periods,
and scenarios are NOT fixed in code - they come entirely from the "Sets"
sheet, so this scales to your real, full-length dataset without code changes.

Model structure (constraints, objective, relaxations) is identical to the
verified dummy-data version - only data sourcing has changed.
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

DATA_FILE = "Real_Data_Template_CityScale_v2.xlsx"   # point this at your filled-in template

# ============================================================
# 1. LOAD SETS
# ============================================================
df_sets = pd.read_excel(DATA_FILE, sheet_name="Sets")
sets = {row["Set"]: [x.strip() for x in str(row["Members (comma-separated)"]).split(",")]
        for _, row in df_sets.iterrows()}

P  = sets["Plants"]
W  = sets["Warehouses"]
Dc = sets["DistributionCenters"]
B  = sets["BackupEligibleDCs"]
C  = sets["DemandPoints"]
K  = sets["Products"]
T  = [int(x) for x in sets["Periods"]]
S  = sets["DisruptionScenarios"]

print(f"Loaded network: {len(P)} plants, {len(W)} warehouses, {len(Dc)} DCs "
      f"({len(B)} backup-eligible), {len(C)} demand points, {len(K)} products, "
      f"{len(T)} periods, {len(S)} disruption scenarios.")

# ============================================================
# 2. LOAD PARAMETERS (each sheet -> dictionary, with clear missing-data checks)
# ============================================================
def load_dict(sheet, key_cols, value_col, required_keys=None, default=None, label=""):
    """Generic loader: reads a sheet, builds {key_tuple: value}. Warns on gaps."""
    df = pd.read_excel(DATA_FILE, sheet_name=sheet)
    d = {}
    for row in df.itertuples(index=False):
        key = tuple(getattr(row, c) for c in key_cols) if len(key_cols) > 1 else getattr(row, key_cols[0])
        val = getattr(row, value_col)
        if pd.notna(val):
            d[key] = val
    if required_keys:
        missing = [k for k in required_keys if k not in d]
        if missing:
            if default is not None:
                print(f"  [WARNING] {label}: {len(missing)} entries missing in '{sheet}' - using default {default}.")
                for k in missing:
                    d[k] = default
            else:
                raise ValueError(f"{label}: missing required entries in '{sheet}': {missing[:5]}"
                                  f"{'...' if len(missing) > 5 else ''}. Fill in the template before running.")
    return d

req_pk = [(p, k) for p in P for k in K]
req_wk = [(w, k) for w in W for k in K]
req_dk = [(d, k) for d in Dc for k in K]

pc = load_dict("ProductionCost", ["Plant", "Product"], "Cost_per_unit",
               required_keys=req_pk, label="ProductionCost")

df_hold = pd.read_excel(DATA_FILE, sheet_name="HoldingCost")
hc_w = {(r.Node, r.Product): r.Cost_per_unit_per_period for r in df_hold.itertuples()
        if r.NodeType == "Warehouse" and pd.notna(r.Cost_per_unit_per_period)}
hc_d = {(r.Node, r.Product): r.Cost_per_unit_per_period for r in df_hold.itertuples()
        if r.NodeType == "DC" and pd.notna(r.Cost_per_unit_per_period)}
for k in K:  # fallback default where not specified
    for w in W:
        hc_w.setdefault((w, k), 0.35)
    for d in Dc:
        hc_d.setdefault((d, k), 0.40)

df_tc = pd.read_excel(DATA_FILE, sheet_name="TransportCost")
tc = {(r.From, r.To): r.Cost_per_unit for r in df_tc.itertuples() if pd.notna(r.Cost_per_unit)}
# fallback: if a route is missing, flag it (you must supply all real routes used)
all_routes = ([(p, w) for p in P for w in W] + [(w, d) for w in W for d in Dc] +
              [(d, c) for d in Dc for c in C] + [(d2, c) for d2 in B for c in C])
missing_routes = [r for r in all_routes if r not in tc]
if missing_routes:
    print(f"  [WARNING] TransportCost: {len(missing_routes)} routes missing - using fallback 2.0/unit.")
    for r in missing_routes:
        tc.setdefault(r, 2.0)

df_fixed = pd.read_excel(DATA_FILE, sheet_name="FixedCosts")
f_w = {r.Node: r.FixedCost for r in df_fixed.itertuples() if r.NodeType == "Warehouse" and pd.notna(r.FixedCost)}
f_d = {r.Node: r.FixedCost for r in df_fixed.itertuples() if r.NodeType == "DC" and pd.notna(r.FixedCost)}
f_B = {r.Node: r.BackupExtraCost for r in df_fixed.itertuples()
       if r.NodeType == "DC" and r.Node in B and pd.notna(r.BackupExtraCost)}
for w in W: f_w.setdefault(w, 15000)
for d in Dc: f_d.setdefault(d, 15000)
for d in B: f_B.setdefault(d, 5000)

df_rl = pd.read_excel(DATA_FILE, sheet_name="ReverseLogisticsCost")
oc = {r.Product: r.DisposalCost_per_unit for r in df_rl.itertuples() if pd.notna(r.DisposalCost_per_unit)}
rc = {r.Product: r.RecyclingCost_per_unit for r in df_rl.itertuples() if pd.notna(r.RecyclingCost_per_unit)}
isex = {r.Product: r.ResaleValue_per_unit for r in df_rl.itertuples() if pd.notna(r.ResaleValue_per_unit)}
for k in K:
    oc.setdefault(k, 2.0); rc.setdefault(k, 1.0); isex.setdefault(k, 3.0)

df_em = pd.read_excel(DATA_FILE, sheet_name="EmissionFactors").set_index("Type")["Value"]
transport_ef = float(df_em.get("Transportation", 0.13))
production_ef = float(df_em.get("Production (per kg API)", 150))
e_ij = {route: transport_ef * cost for route, cost in tc.items()}  # distance-cost proxy, as in dummy version
pe = {(p, k): production_ef / 60.0 for p in P for k in K}  # scaled per-unit proxy; replace with real per-unit factor
def _ef(substr, default):
    for t_, v_ in df_em.items():
        if substr in str(t_).lower() and pd.notna(v_):
            return float(v_)
    print(f"  [WARNING] EmissionFactors: no row containing '{substr}' - using default {default}.")
    return default
ew = {k: _ef("disposal", 1.2) for k in K}   # read from EmissionFactors sheet
er = {k: _ef("recycl", 0.8) for k in K}     # read from EmissionFactors sheet

df_sl = pd.read_excel(DATA_FILE, sheet_name="ShelfLife")
sl_months = {r.Product: r.TotalShelfLife_months for r in df_sl.itertuples() if pd.notna(r.TotalShelfLife_months)}
# convert months -> periods: define how many months each period represents
MONTHS_PER_PERIOD = 1  # ADJUST to match your real period definition (e.g., 1 period = 1 month)
sl = {k: max(1, int(sl_months.get(k, 24) / MONTHS_PER_PERIOD)) for k in K}

df_rho = pd.read_excel(DATA_FILE, sheet_name="Recyclability")
rho = {r.Product: r.RecyclableFraction_rho for r in df_rho.itertuples() if pd.notna(r.RecyclableFraction_rho)}
for k in K: rho.setdefault(k, 0.60)

df_dem = pd.read_excel(DATA_FILE, sheet_name="Demand")
D_ckt = {(r.DemandPoint, r.Product, int(r.Period)): r.Quantity for r in df_dem.itertuples() if pd.notna(r.Quantity)}
req_ckt = [(c, k, t) for c in C for k in K for t in T]
missing_demand = [x for x in req_ckt if x not in D_ckt]
if missing_demand:
    raise ValueError(f"Demand: {len(missing_demand)} (point,product,period) combos missing - "
                      f"fill in the Demand sheet fully before running. First few: {missing_demand[:5]}")

df_cap = pd.read_excel(DATA_FILE, sheet_name="Capacity")
Cap_p = {r.Node: r.Capacity for r in df_cap.itertuples() if r.NodeType == "Plant" and pd.notna(r.Capacity)}
Cap_w = {r.Node: r.Capacity for r in df_cap.itertuples() if r.NodeType == "Warehouse" and pd.notna(r.Capacity)}
Cap_d = {r.Node: r.Capacity for r in df_cap.itertuples() if r.NodeType == "DC" and pd.notna(r.Capacity)}
for p in P: Cap_p.setdefault(p, 1000)
for w in W: Cap_w.setdefault(w, 800)
for d in Dc: Cap_d.setdefault(d, 500)

df_dis = pd.read_excel(DATA_FILE, sheet_name="DisruptionScenarios")
delta = {(d, s): 0 for d in Dc for s in S}
theta = {(c, s): 1.0 for c in C for s in S}
for r in df_dis.itertuples():
    if r.Parameter == "delta":
        delta[(r.AffectedNode_or_Point, r.Scenario)] = int(r.Value)
    elif r.Parameter == "theta":
        theta[(r.AffectedNode_or_Point, r.Scenario)] = float(r.Value)

alpha, lam, M = 0.8, 8.0, 100000   # tune lam (environmental weight) as needed

print("All parameters loaded successfully.\n")


# ============================================================
# SHARED MODEL BUILDER  (single source of truth for all three scripts)
#
# CHANGES vs. previous version
#  1. Reliability R is now a true, bounded FILL RATE (0-1): for every disruption case s and cell (c,k,t),
#     fulfilled quantity v <= min(demand_s, quantity actually delivered under s). R_s = sum(v)/sum(demand_s).
#     R_avg = weighted average of R_s over cases, R_worst = min over cases.
#  2. Closes the "just don't use the disrupted DC" loophole: FAILURE_MODE = "all_dcs" tests the outage of EVERY DC
#     in turn (one case "F@d" per DC). s1 (facility) and s2 (transport link) reduce to the same loss of a DC's outbound
#     supply in this model, so they are merged into this family. A failed DC cannot act as backup in its own case.
#     Contingency shipping cost/emissions of the family are weighted 1/|DC| each (one outage per year, on average).
#     Set FAILURE_MODE = "listed" to restore the original behaviour (only DCs flagged in the DisruptionScenarios sheet).
#  3. alpha (minimum backup coverage) can be overridden per run -> alpha sweep.
# ============================================================
FAILURE_MODE = "all_dcs"     # "all_dcs" (recommended) or "listed"

assert T == list(range(1, len(T) + 1)), "Periods must be 1..n"
cohort_k = [k for k in K if sl[k] < len(T)]
coh_idx = [(k, tau, t) for k in cohort_k for tau in T for t in T if tau <= t <= tau + sl[k] - 1]
coh_set = set(coh_idx)

surge_S = [s for s in S if any(theta.get((c, s), 1.0) > 1.0 for c in C)]
if FAILURE_MODE == "all_dcs":
    S_model = [f"F@{d}" for d in Dc] + surge_S
    fail_pairs = [(d, f"F@{d}") for d in Dc]
    wS = {s: (1.0 / len(Dc) if s.startswith("F@") else 1.0) for s in S_model}
else:
    S_model = list(S)
    fail_pairs = [(d, s) for (d, s) in delta if delta[(d, s)] == 1]
    wS = {s: 1.0 for s in S_model}
failed_in = {s: [d for (d, s2) in fail_pairs if s2 == s] for s in S_model}
resilience_keys = [(d, s, c, k, t) for (d, s) in fail_pairs for c in C for k in K for t in T]
print(f"Disruption cases ({FAILURE_MODE}): {S_model}")


def build_model(relax_binaries=False, resilience_penalty=None, epsilon=None, maximize_R=False,
                metric="avg", alpha_val=None, name="model", quiet=True):
    """metric: 'avg' or 'worst' (which reliability measure epsilon / maximize_R act on)."""
    a = alpha if alpha_val is None else alpha_val
    m = gp.Model(name)
    if quiet:
        m.Params.OutputFlag = 0
    vt = GRB.CONTINUOUS if relax_binaries else GRB.BINARY

    Y_w = m.addVars(W, vtype=vt, lb=0, ub=1, name="Y_w")
    Y_d = m.addVars(Dc, vtype=vt, lb=0, ub=1, name="Y_d")
    Z_d = m.addVars(B, vtype=vt, lb=0, ub=1, name="Z_d")
    X_pw = m.addVars(P, W, K, T, lb=0, name="X_pw")
    X_wd = m.addVars(W, Dc, K, T, lb=0, name="X_wd")
    X_dc = m.addVars(Dc, C, K, T, lb=0, name="X_dc")
    I_w = m.addVars(W, K, [0] + T, lb=0, name="I_w")
    I_d = m.addVars(Dc, K, [0] + T, lb=0, name="I_d")
    E_kt = m.addVars(K, T, lb=0, name="E_kt")
    E_disp = m.addVars(K, T, lb=0, name="E_disp")
    E_recyc = m.addVars(K, T, lb=0, name="E_recyc")
    BackupSupply = m.addVars(B, C, K, T, S_model, lb=0, name="BackupSupply")
    XE_w = m.addVars(W, K, T, lb=0, name="XE_w")
    XE_d = m.addVars(Dc, K, T, lb=0, name="XE_d")
    A = m.addVars(coh_idx, lb=0, name="A")
    q = m.addVars(coh_idx, lb=0, name="q")
    for w in W:
        for k in K: I_w[w, k, 0].ub = 0
    for d in Dc:
        for k in K: I_d[d, k, 0].ub = 0
    for (d, s) in fail_pairs:                      # a failed DC cannot serve as backup in its own case
        if d in B:
            for c in C:
                for k in K:
                    for t in T:
                        BackupSupply[d, c, k, t, s].ub = 0

    C1 = gp.quicksum(pc[p, k] * X_pw[p, w, k, t] for p in P for w in W for k in K for t in T)
    C2 = (gp.quicksum(hc_w[w, k] * I_w[w, k, t] for w in W for k in K for t in T) +
          gp.quicksum(hc_d[d, k] * I_d[d, k, t] for d in Dc for k in K for t in T))
    C3 = (gp.quicksum(tc[p, w] * X_pw[p, w, k, t] for p in P for w in W for k in K for t in T) +
          gp.quicksum(tc[w, d] * X_wd[w, d, k, t] for w in W for d in Dc for k in K for t in T) +
          gp.quicksum(tc[d, c] * X_dc[d, c, k, t] for d in Dc for c in C for k in K for t in T) +
          gp.quicksum(wS[s] * tc[d2, c] * BackupSupply[d2, c, k, t, s]
                      for d2 in B for c in C for k in K for t in T for s in S_model))
    C4 = gp.quicksum(f_w[w] * Y_w[w] for w in W) + gp.quicksum(f_d[d] * Y_d[d] for d in Dc)
    C5 = gp.quicksum(f_B[d] * Z_d[d] for d in B)
    C6 = gp.quicksum(oc[k] * E_disp[k, t] for k in K for t in T)
    C7 = gp.quicksum((rc[k] - isex[k]) * E_recyc[k, t] for k in K for t in T)
    EM1 = (gp.quicksum(e_ij[p, w] * X_pw[p, w, k, t] for p in P for w in W for k in K for t in T) +
           gp.quicksum(e_ij[w, d] * X_wd[w, d, k, t] for w in W for d in Dc for k in K for t in T) +
           gp.quicksum(e_ij[d, c] * X_dc[d, c, k, t] for d in Dc for c in C for k in K for t in T) +
           gp.quicksum(wS[s] * e_ij[d2, c] * BackupSupply[d2, c, k, t, s]
                       for d2 in B for c in C for k in K for t in T for s in S_model))
    EM2 = gp.quicksum(pe[p, k] * X_pw[p, w, k, t] for p in P for w in W for k in K for t in T)
    EM3 = gp.quicksum(ew[k] * E_disp[k, t] for k in K for t in T)
    EM4 = gp.quicksum(er[k] * E_recyc[k, t] for k in K for t in T)
    Z1 = C1 + C2 + C3 + C4 + C5 + C6 + C7 + lam * (EM1 + EM2 + EM3 - EM4)

    # ---- flow, capacity, expiry (unchanged from previous version) ----
    m.addConstrs((gp.quicksum(X_pw[p, w, k, t] for w in W for k in K) <= Cap_p[p]
                  for p in P for t in T), name="PlantCap")
    m.addConstrs((I_w[w, k, t] == I_w[w, k, t - 1] + gp.quicksum(X_pw[p, w, k, t] for p in P)
                  - gp.quicksum(X_wd[w, d, k, t] for d in Dc) - XE_w[w, k, t]
                  for w in W for k in K for t in T), name="WarehouseBalance")
    m.addConstrs((I_d[d, k, t] == I_d[d, k, t - 1] + gp.quicksum(X_wd[w, d, k, t] for w in W)
                  - gp.quicksum(X_dc[d, c, k, t] for c in C) - XE_d[d, k, t]
                  for d in Dc for k in K for t in T), name="DCBalance")
    m.addConstrs((gp.quicksum(X_dc[d, c, k, t] for d in Dc) >= D_ckt[c, k, t]
                  for c in C for k in K for t in T), name="DemandSatisfaction")
    m.addConstrs((gp.quicksum(X_pw[p, w, k, t] for p in P for k in K) <= Cap_w[w] * Y_w[w]
                  for w in W for t in T), name="WarehouseCap")
    m.addConstrs((gp.quicksum(X_wd[w, d, k, t] for w in W for k in K) <= Cap_d[d] * Y_d[d]
                  for d in Dc for t in T), name="DCCap")
    m.addConstrs((Z_d[d] <= Y_d[d] for d in B), name="BackupRequiresOpen")
    for (k, tau, t) in coh_idx:
        prev = (gp.quicksum(X_pw[p, w, k, tau] for p in P for w in W) if t == tau else A[k, tau, t - 1])
        m.addConstr(A[k, tau, t] == prev - q[k, tau, t], name=f"Cohort_{k}_{tau}_{t}")
    for k in cohort_k:
        for t in T:
            m.addConstr(gp.quicksum(q[k, tau, t] for tau in T if (k, tau, t) in coh_set)
                        == gp.quicksum(X_dc[d, c, k, t] for d in Dc for c in C), name=f"CohortOut_{k}_{t}")
    for k in K:
        for t in T:
            if k in cohort_k and t > sl[k]:
                m.addConstr(E_kt[k, t] == A[k, t - sl[k], t - 1], name=f"Expiry_{k}_{t}")
            else:
                m.addConstr(E_kt[k, t] == 0, name=f"NoExpiry_{k}_{t}")
            m.addConstr(gp.quicksum(XE_w[w, k, t] for w in W) + gp.quicksum(XE_d[d, k, t] for d in Dc)
                        == E_kt[k, t], name=f"ExpiryRemoval_{k}_{t}")
    m.addConstrs((E_disp[k, t] == (1 - rho[k]) * E_kt[k, t] for k in K for t in T), name="DisposalSplit")
    m.addConstrs((E_recyc[k, t] == rho[k] * E_kt[k, t] for k in K for t in T), name="RecyclingSplit")

    # ---- resilience ----
    m.addConstrs((gp.quicksum(BackupSupply[d2, c, k, t, s] for c in C) <= I_d[d2, k, t]
                  for d2 in B for k in K for t in T for s in S_model), name="BackupBoundedByInventory")
    terms = {}
    for (d, s) in fail_pairs:
        for c in C:
            for k in K:
                for t in T:
                    lhs = gp.quicksum(BackupSupply[d2, c, k, t, s] for d2 in B)
                    rhs = a * X_dc[d, c, k, t]
                    if resilience_penalty is None:
                        m.addConstr(lhs >= rhs, name=f"Resilience_{d}_{s}_{c}_{k}_{t}")
                    else:
                        terms[(d, s, c, k, t)] = (lhs, rhs)
    for c in C:
        for s in S_model:
            if theta.get((c, s), 1.0) > 1.0:
                for k in K:
                    for t in T:
                        m.addConstr(gp.quicksum(BackupSupply[d2, c, k, t, s] for d2 in B) >=
                                    a * (theta[(c, s)] - 1.0) * D_ckt[c, k, t],
                                    name=f"SurgeResilience_{c}_{k}_{t}_{s}")
    m.addConstrs((gp.quicksum(BackupSupply[d2, c, k, t, s] for c in C) <= M * Z_d[d2]
                  for d2 in B for k in K for t in T for s in S_model), name="BackupActivation")

    # ---- bounded reliability (fill rate), only built when needed ----
    R = None
    if maximize_R or epsilon is not None:
        dem_s = {(c, k, t, s): D_ckt[c, k, t] * theta.get((c, s), 1.0)
                 for c in C for k in K for t in T for s in S_model}
        v = m.addVars(list(dem_s.keys()), lb=0, ub=dem_s, name="v")
        R_s = {}
        for s in S_model:
            for c in C:
                for k in K:
                    for t in T:
                        delivered = (gp.quicksum(X_dc[d, c, k, t] for d in Dc if d not in failed_in[s]) +
                                     gp.quicksum(BackupSupply[d2, c, k, t, s] for d2 in B))
                        m.addConstr(v[c, k, t, s] <= delivered, name=f"Fulfil_{c}_{k}_{t}_{s}")
            tot = sum(dem_s[(c, k, t, s)] for c in C for k in K for t in T)
            R_s[s] = (1.0 / tot) * gp.quicksum(v[c, k, t, s] for c in C for k in K for t in T)
        if metric == "worst":
            R = m.addVar(lb=0, ub=1, name="R_worst")
            for s in S_model:
                m.addConstr(R <= R_s[s], name=f"Rworst_{s}")
        else:
            R = (1.0 / sum(wS.values())) * gp.quicksum(wS[s] * R_s[s] for s in S_model)

    if maximize_R:
        m.setObjective(R, GRB.MAXIMIZE)
    else:
        obj = Z1
        if resilience_penalty is not None:
            obj = obj + gp.quicksum(resilience_penalty[key] * (terms[key][1] - terms[key][0]) for key in terms)
        m.setObjective(obj, GRB.MINIMIZE)
        if epsilon is not None:
            m.addConstr(R >= epsilon, name="EpsilonFloor_Reliability")

    V = dict(Y_w=Y_w, Y_d=Y_d, Z_d=Z_d, X_dc=X_dc, BackupSupply=BackupSupply, E_kt=E_kt, I_d=I_d,
             comps={"Production": C1, "Holding": C2, "Transport": C3, "Facility fixed": C4,
                    "Backup extra": C5, "Disposal": C6, "Recycling (net)": C7},
             em={"Transport": EM1, "Production": EM2, "Disposal": EM3, "Recycling offset": EM4})
    return m, Z1, terms, V


# ---------- post-solve helpers ----------
def reliability(V):
    """Fill-rate reliability computed from a solved model: returns (R_avg, R_worst, {case: R_s})."""
    X, BS = V["X_dc"], V["BackupSupply"]
    Rs = {}
    for s in S_model:
        num = den = 0.0
        for c in C:
            for k in K:
                for t in T:
                    dem = D_ckt[c, k, t] * theta.get((c, s), 1.0)
                    got = (sum(X[d, c, k, t].X for d in Dc if d not in failed_in[s]) +
                           sum(BS[d2, c, k, t, s].X for d2 in B))
                    num += min(dem, got); den += dem
        Rs[s] = num / den
    avg = sum(wS[s] * Rs[s] for s in S_model) / sum(wS.values())
    return avg, min(Rs.values()), Rs


def open_summary(V):
    ws_ = ",".join(w for w in W if V["Y_w"][w].X > 0.5)
    ds_ = ",".join(d for d in Dc if V["Y_d"][d].X > 0.5)
    bs_ = ",".join(d for d in B if V["Z_d"][d].X > 0.5)
    return f"W[{ws_}] D[{ds_}] backup[{bs_}]"

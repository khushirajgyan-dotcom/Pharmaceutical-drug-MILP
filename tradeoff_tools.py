"""Pure-Python helpers for the cost-vs-reliability intersection point and plots (no Gurobi needed)."""

def normalise(points):
    """points: list of dicts with 'R' (reliability) and 'cost'. Returns (pts_sorted, Rmin, Rmax, Cmin, Cmax)."""
    pts = sorted(points, key=lambda p: p["R"])
    Rmin, Rmax = pts[0]["R"], pts[-1]["R"]
    Cmin, Cmax = min(p["cost"] for p in pts), max(p["cost"] for p in pts)
    for p in pts:
        p["mu_rel"] = (p["R"] - Rmin) / (Rmax - Rmin) if Rmax > Rmin else 1.0     # share of attainable reliability gain achieved
        p["mu_cost"] = (Cmax - p["cost"]) / (Cmax - Cmin) if Cmax > Cmin else 1.0  # share of cost headroom still unused
    return pts, Rmin, Rmax, Cmin, Cmax

def find_bracket(points):
    """Interval (a, b) of consecutive points where mu_rel - mu_cost changes sign (the intersection lies inside)."""
    pts = normalise(points)[0]
    for a, b in zip(pts, pts[1:]):
        if (a["mu_rel"] - a["mu_cost"]) < 0 <= (b["mu_rel"] - b["mu_cost"]):
            return a, b
    return None

def interpolate_crossing(points):
    """Linear interpolation of the intersection of the two normalised curves (use after refining the bracket)."""
    br = find_bracket(points)
    if br is None: return None
    a, b = br
    da, db = a["mu_rel"] - a["mu_cost"], b["mu_rel"] - b["mu_cost"]
    x = -da / (db - da)
    return {"R": a["R"] + x * (b["R"] - a["R"]), "cost": a["cost"] + x * (b["cost"] - a["cost"]),
            "mu": a["mu_rel"] + x * (b["mu_rel"] - a["mu_rel"])}

def plot_tradeoff(points, fname, crossing=None, band=None, title_suffix=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts, Rmin, Rmax, Cmin, Cmax = normalise(points)
    R = [p["R"] for p in pts]; C = [p["cost"] / 1e6 for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A: Pareto frontier
    ax[0].plot(R, C, "o-", color="#1f4e79", lw=2, label="Pareto frontier (epsilon-constraint)")
    ax[0].scatter([R[0]], [C[0]], s=130, color="green", zorder=5, label=f"Min-cost solution (R={R[0]:.4f})")
    ax[0].scatter([R[-1]], [C[-1]], s=130, color="crimson", zorder=5, label=f"Max-reliability solution (R={R[-1]:.4f})")
    if crossing:
        ax[0].scatter([crossing["R"]], [crossing["cost"] / 1e6], s=220, marker="*", color="gold", edgecolor="k", zorder=6,
                      label=f"Compromise (intersection): R={crossing['R']:.4f}, cost={crossing['cost']/1e6:.3f}M")
        ax[0].axvline(crossing["R"], ls=":", color="gray"); ax[0].axhline(crossing["cost"] / 1e6, ls=":", color="gray")
    if band:
        ax[0].axvspan(band[0], band[1], color="gold", alpha=0.25, label="Intersection lies in this interval (refine)")
    ax[0].set_xlabel("Network reliability R (average fill rate under disruption)")
    ax[0].set_ylabel("Total cost incl. environmental penalty (million)")
    ax[0].set_title("A. Cost vs reliability trade-off" + title_suffix); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, loc="upper left")
    # Panel B: normalised objectives and their intersection
    ax[1].plot(R, [p["mu_cost"] for p in pts], "s-", color="#c0504d", lw=2, label="Cost objective: unused cost headroom (1 = cheapest)")
    ax[1].plot(R, [p["mu_rel"] for p in pts], "o-", color="#1f4e79", lw=2, label="Reliability objective: share of attainable gain (1 = max R)")
    if crossing:
        ax[1].scatter([crossing["R"]], [crossing["mu"]], s=220, marker="*", color="gold", edgecolor="k", zorder=6,
                      label=f"Intersection: both objectives {crossing['mu']*100:.0f}% satisfied")
        ax[1].axvline(crossing["R"], ls=":", color="gray")
    if band:
        ax[1].axvspan(band[0], band[1], color="gold", alpha=0.25)
    ax[1].set_xlabel("Network reliability R"); ax[1].set_ylabel("Normalised satisfaction (0-1)")
    ax[1].set_title("B. Intersection of the two normalised objectives"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, loc="lower left")
    ax[1].set_ylim(-0.02, 1.05)
    plt.tight_layout(); plt.savefig(fname, dpi=160); plt.close()

def plot_alpha(rows, fname):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = [r[0] for r in rows]; cost = [r[1] / 1e6 for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(a, cost, "o-", color="#c0504d", lw=2, label="Total cost (left axis)")
    ax.set_xlabel("Minimum backup coverage alpha"); ax.set_ylabel("Total cost (million)", color="#c0504d"); ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(a, [r[2] for r in rows], "s--", color="#1f4e79", label="R average"); ax2.plot(a, [r[3] for r in rows], "^--", color="green", label="R worst case")
    ax2.set_ylabel("Reliability (fill rate)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title("Cost and reliability vs backup coverage level (alpha)")
    plt.tight_layout(); plt.savefig(fname, dpi=160); plt.close()

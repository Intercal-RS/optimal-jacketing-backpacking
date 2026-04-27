"""
Average-cost MDP for optimal jacket + backpack usage in RS3 elite treasure trails.

Problem
-------
Elite clues are sequences of steps, each independently a compass dig (P = 482/1059)
or a scan (P = 577/1059).  The player has two item types, each with 0-4 charges
and progress accumulating 1/clue toward the next charge:

  - Jacket: skip to the correct dig spot in 3 ticks; completes the step.
  - Backpack: replace the current step with a fresh draw in 2 ticks; the
    replacement step still has to be solved (and may itself use items).

The goal is to minimise long-run average ticks per clue by deciding at each
step whether to jacket, backpack, or proceed.  Each step's outcome, combined
with the once-per-clue charge-earn rule, drives the MDP transitions.

Code layout
-----------
data.py              - parse compass.json / scan.json; shared constants (JACKET_TICKS, BP_TICKS)
frequencies.py       - realistic compass/scan frequency weights (wilderness, hot spots, clue 36)
scan.py              - scan-tree recursion for (g, q_j, q_b) and per-step scan aggregation
mdp.py               - 289-state MDP: within-clue DP, Bellman solve, policy iteration,
                       steady-state analysis, low-charge-penalty search
scan_analysis.py     - post-hoc analysis of which scan nodes trigger item use per threshold level
plot_compass_map.py  - visualise compass spots on the world map, coloured by threshold band
main.py              - entry point: runs the three policy variants and prints their tables

Policy variants (run by main())
-------------------------------
1. Optimal (unconstrained): pure minimisation of average ticks per clue. In
   practice, this has convergence issues due to nearly-equivalent marginal costs
   between jackets and backpacks in a lot of the state space, leading to unstable
   action order.
2. Load-balanced: item choice forced by progress ordering
   (jacket when 4*c_j + p_j >= 4*c_b + p_b, else backpack).  Small lambda cost,
   symmetric item usage, no anomalous near-tie states.
3. Load-balanced + low-charge-constrained: adds a Lagrangian penalty so that
   P(c_j + c_b <= 1) < 5% in steady state.

By default all three use realistic compass/scan frequency weights and avoid
difficult powerbursts (far NW compass spot, Mos Le'Harmless scan).  Both
behaviours are controlled by CLI flags; see main() for details.

See mdp.py's module docstring for the mathematical formulation (state space,
Bellman equation, per-step decision rule A_j / A_b, per-clue random effects,
policy iteration, steady-state flow balance).
"""

import argparse
import enum

from data import load_compass, load_scans, JACKET_TICKS, BP_TICKS
from frequencies import realistic_compass_frequencies, realistic_scan_frequencies
from mdp import (
    STATES, policy_iteration, find_low_charge_penalty,
    steady_state, expected_items_per_clue, p_low_charge,
)

class FreqDist(enum.Enum):
    REALISTIC = "realistic"
    UNIFORM = "uniform"


# Scan clue IDs whose ending positions are inside dungeons, leaving the player
# underground with a worse compass triangulation position than surface scans.
_DUNGEON_SCAN_IDS = frozenset([356, 357, 365, 366])
# 356=Brimhaven Dungeon, 357=Taverley Dungeon, 365=Dorgesh-Kaan, 366=Fremennik Slayer Dungeons


def _aggregate_policy(V, trans_all, h_map, load_balance):
    """
    Collapse the 289 raw states into (c_j, c_b) aggregates for display.

    These (c_j, c_b) are intentionally a compression of the model state to be
    more easily human-interpretable, but it also carries some misinterpretation
    risk that the full model wouldn't have, and extrapolating the actions in
    these states more generally could lead to a fairly different steady-state
    distribution.

    Returns (pi, agg, diag_split) where:
      pi         - steady-state dict over 289 states.
      agg        - {(c_j, c_b): {pi, wAj, wAj_denom, wAb, wAb_denom, jacket_pi}}
                   with steady-state-weighted threshold sums per cell.
      diag_split - for load_balance, splits each (c, c) diagonal cell into
                   {'j': <p_j >= p_b sub-row>, 'b': <p_j < p_b sub-row>}.

    Inputs:
      V            — final value function {state: V(state)} from policy_iteration.
      trans_all    — {state: {next_state: probability}} transition dict for the
                     policy; used to compute the steady-state distribution pi.
      h_map        — {(c_j, c_b): H} expected-step-cost lookup; folded into the
                     backpack threshold A_b = BP_TICKS + delta_b + H.
      load_balance — if True, also fill jacket_pi and diag_split; if False both
                     are left at their zero initialisation.
    """
    pi = steady_state(trans_all)

    A_j_map, A_b_map = {}, {}
    for s in STATES:
        c_j, p_j, c_b, p_b = s
        if c_j > 0:
            A_j_map[s] = JACKET_TICKS + V[(c_j - 1, p_j, c_b, p_b)] - V[s]
        if c_b > 0:
            A_b_map[s] = (BP_TICKS + V[(c_j, p_j, c_b - 1, p_b)] - V[s]
                          + h_map.get((c_j, c_b - 1), 0.0))

    agg = {(cj, cb): dict(pi=0.0, wAj=0.0, wAj_denom=0.0,
                          wAb=0.0, wAb_denom=0.0, jacket_pi=0.0)
           for cj in range(5) for cb in range(5)}
    diag_split = {c: {'j': dict(pi=0.0, wA=0.0, wA_denom=0.0),
                      'b': dict(pi=0.0, wA=0.0, wA_denom=0.0)} for c in range(5)}

    for s, prob in pi.items():
        prob = max(prob, 0.0)
        c_j, p_j, c_b, p_b = s
        cell = agg[(c_j, c_b)]
        cell['pi'] += prob
        if s in A_j_map:
            cell['wAj'] += prob * A_j_map[s]
            cell['wAj_denom'] += prob
        if s in A_b_map:
            cell['wAb'] += prob * A_b_map[s]
            cell['wAb_denom'] += prob
        if load_balance and c_j > 0 and (c_b == 0 or 4 * c_j + p_j >= 4 * c_b + p_b):
            cell['jacket_pi'] += prob
        if load_balance and c_j == c_b:
            sub = diag_split[c_j]['j'] if p_j >= p_b else diag_split[c_j]['b']
            sub['pi'] += prob
            if p_j >= p_b and s in A_j_map:
                sub['wA'] += prob * A_j_map[s]
                sub['wA_denom'] += prob
            elif p_j < p_b and s in A_b_map:
                sub['wA'] += prob * A_b_map[s]
                sub['wA_denom'] += prob

    # Fallback for cells with zero steady-state probability: use the (p_j=0,
    # p_b=0) slice as a representative threshold so the column is not empty.
    for cj in range(5):
        for cb in range(5):
            s0 = (cj, 0, cb, 0)
            cell = agg[(cj, cb)]
            if cell['wAj_denom'] == 0.0 and s0 in A_j_map:
                cell['wAj'], cell['wAj_denom'] = A_j_map[s0], 1.0
            if cell['wAb_denom'] == 0.0 and s0 in A_b_map:
                cell['wAb'], cell['wAb_denom'] = A_b_map[s0], 1.0

    return pi, agg, diag_split


def _infer_action(c_j, c_b, A_j, A_b, load_balance):
    """
    Determine the policy action label for a (c_j, c_b) aggregate cell.

    Inputs:
      c_j, c_b     — charge counts for this cell.
      A_j, A_b     — aggregated threshold values for jacket / backpack; only
                     compared when both items are available and load_balance=False.
      load_balance — if True the action is forced by the charge ordering rule.
                     Diagonal cells (c_j == c_b > 0) are split into sub-rows by
                     print_policy_table before reaching this function, so here
                     c_j != c_b always holds when both are > 0.  For off-diagonal
                     cells a charge difference of >=1 dominates any progress
                     difference, so the full rule 4*c_j+p_j >= 4*c_b+p_b
                     simplifies to c_j > c_b.  If False the action is whichever
                     threshold is lower.
    """
    if c_j == 0 and c_b == 0:
        return "proceed"
    if c_j == 0:
        return "backpack"
    if c_b == 0:
        return "jacket"
    if load_balance:
        return "jacket" if c_j > c_b else "backpack"
    return "jacket" if A_j <= A_b else "backpack"


def print_policy_table(V, lam, trans_all, h_map, label="", load_balance=False):
    """
    Print the steady-state-weighted threshold table for a solved policy.

    Each row is one (c_j, c_b) charge cell with its steady-state probability
    and aggregated thresholds A_j, A_b; the action column shows the policy
    choice in that cell.

    load_balance=True splits each diagonal cell into two sub-rows by the
    progress comparison (p_j >= p_b goes to jacket, p_j < p_b to backpack).
    Only the active threshold is printed on each sub-row.  Low-probability
    cells (P < 0.01%) are suppressed under load_balance.

    Inputs:
      V            — final value function {state: V(state)} from policy_iteration.
      lam          — long-run average ticks per clue under this policy.
      trans_all    — {state: {next_state: probability}} for the policy.
      h_map        — {(c_j, c_b): H} expected-step-cost lookup.
      label        — optional section header printed above the table.
      load_balance — if True, use the load-balanced display (diagonal split,
                     low-probability cell suppression).

    Returns the raw steady-state distribution pi.
    """
    pi, agg, diag_split = _aggregate_policy(V, trans_all, h_map, load_balance)

    if label:
        print(f"\n=== {label} ===")
    print(f"Lambda: {lam:.4f} ticks/clue")
    E_j, E_b = expected_items_per_clue(pi)
    print(f"Expected jackets per clue:   {E_j:.4f}")
    print(f"Expected backpacks per clue: {E_b:.4f}")
    print()

    def _fmt(v, action, expected):
        if action != expected or v == float('inf'):
            return f"{'—':>10}"
        return f"{v:>10.2f}"

    def _row(c_j, c_b, prob, A_j, A_b, action):
        print(f"{c_j:>4}  {c_b:>4}  {prob:>10.4f}  "
              f"{_fmt(A_j, action, 'jacket')}  {_fmt(A_b, action, 'backpack')}  "
              f"{action:>8}")

    hdr = f"{'c_j':>4}  {'c_b':>4}  {'P(state)':>10}  {'A_j':>10}  {'A_b':>10}  {'action':>8}"
    print(hdr)
    print("-" * len(hdr))

    for c_j in range(5):
        for c_b in range(5):
            cell = agg[(c_j, c_b)]
            prob = cell['pi']
            if load_balance and prob < 1e-4:
                continue

            if load_balance and c_j == c_b and c_j > 0:
                # Diagonal under load_balance: split into jacket/backpack sub-rows.
                for sub_key, action in [('j', 'jacket'), ('b', 'backpack')]:
                    sub = diag_split[c_j][sub_key]
                    if sub['pi'] < 1e-4:
                        continue
                    thresh = (sub['wA'] / sub['wA_denom']
                              if sub['wA_denom'] > 0 else float('inf'))
                    A_j = thresh if action == 'jacket' else float('inf')
                    A_b = thresh if action == 'backpack' else float('inf')
                    _row(c_j, c_b, sub['pi'], A_j, A_b, action)
            else:
                A_j = (cell['wAj'] / cell['wAj_denom']
                       if cell['wAj_denom'] > 0 else float('inf'))
                A_b = (cell['wAb'] / cell['wAb_denom']
                       if cell['wAb_denom'] > 0 else float('inf'))
                action = _infer_action(c_j, c_b, A_j, A_b, load_balance)
                _row(c_j, c_b, prob, A_j, A_b, action)

    return pi


def print_v_debug(V, h_map):
    """
    Print raw V values and per-progress-state thresholds for the c_j=4 slice.

    Useful for diagnosing non-monotonicity and near-tie anomalies at the cap.
    The c_j=4 boundary is the most irregular (cap dynamics break the uniform
    charge progression), so inspecting it directly is the quickest check.

    Inputs:
      V     — final value function {state: V(state)} from policy_iteration.
      h_map — {(c_j, c_b): H} expected-step-cost lookup; values printed in the
              H(4, c_b-1) column and used to compute A_b.
    """
    print("\n--- Raw V and per-progress-state thresholds for c_j=4 ---")
    print(f"{'c_b':>4}  {'p_b':>4}  {'V(4,0,cb,pb)':>14}  {'V(3,0,cb,pb)':>14}  "
          f"{'delta_j':>8}  {'A_j':>8}", end="")
    print(f"  {'V(4,0,cb-1,pb)':>15}  {'delta_b':>8}  {'H(4,cb-1)':>10}  {'A_b':>8}")
    for c_b in range(5):
        p_b_vals = [0] if c_b == 4 else range(4)
        for p_b in p_b_vals:
            v_cur = V[(4, 0, c_b, p_b)]
            v_jm1 = V[(3, 0, c_b, p_b)]
            delta_j = v_jm1 - v_cur
            A_j = JACKET_TICKS + delta_j
            if c_b > 0:
                v_bm1 = V[(4, 0, c_b - 1, p_b)]
                delta_b = v_bm1 - v_cur
                H = h_map.get((4, c_b - 1), 0.0)
                A_b = BP_TICKS + delta_b + H
                print(f"{c_b:>4}  {p_b:>4}  {v_cur:>14.4f}  {v_jm1:>14.4f}  "
                      f"{delta_j:>8.4f}  {A_j:>8.4f}  {v_bm1:>15.4f}  "
                      f"{delta_b:>8.4f}  {H:>10.4f}  {A_b:>8.4f}")
            else:
                print(f"{c_b:>4}  {p_b:>4}  {v_cur:>14.4f}  {v_jm1:>14.4f}  "
                      f"{delta_j:>8.4f}  {A_j:>8.4f}")


def _load_inputs(freq_dist=FreqDist.REALISTIC, avoid_difficult_powerbursts=True):
    """
    Load compass and scan data.

    When freq_dist is REALISTIC, a two-pass load is used: a first pass under
    uniform weights enumerates entries and feeds the frequency-weight builders,
    then a second pass applies the resulting weighted distributions.  When
    UNIFORM, a single pass suffices and both distributions default to uniform.

    Returns (compass_dist, scan_trees) ready to pass into policy_iteration.
    Pass underground_ids=_DUNGEON_SCAN_IDS to policy_iteration so the MDP
    applies the correct look-ahead triangulation overheads per scan type.

    Inputs:
      freq_dist                  — FreqDist.REALISTIC or FreqDist.UNIFORM.
      avoid_difficult_powerbursts — forwarded to load_compass / load_scans.
    """
    if freq_dist is FreqDist.REALISTIC:
        scan_trees_raw = load_scans("scan.json",
                                    avoid_difficult_powerbursts=avoid_difficult_powerbursts)
        scan_freqs = realistic_scan_frequencies(scan_trees_raw)
        scan_trees = load_scans(
            "scan.json",
            scan_frequencies=scan_freqs,
            avoid_difficult_powerbursts=avoid_difficult_powerbursts,
        )
        compass_spot_freqs = realistic_compass_frequencies("compass.json")
        compass_dist = load_compass(
            "compass.json",
            spot_frequencies=compass_spot_freqs,
            avoid_difficult_powerbursts=avoid_difficult_powerbursts,
        )
    else:
        scan_trees = load_scans("scan.json",
                                avoid_difficult_powerbursts=avoid_difficult_powerbursts)
        compass_dist = load_compass("compass.json",
                                    avoid_difficult_powerbursts=avoid_difficult_powerbursts)
    return compass_dist, scan_trees


def main():
    """
    Run all three policy variants and print their result tables.

    Flags:
      --frequencies {realistic,uniform}   frequency distribution (default: realistic)
      --avoid-difficult-powerbursts /
      --no-avoid-difficult-powerbursts    exclude hard powerburst methods (default: True)

    Reads compass.json and scan.json; prints to stdout; returns nothing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frequencies",
        type=lambda s: FreqDist(s),
        choices=list(FreqDist),
        default=FreqDist.REALISTIC,
        metavar=f"{{{','.join(m.value for m in FreqDist)}}}",
        help="frequency distribution to use (default: realistic)",
    )
    parser.add_argument(
        "--avoid-difficult-powerbursts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude difficult powerburst compass/scan methods (default: True)",
    )
    args = parser.parse_args()

    compass_dist, scan_trees = _load_inputs(
        freq_dist=args.frequencies,
        avoid_difficult_powerbursts=args.avoid_difficult_powerbursts,
    )

    print("=" * 60)
    print("Variant 1: \"Optimal\" (unconstrained) policy")
    print("=" * 60)
    V, lam, trans, h_map = policy_iteration(
        compass_dist, scan_trees, underground_ids=_DUNGEON_SCAN_IDS
    )
    print_v_debug(V, h_map)
    print_policy_table(V, lam, trans, h_map, label="Optimal (unconstrained)")

    print("\n" + "=" * 60)
    print("Variant 2: Load-balanced policy")
    print("=" * 60)
    V_lb, lam_lb, trans_lb, h_map_lb = policy_iteration(
        compass_dist, scan_trees, load_balance=True, underground_ids=_DUNGEON_SCAN_IDS
    )
    print_policy_table(
        V_lb, lam_lb, trans_lb, h_map_lb,
        label="Load-balanced (jacket when 4*c_j+p_j >= 4*c_b+p_b)",
        load_balance=True,
    )

    gap = lam_lb - lam
    pct = 100.0 * gap / lam
    print(f"\nLoad-balanced suboptimality vs optimal:")
    print(f"  gap: {gap:+.4f} ticks/clue ({pct:+.3f}%)")

    print("\n" + "=" * 60)
    print("Variant 3: Load-balanced + low-charge-constrained policy")
    print("=" * 60)
    V_c, lam_c, trans_c, h_map_c, penalty = find_low_charge_penalty(
        compass_dist, scan_trees, target=0.05, tol=0.5, load_balance=True,
        underground_ids=_DUNGEON_SCAN_IDS,
    )
    pi_c = print_policy_table(
        V_c, lam_c, trans_c, h_map_c,
        label=f"Load-balanced, low-charge-constrained (penalty={penalty:.3f})",
        load_balance=True,
    )
    print(f"P(c_j + c_b <= 1) = {p_low_charge(pi_c):.4f}")


if __name__ == "__main__":
    main()

"""
Average-cost MDP over 289 states (c_j, p_j, c_b, p_b).


Mathematics
-----------

State space.  Between clues the player is in s = (c_j, p_j, c_b, p_b), where
c_j, c_b in {0..4} are the remaining charges and p_j, p_b in {0..3} are the
progress values (1/clue) toward the next charge.  The cap is absorbing in the
sense that progress stays at 0 while c = 4, so each item has 17 valid (c, p)
pairs and the joint state space has 17*17 = 289 states.

Objective.  Minimise the long-run average ticks per clue lambda.  The
average-cost Bellman equation for a relative value function V is

    V(s) + lambda = E_T(s)  +  sum_{s'} P(s -> s') V(s')      (*)

where E_T(s) is the expected ticks for one clue starting in s and P(s -> s')
is the state transition induced by the policy.  V is underdetermined by one
additive constant; we pin V(0,0,0,0) = 0.  Solving (*) is a linear system
in (V, lambda); see _solve_bellman.

Within-clue decision rule.  The MDP is solved relative to V, but each
individual step decision is local.  Let delta_j = V(c_j-1, p_j, c_b, p_b) -
V(c_j, p_j, c_b, p_b) be the opportunity cost of burning one jacket charge
(infinity when c_j = 0); define delta_b analogously.  At a step with
observed travel time t the effective action costs are

    A_j = JACKET_TICKS + delta_j
    A_b = BP_TICKS + delta_b + H(c_j, c_b - 1)

where H(c_j, c_b) is the expected cost of the *replacement* step that a
backpack triggers, averaged over valid progress values.  The optimal action
is: use whichever item has the smaller A if min(A_j, A_b) < t, else proceed.

For compass steps, some ticks of triangulation overhead are spent before the
player learns which spot they drew.  This overhead is attributed as a look-ahead
cost on the preceding step (see _build_step_cache and compute_scan_step): each
step adds P_COMPASS * overhead to its expected time, where overhead depends on
whether the preceding step ended underground.  Because the overhead is added as
a flat constant after the per-spot loop, it does not enter the threshold
comparison.  Back-to-back steps — where the previous step was a compass to the
same spot — bypass triangulation entirely.

See compute_g_q (scan.py) and _build_step_cache for the per-step logic; H is
computed by _compute_h_map via nested fixed-point over (c_j, c_b).

Per-clue dynamics.  A clue has n_steps steps drawn uniformly from {4, 5, 6}.
At each non-final step two independent random effects fire:
  - P_INST = 1%: instant completion, clue ends immediately.
  - P_DBL  = 1% * (1 - P_INST): next step is skipped.
If both fire simultaneously, instant-completion wins.  _run_clue_dp tracks the
joint distribution over (k_j, k_b) = items used so far and folds both random
effects in.  Each clue end increments p by 1 mod 4; wrapping 3 -> 0 grants a
charge unless the cap (c = 4) is already reached.

Policy iteration.  Alternate (i) solving (*) for V, lambda given the current
transition probabilities and (ii) recomputing the transition probabilities
from V (via the decision rule above).  Convergence is declared by a plateau
of best-seen lambda rather than a strict fixed point, because thin near-tie
regions can oscillate indefinitely between nearly equivalent choices.

Scan subtrees.  Compass steps are scalar times t; scan steps are trees of
partial-information pulses with branch-conditional continuations.  scan.py's
compute_g_q walks the tree bottom-up returning (g, q_j, q_b) = (expected
time, P(jacket used), P(backpack used)) under the same decision rule,
handling the pre-root subtlety that root.tick == 0 lets each first-pulse
branch decide independently after observing its pulse.

Variants (optional policy constraints).
  - load_balance=True: force item choice by progress ordering (jacket when
    4*c_j + p_j >= 4*c_b + p_b).  Produces symmetric usage at a small lambda
    cost by suppressing near-tie anomalies where the optimal policy picks
    whichever item's V is infinitesimally lower.
  - low_charge_penalty > 0: add a Lagrangian penalty to E_T(s) for states
    with c_j + c_b <= 1 inside the Bellman solve.  Penalty-biased V pushes
    the policy away from those states; after convergence we re-solve with
    penalty = 0 on the (E_T, trans) from the best iteration to report the
    true lambda of the constrained policy.  find_low_charge_penalty
    binary-searches the smallest penalty meeting a target low-charge
    probability.

Steady state.  Once the policy is fixed the chain has a stationary
distribution pi (left eigenvector of P, eigenvalue 1).  steady_state returns
it as a dict; expected_items_per_clue derives E[items/clue] by flow balance
(E[jackets] = P(p_j = 3 and c_j < 4) since exactly that many charges are
earned per clue in steady state, and charges earned equal charges consumed).
"""

import numpy as np
from data import JACKET_TICKS, BP_TICKS, P_COMPASS, P_SCAN
from scan import compute_scan_step

# Uniform prior over elite clue step counts; E[n_steps] = 5.
STEP_COUNTS = (4, 5, 6)

# Per-step random effects.  Both are independent 1% rolls; if they fire on the
# same step the instant-completion effect takes precedence.  Neither has any
# effect on the final step of the clue.
_P_INST = 0.01            # instant completion: clue ends after this step
_P_DBL  = 0.01 * 0.99    # double increment (without instant): next step skipped
_P_NORM = 0.99 * 0.99    # neither effect: step increments normally by 1

# Each charge system independently has 17 valid (c, p) pairs: c in {0..4},
# p in {0..3}, with (c=4, p>0) excluded (progress is always 0 at cap).
_CHARGE_STATES = [(c, p) for c in range(5) for p in range(4) if not (c == 4 and p > 0)]
_CHARGE_STATES_SET = set(_CHARGE_STATES)

# Full MDP state: (c_j, p_j, c_b, p_b).  17 x 17 = 289 states.
STATES = [
    (cj, pj, cb, pb)
    for (cj, pj) in _CHARGE_STATES
    for (cb, pb) in _CHARGE_STATES
]
STATE_INDEX = {s: i for i, s in enumerate(STATES)}

# Best low-charge penalty found historically; used as the initial upper bracket
# in find_low_charge_penalty to skip the exponential-growth phase on re-runs.
_LOW_CHARGE_PENALTY_HINT = 2.523


def next_charge_state(c_end, p):
    """
    End-of-clue transition for a single charge system.

    c_end: charges remaining after the clue (start charges minus items used).
    p:     progress value carried through the clue (unchanged by item use).

    Each clue completion increments p by 1 (mod 4); when p wraps to 0, one
    charge is earned (provided c_end < 4).  At the cap (c_end=4), progress
    stays at 0.  Returns (c_new, p_new).
    """
    if c_end < 4:
        p_new = (p + 1) % 4
        c_new = c_end + 1 if p_new == 0 else c_end
    else:
        p_new, c_new = 0, 4
    return (c_new, p_new)


def _build_step_cache(c_j, p_j, c_b, p_b, V, compass_dist, scan_trees,
                      h_approx=None, load_balance=False, back_to_back=False,
                      underground_ids=frozenset()):
    """
    Precompute per-step (E_time, p_jacket, p_backpack) for every feasible
    (cj_cur, cb_cur) the clue can reach from its starting state.

    Inputs:
      c_j, c_b       — starting charges for jacket/backpack at clue start;
                       the cache covers all cj in [0..c_j], cb in [0..c_b].
      p_j, p_b       — clue-start progress values; held fixed throughout the
                       clue since progress only advances at clue completion.
      V              — current value function {state: V(state)}; used to form
                       the opportunity costs delta_j, delta_b.
      compass_dist   — list of (travel_time, freq) pairs from load_compass;
                       freq values sum to 1.  Travel times are raw (no
                       triangulation overhead); the look-ahead overhead
                       P_COMPASS * 1 is added here to the compass contribution.
      underground_ids — forwarded to compute_scan_step; set of clue_ids whose
                        ending position is underground (6-tick triangulation
                        overhead for a subsequent compass vs 3 for overground).
      scan_trees     — list of ScanTree (from load_scans); per-tree freq sums to 1.
      h_approx       — controls H in the backpack cost A_b = BP_TICKS + delta_b + H(cj, cb-1):
                         None  — self-referential: reuse cache[(cj, cb-1)][3]
                                 from this build (requires bottom-up fill).
                         float — fixed scalar applied uniformly at every level.
                         dict  — {(cj, cb): float} looked up per level; 0.0 if missing.
      load_balance   — if True, force item choice by 4*cj + p_j vs 4*cb + p_b
                       (jacket wins on >=); the unused item's effective cost is
                       set to infinity so the existing min(A_j, A_b) logic picks
                       the designated item.
      back_to_back   — if True, apply the back-to-back compass discount: for each
                       spot with frequency f, probability P_COMPASS * f reflects
                       the chance the previous step was also compass to this spot,
                       leaving the player already there (JACKET_TICKS cost, no item
                       needed).  The remaining probability (1 - P_COMPASS * f) uses
                       the normal travel time and threshold comparison.  Should be
                       False for the first step of a clue and for replacement steps
                       (H computation), where no back-to-back context exists.

    Returns dict {(cj, cb): (E_time, p_jacket, p_backpack, E_time_copy)}; the
    duplicated E_time in slot 3 is read back as H by later entries when
    h_approx=None.
    """
    cache = {}
    for cj in range(c_j + 1):
        for cb in range(c_b + 1):
            delta_j = (
                V[(cj - 1, p_j, cb, p_b)] - V[(cj, p_j, cb, p_b)]
                if cj > 0 else float("inf")
            )
            if cb > 0:
                delta_b = V[(cj, p_j, cb - 1, p_b)] - V[(cj, p_j, cb, p_b)]
                if h_approx is None:
                    H_prev = cache[(cj, cb - 1)][3]
                elif isinstance(h_approx, dict):
                    H_prev = h_approx.get((cj, cb - 1), 0.0)
                else:
                    H_prev = float(h_approx)
            else:
                delta_b = float("inf")
                H_prev = 0.0

            A_j = JACKET_TICKS + delta_j
            A_b = BP_TICKS + delta_b + H_prev if cb > 0 else float("inf")

            if load_balance:
                if 4 * cj + p_j >= 4 * cb + p_b:
                    A_b = float("inf")
                    delta_j_eff, delta_b_eff = delta_j, float("inf")
                else:
                    A_j = float("inf")
                    delta_j_eff, delta_b_eff = float("inf"), delta_b
            else:
                delta_j_eff, delta_b_eff = delta_j, delta_b

            A_star = min(A_j, A_b)

            # Compass contribution: one comparison per (t, freq) pair.
            # Look-ahead: this compass step attributes P_COMPASS * 1 ticks of
            # triangulation overhead for the potential next compass step (1 tick
            # because the player ends above ground at a dig spot).  Added as a
            # flat constant after the loop since it is independent of which spot
            # is drawn and which action is taken.
            e_t_c, p_j_c, p_b_c = 0.0, 0.0, 0.0
            for t, freq in compass_dist:
                if back_to_back:
                    # P(previous step was also compass to this same spot).
                    # Player is already there: JACKET_TICKS cost, no item use.
                    p_btb = P_COMPASS * freq
                    e_t_c += freq * p_btb * JACKET_TICKS
                    eff_freq = freq * (1.0 - p_btb)
                else:
                    eff_freq = freq
                if t > A_star:
                    if A_j <= A_b:
                        e_t_c += eff_freq * JACKET_TICKS
                        p_j_c += eff_freq
                    else:
                        e_t_c += eff_freq * (BP_TICKS + H_prev)
                        p_b_c += eff_freq
                else:
                    e_t_c += eff_freq * t
            e_t_c += P_COMPASS * 1  # look-ahead: 1-tick overhead if next step is compass

            # Scan contribution: delegated to the scan-tree recursion.
            e_t_s, p_j_s, p_b_s = compute_scan_step(
                scan_trees, cj, cb, delta_j_eff, delta_b_eff, H_prev,
                underground_ids=underground_ids,
            )

            e_t = P_COMPASS * e_t_c + P_SCAN * e_t_s
            p_j_step = P_COMPASS * p_j_c + P_SCAN * p_j_s
            p_b_step = P_COMPASS * p_b_c + P_SCAN * p_b_s

            cache[(cj, cb)] = (e_t, p_j_step, p_b_step, e_t)
    return cache


def _compute_h_map(V, compass_dist, scan_trees, h_map_prev, load_balance=False,
                   underground_ids=frozenset()):
    """
    Build the (c_j, c_b) -> expected-step-cost map H used in backpack
    replacement costs, averaged uniformly over all valid progress states
    (p_j, p_b).

    Inputs:
      V               — converged value function {state: V(state)}.
      compass_dist    — compass step distribution from load_compass.
      scan_trees      — scan trees from load_scans.
      h_map_prev      — accepted for API symmetry but unused; the function builds
                        a fresh map bottom-up and reuses already-computed entries
                        within this call via h_approx=h_map.
      load_balance    — forwarded to _build_step_cache.
      underground_ids — forwarded to _build_step_cache.

    Returns {(c_j, c_b): H_value} for every charge pair the MDP cares about.
    """
    h_map = {}
    for cb in range(5):
        for cj in range(5):
            p_j_vals = [0] if cj == 4 else range(4)
            p_b_vals = [0] if cb == 4 else range(4)
            costs = []
            for p_j in p_j_vals:
                if (cj, p_j) not in _CHARGE_STATES_SET:
                    continue
                for p_b in p_b_vals:
                    if (cb, p_b) not in _CHARGE_STATES_SET:
                        continue
                    cache = _build_step_cache(
                        cj, p_j, cb, p_b, V, compass_dist, scan_trees,
                        h_approx=h_map, load_balance=load_balance,
                        underground_ids=underground_ids,
                    )
                    costs.append(cache[(cj, cb)][0])
            if costs:
                h_map[(cj, cb)] = sum(costs) / len(costs)
    return h_map


def _run_clue_dp(step_cache_first, step_cache_later, c_j, p_j, c_b, p_b, n_steps):
    """
    Run the within-clue DP for a fixed step count.

    Tracks the joint distribution over (k_j, k_b) = (jackets used, backpacks
    used) so far.  Handles two per-step random effects: instant completion
    (probability _P_INST) routes mass to final_dp immediately; double increment
    (probability _P_DBL) routes mass to step n+2 via the pending buffer, or to
    final_dp if the skip lands past the last step.  Neither effect applies on
    the last step.

    Inputs:
      step_cache_first  — cache (from _build_step_cache back_to_back=False) used
                          for the first step (n=0) of the clue, where no back-to-
                          back context exists.
      step_cache_later  — cache (from _build_step_cache back_to_back=True) used
                          for all subsequent steps, where a same-spot back-to-back
                          discount may apply.
      c_j, c_b       — clue-start charges; mid-clue cj_cur = c_j - k_j.
      p_j, p_b       — clue-start progress values (unused here except by the
                       caller's next-state projection; accepted for signature
                       symmetry with _clue_expectation).
      n_steps        — number of steps in this clue (one of STEP_COUNTS).

    Returns (E_T, final_dp):
      E_T       — expected ticks summed over the clue.
      final_dp  — {(k_j, k_b): probability} at clue end, summing to 1 over all
                  termination paths (normal last step, instant, double-skip-past-end).
    """
    dp = {(0, 0): 1.0}
    final_dp = {}
    pending = [dict() for _ in range(n_steps)]
    E_T = 0.0

    for n in range(n_steps):
        # Absorb any double-skip mass that lands on this step.
        for key, p in pending[n].items():
            dp[key] = dp.get(key, 0.0) + p

        is_last = (n == n_steps - 1)
        new_dp = {}
        step_cache = step_cache_first if n == 0 else step_cache_later

        for (k_j, k_b), prob in dp.items():
            cj_cur = c_j - k_j
            cb_cur = c_b - k_b
            e_t, p_j_step, p_b_step, _ = step_cache[(cj_cur, cb_cur)]
            E_T += prob * e_t

            item_outcomes = [(k_j, k_b, 1.0 - p_j_step - p_b_step)]
            if cj_cur > 0:
                item_outcomes.append((k_j + 1, k_b, p_j_step))
            if cb_cur > 0:
                item_outcomes.append((k_j, k_b + 1, p_b_step))

            for nkj, nkb, ip in item_outcomes:
                p = prob * ip
                nkey = (nkj, nkb)
                if is_last:
                    final_dp[nkey] = final_dp.get(nkey, 0.0) + p
                else:
                    new_dp[nkey] = new_dp.get(nkey, 0.0) + p * _P_NORM
                    final_dp[nkey] = final_dp.get(nkey, 0.0) + p * _P_INST
                    if n + 2 < n_steps:
                        pending[n + 2][nkey] = pending[n + 2].get(nkey, 0.0) + p * _P_DBL
                    else:
                        final_dp[nkey] = final_dp.get(nkey, 0.0) + p * _P_DBL

        dp = new_dp

    return E_T, final_dp


def _clue_expectation(c_j, p_j, c_b, p_b, V, compass_dist, scan_trees,
                     h_approx=None, load_balance=False, underground_ids=frozenset()):
    """
    Expected cost and next-state distribution for one clue starting in
    (c_j, p_j, c_b, p_b) under the policy implied by V.

    Builds two step caches (first step and later steps) to handle the
    back-to-back compass discount, then averages (_run_clue_dp, next-state
    projection) uniformly over STEP_COUNTS.

    Inputs:
      c_j, p_j, c_b, p_b — clue-start state.
      V                  — current value function {state: V(state)}.
      compass_dist       — compass step distribution from load_compass.
      scan_trees         — scan trees from load_scans.
      h_approx           — forwarded to _build_step_cache (None / float / dict).
      load_balance       — forwarded to _build_step_cache.
      underground_ids    — forwarded to _build_step_cache.

    Returns (E_T, trans):
      E_T   — expected ticks for the clue, averaged over STEP_COUNTS.
      trans — {(c_j', p_j', c_b', p_b'): probability} for the next-clue state.
    """
    step_cache_first = _build_step_cache(
        c_j, p_j, c_b, p_b, V, compass_dist, scan_trees, h_approx, load_balance,
        back_to_back=False, underground_ids=underground_ids,
    )
    step_cache_later = _build_step_cache(
        c_j, p_j, c_b, p_b, V, compass_dist, scan_trees, h_approx, load_balance,
        back_to_back=True, underground_ids=underground_ids,
    )

    E_T_sum = 0.0
    trans_sum = {}

    for n_steps in STEP_COUNTS:
        E_T, final_dp = _run_clue_dp(
            step_cache_first, step_cache_later, c_j, p_j, c_b, p_b, n_steps
        )
        E_T_sum += E_T
        for (k_j, k_b), prob in final_dp.items():
            sj = next_charge_state(c_j - k_j, p_j)
            sb = next_charge_state(c_b - k_b, p_b)
            s_next = (sj[0], sj[1], sb[0], sb[1])
            trans_sum[s_next] = trans_sum.get(s_next, 0.0) + prob

    w = 1.0 / len(STEP_COUNTS)
    trans = {s: p * w for s, p in trans_sum.items()}
    return E_T_sum * w, trans


def _solve_bellman(E_T_all, trans_all, low_charge_penalty=0.0):
    """
    Solve the average-cost Bellman system for V and lambda.

    System:  V(s) - sum_{s'} P(s, s') V(s') + lambda = E_T(s)  for each s.

    The system is underdetermined by one degree of freedom; pin scale by
    replacing the row for (0,0,0,0) with V(0,0,0,0) = 0.  Uses lstsq rather
    than solve because the load-balance rule can leave the chain with several
    communicating components, each needing its own pin; lstsq returns the
    minimum-norm solution satisfying the explicit pin.

    Inputs:
      E_T_all            — {state: expected ticks per clue} from the current policy
                           (typically built by _clue_expectation over every state).
      trans_all          — {state: {next_state: probability}} transition dict from
                           the current policy.
      low_charge_penalty — extra ticks added to E_T(s) for states with
                           c_j + c_b <= 1.  > 0 biases the policy toward keeping
                           those states rare.  The returned lambda includes the
                           penalty; callers wanting the true lambda should
                           re-solve with penalty=0 using the same trans_all.

    Returns (V, lambda):
      V       — {state: V(state)} relative value function (pinned V(0,0,0,0)=0).
      lambda  — long-run average ticks per clue under the current policy
                (plus any penalty mass above).
    """
    n = len(STATES)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    A[0, 0] = 1.0   # pin V(0,0,0,0) = 0

    for i, s in enumerate(STATES):
        row = i + 1
        A[row, i] = 1.0
        A[row, n] = 1.0
        for s_next, prob in trans_all[s].items():
            A[row, STATE_INDEX[s_next]] -= prob
        penalty = low_charge_penalty if s[0] + s[2] <= 1 else 0.0
        b[row] = E_T_all[s] + penalty

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    V = {STATES[i]: x[i] for i in range(n)}
    return V, float(x[n])


def policy_iteration(compass_dist, scan_trees, max_iter=500, warm_start_iter=30,
                     plateau=100, load_balance=False, low_charge_penalty=0.0,
                     underground_ids=frozenset()):
    """
    Run policy iteration over the 289-state MDP.

    Inputs:
      compass_dist       — compass step distribution from load_compass.
      scan_trees         — scan trees from load_scans.
      max_iter           — hard upper bound on main-loop iterations after warm
                           start; the plateau check usually triggers earlier.
      warm_start_iter    — number of backpack-suppressed iterations used to
                           seed V (ignored when load_balance=True).
      plateau            — stop main loop after this many consecutive
                           iterations with no improvement in best-seen lambda.
      load_balance       — if False, unconstrained optimisation (warm start
                           used; near-tie oscillation handled by plateau rule).
                           If True, item choice is forced by progress ordering
                           (jacket when 4*c_j+p_j >= 4*c_b+p_b), no warm start
                           needed, converges before the plateau trigger.
      low_charge_penalty — Lagrangian penalty applied inside _solve_bellman to
                           states with c_j + c_b <= 1.  After convergence the
                           function re-solves without the penalty using the
                           best-iteration (E_T, trans) to report the true lambda.
      underground_ids    — forwarded to _clue_expectation and _compute_h_map;
                           scan clue ids whose ending position is underground.

    Returns (V, lambda, trans_all, h_map):
      V         — final value function {state: V(state)}.
      lambda    — true (penalty-free) average ticks per clue.
      trans_all — {state: {next_state: probability}} for the final policy.
      h_map     — {(c_j, c_b): H} expected step costs, for display / next runs.
    """
    V = {s: 0.0 for s in STATES}

    if not load_balance:
        # Warm start: suppress backpacking so V converges to jacket-only thresholds.
        for _ in range(warm_start_iter):
            E_T_ws = {}
            trans_ws = {}
            for s in STATES:
                c_j, p_j, c_b, p_b = s
                E_T_ws[s], trans_ws[s] = _clue_expectation(
                    c_j, p_j, c_b, p_b, V, compass_dist, scan_trees,
                    h_approx=1e9, underground_ids=underground_ids,
                )
            V, _ = _solve_bellman(E_T_ws, trans_ws, low_charge_penalty)

    best_lam = float("inf")
    best_E_T = None
    best_trans = None
    last_improved = 0

    for iteration in range(max_iter):
        E_T_all = {}
        trans_all = {}
        for s in STATES:
            c_j, p_j, c_b, p_b = s
            E_T_all[s], trans_all[s] = _clue_expectation(
                c_j, p_j, c_b, p_b, V, compass_dist, scan_trees,
                load_balance=load_balance, underground_ids=underground_ids,
            )

        V, lam = _solve_bellman(E_T_all, trans_all, low_charge_penalty)

        if lam < best_lam - 1e-6:
            best_lam = lam
            best_E_T = E_T_all
            best_trans = trans_all
            last_improved = iteration

        if iteration - last_improved >= plateau:
            print(f"No improvement for {plateau} iterations. Best lambda={best_lam:.4f}")
            break
    else:
        print(f"Returning best lambda={best_lam:.4f} found across {max_iter} iterations.")

    # Re-solve without penalty to recover the true lambda of the constrained policy.
    # best_E_T contains unpenalised step costs; best_trans encodes the policy
    # chosen under the penalty.  Solving without penalty gives the real average cost.
    V_best, lam_best = _solve_bellman(best_E_T, best_trans)
    h_map = _compute_h_map(V_best, compass_dist, scan_trees, {},
                           load_balance=load_balance, underground_ids=underground_ids)
    return V_best, lam_best, best_trans, h_map


def steady_state(trans_all):
    """
    Compute the steady-state distribution pi over the 289 MDP states.

    Inputs:
      trans_all — {state: {next_state: probability}} transition dict for the
                  policy whose stationary distribution is wanted.

    Builds the transition matrix P and returns the left eigenvector of P
    corresponding to eigenvalue 1 (pi @ P = pi), normalised to sum to 1 with
    all entries clipped to be non-negative.

    Returns a dict {state: probability}.
    """
    n = len(STATES)
    P = np.zeros((n, n))
    for i, s in enumerate(STATES):
        for s_next, prob in trans_all[s].items():
            P[i, STATE_INDEX[s_next]] += prob

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = eigenvectors[:, idx].real
    if pi.mean() < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return {STATES[i]: float(pi[i]) for i in range(n)}


def expected_items_per_clue(pi):
    """
    Expected number of jackets and backpacks used per clue, from steady state.

    Inputs:
      pi — {state: probability} dict as returned by steady_state.

    Derived by flow balance: in steady state the number of charges consumed per
    clue equals the number of charges earned per clue.  A jacket charge is
    earned exactly when p_j wraps from 3 to 0 with c_j < 4, which happens each
    clue deterministically regardless of item use.  So:

        E[jackets/clue]  = P(p_j = 3 and c_j < 4)
        E[backpacks/clue] = P(p_b = 3 and c_b < 4)

    This formulation is exact and does not depend on which V is used.  See
    jacket_backpack_convergence.md for why a simulation-based derivation is
    inconsistent when the policy oscillates.

    Returns (E_j, E_b) — expected jackets and backpacks used per clue.
    """
    E_j = sum(prob for s, prob in pi.items() if s[1] == 3 and s[0] < 4)
    E_b = sum(prob for s, prob in pi.items() if s[3] == 3 and s[2] < 4)
    return E_j, E_b


def p_low_charge(pi):
    """
    Steady-state probability of being low on total charges, i.e. c_j + c_b <= 1.

    Inputs:
      pi — {state: probability} dict as returned by steady_state.

    Corresponds to states (0,0), (1,0)xprogress, (0,1)xprogress - where the
    player has at most one charge between the two items.  The
    low_charge_penalty in policy_iteration is tuned to keep this below a target.
    """
    return sum(prob for s, prob in pi.items() if s[0] + s[2] <= 1)


def find_low_charge_penalty(compass_dist, scan_trees, target=0.05, tol=1e-3,
                            load_balance=True,
                            penalty_hint=_LOW_CHARGE_PENALTY_HINT,
                            **pi_kwargs):
    """
    Binary-search the smallest low_charge_penalty that drives P(c_j+c_b<=1)
    below `target` in steady state.

    Starts by checking whether the constraint is already satisfied at penalty=0.
    If not, tries penalty_hint as the initial upper bracket to skip the
    exponential-growth phase on subsequent runs.  Falls back to exponential
    growth (*4 per step) if the hint is insufficient, then bisects within
    [lo, hi] until hi - lo <= tol.

    Inputs:
      compass_dist   — compass step distribution from load_compass.
      scan_trees     — scan trees from load_scans.
      target         — upper bound on the steady-state P(c_j + c_b <= 1); the
                       returned policy satisfies P(low) < target.
      tol            — stopping tolerance on the penalty bracket width.
      load_balance   — forwarded to policy_iteration; typically True (a
                       constraint only makes sense alongside load-balancing).
      penalty_hint   — initial upper bracket guess; defaults to the best
                       previously observed value so re-runs skip the exponential
                       growth phase.
      **pi_kwargs    — extra kwargs forwarded to policy_iteration (e.g. plateau,
                       max_iter, warm_start_iter).

    Returns (V, lambda, trans_all, h_map, penalty):
      V, lambda, trans_all, h_map — as returned by policy_iteration for the
                                    smallest-penalty feasible policy found.
      penalty                     — the penalty value used.
    """
    def run(penalty):
        print(f"  [search] penalty={penalty:.3f} ...", flush=True)
        V, lam, trans, h = policy_iteration(
            compass_dist, scan_trees,
            load_balance=load_balance,
            low_charge_penalty=penalty,
            **pi_kwargs,
        )
        pi = steady_state(trans)
        p = p_low_charge(pi)
        print(f"           lam={lam:.4f}  P(low)={p:.4f}", flush=True)
        return V, lam, trans, h, pi, p

    V0, lam0, trans0, h0, pi0, p0 = run(0.0)
    if p0 < target:
        print(f"Constraint already satisfied at penalty=0 (P(low)={p0:.4f})")
        return V0, lam0, trans0, h0, 0.0

    lo, hi = 0.0, penalty_hint
    V_hi, lam_hi, trans_hi, h_hi, _, p_hi = run(hi)
    V_best, lam_best, trans_best, h_best = V_hi, lam_hi, trans_hi, h_hi

    if not p_hi < target:
        # Hint insufficient; grow exponentially until the upper bracket works.
        while True:
            lo = hi
            hi *= 4.0
            V_hi, lam_hi, trans_hi, h_hi, _, p_hi = run(hi)
            if p_hi < target:
                V_best, lam_best, trans_best, h_best = V_hi, lam_hi, trans_hi, h_hi
                break

    # Bisect within [lo, hi]; keep the best (smallest-penalty) feasible policy.
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        V_mid, lam_mid, trans_mid, h_mid, _, p_mid = run(mid)
        if p_mid < target:
            hi = mid
            V_best, lam_best, trans_best, h_best = V_mid, lam_mid, trans_mid, h_mid
        else:
            lo = mid

    print(f"\nFound penalty={hi:.4f}  lam={lam_best:.4f}")
    return V_best, lam_best, trans_best, h_best, hi

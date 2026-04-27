"""
Scan-tree recursion and per-step scan aggregation.

compute_g_q walks a single scan tree bottom-up and returns, for each node,
the expected time-to-completion and the probability that a jacket/backpack is
used somewhere below it (given the current effective action costs).

compute_scan_step averages over all scan trees and handles the pre-root
decision, including the special case where root.tick == 0 (first pulse is
free, each branch gets its own independent decision).
"""

from data import ScanNode, ScanTree, JACKET_TICKS, BP_TICKS, P_COMPASS


def compute_g_q(node, delta_j, A_b, H_val, spot_freqs):
    """
    Returns (g, q_j, q_b) for a scan node given effective action costs.

    g    — expected ticks to complete the scan from this node.
    q_j  — probability that a jacket is used from this node onward (in this scan).
    q_b  — probability that a backpack is used from this node onward (in this scan).

    delta_j: opportunity cost of one jacket charge (V(c_j-1,...) - V(c_j,...)); inf if c_j=0.
    A_b:     fixed effective cost of backpacking (BP_TICKS + delta_b + H_val); inf if c_b=0.
    H_val:   expected cost of the replacement step when backpacking (H(c_j, c_b-1)).

    The jacket cost at a node is node-local: A_j = JACKET_TICKS + (1-continue_q_j)*delta_j.
    The (1-continue_q_j) factor accounts for the residual opportunity cost — if continuing
    would also use a jacket with probability continue_q_j, only the remaining fraction of
    delta_j separates jacketing now from continuing.

    The backpack cost A_b is fixed (not node-local): backpacking always exits the scan
    entirely and costs BP_TICKS + delta_b + H_val regardless of where in the tree we are.

    Inputs:
      node        — ScanNode to recurse from (subtree root); leaves return
                    (0, 0, 0) directly.
      delta_j     — opportunity cost of one jacket charge at the current MDP
                    state (V(c_j-1,...) - V(c_j,...)).  float('inf') if c_j=0.
      A_b         — fixed effective backpack cost BP_TICKS + delta_b + H_val for
                    this call; float('inf') if c_b=0.  Not node-local because
                    backpacking exits the scan entirely.
      H_val       — H(c_j, c_b-1), the expected replacement-step cost paid when
                    backpacking; already folded into A_b, kept as a separate
                    argument so returned g values can add it to BP_TICKS.
      spot_freqs  — {(x, y): probability} for every dig spot in this scan,
                    summing to 1 over the root's remaining candidates.
    """
    if node.is_leaf:
        return 0.0, 0.0, 0.0

    total_weight = sum(spot_freqs.get(s, 0.0) for s in node.rc)

    continue_time = 0.0
    continue_q_j = 0.0
    continue_q_b = 0.0
    for _pulse, child in node.children:
        w = sum(spot_freqs.get(s, 0.0) for s in child.rc) / total_weight
        travel = child.tick - node.tick
        g_c, q_j_c, q_b_c = compute_g_q(child, delta_j, A_b, H_val, spot_freqs)
        continue_time += w * (travel + g_c)
        continue_q_j += w * q_j_c
        continue_q_b += w * q_b_c

    A_j_node = JACKET_TICKS + (1.0 - continue_q_j) * delta_j
    best_A = min(A_j_node, A_b)

    if best_A <= continue_time:
        if A_j_node <= A_b:
            return JACKET_TICKS, 1.0, 0.0
        else:
            return BP_TICKS + H_val, 0.0, 1.0
    return continue_time, continue_q_j, continue_q_b


def compute_scan_step(scan_trees, c_j, c_b, delta_j, delta_b, H_val,
                      underground_ids=frozenset()):
    """
    Returns (E_time, p_jacket, p_backpack) for one scan step, averaged over all
    scan trees weighted by their frequencies.

    When neither item is available, returns (f0_preroot, 0, 0).

    Each tree's expected time includes a look-ahead overhead of P_COMPASS *
    overhead ticks, attributing the triangulation cost of a potential following
    compass step to this scan step:
      - underground_ids scans: overhead = 6 (player ends underground)
      - all other scans:       overhead = 3 (player ends above ground)

    Pre-root decision:

      Normal case (root.tick > 0):
        The player travels to the root before the first pulse; the pre-root
        decision is made before any scan information is observed.

      Special case (root.tick == 0):
        The first pulse is free; each branch gets an independent pre-root decision
        made after observing the pulse result.
        This corresponds to specific TF/DL scan results that can be run from almost
        anywhere (which we approximate as actually anywhere).

    Inputs:
      scan_trees      — list of ScanTree; each tree's freq is used to weight the
                        per-scan (E_time, p_j, p_b) into the average.  freqs sum to 1.
      c_j, c_b        — current charges.  Only used to short-circuit when both are 0.
      delta_j         — opportunity cost of one jacket charge; float('inf') if c_j=0.
      delta_b         — opportunity cost of one backpack charge; float('inf') if c_b=0.
      H_val           — expected cost of one replacement step at charges (c_j, c_b-1);
                        ignored (may be 0) when c_b == 0 since A_b will be inf anyway.
      underground_ids — set of clue_ids whose ending position is underground, giving
                        6-tick triangulation overhead for a subsequent compass step
                        (vs 3 ticks for overground scans).
    """
    if c_j == 0 and c_b == 0:
        e_time = sum(
            t.freq * (t.f0_preroot
                      + P_COMPASS * (6 if t.clue_id in underground_ids else 3))
            for t in scan_trees
        )
        return e_time, 0.0, 0.0

    A_b = BP_TICKS + delta_b + H_val   # fixed across all nodes for this call

    e_time = 0.0
    p_jacket = 0.0
    p_backpack = 0.0

    for tree in scan_trees:
        g_root, q_j_root, q_b_root = compute_g_q(
            tree.root, delta_j, A_b, H_val, tree.spot_freqs
        )

        if tree.root.tick == 0:
            total_w = sum(tree.spot_freqs.get(s, 0.0) for s in tree.root.rc)
            g_pre, q_j_pre, q_b_pre = 0.0, 0.0, 0.0
            for _pulse, child in tree.root.children:
                w = sum(tree.spot_freqs.get(s, 0.0) for s in child.rc) / total_w
                travel_c = child.tick
                g_c, q_j_c, q_b_c = compute_g_q(
                    child, delta_j, A_b, H_val, tree.spot_freqs
                )
                A_j_branch = JACKET_TICKS + (1.0 - q_j_c) * delta_j
                best_A = min(A_j_branch, A_b)
                if best_A <= travel_c + g_c:
                    if A_j_branch <= A_b:
                        g_pre += w * JACKET_TICKS
                        q_j_pre += w * 1.0
                    else:
                        g_pre += w * (BP_TICKS + H_val)
                        q_b_pre += w * 1.0
                else:
                    g_pre += w * (travel_c + g_c)
                    q_j_pre += w * q_j_c
                    q_b_pre += w * q_b_c
        else:
            travel_to_root = tree.root.tick
            continue_time = travel_to_root + g_root
            A_j_pre = JACKET_TICKS + (1.0 - q_j_root) * delta_j
            best_A = min(A_j_pre, A_b)
            if best_A <= continue_time:
                if A_j_pre <= A_b:
                    g_pre, q_j_pre, q_b_pre = JACKET_TICKS, 1.0, 0.0
                else:
                    g_pre, q_j_pre, q_b_pre = BP_TICKS + H_val, 0.0, 1.0
            else:
                g_pre, q_j_pre, q_b_pre = continue_time, q_j_root, q_b_root

        g_pre += P_COMPASS * (6 if tree.clue_id in underground_ids else 3)
        e_time += tree.freq * g_pre
        p_jacket += tree.freq * q_j_pre
        p_backpack += tree.freq * q_b_pre

    return e_time, p_jacket, p_backpack

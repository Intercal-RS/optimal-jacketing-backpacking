"""
Scan jacketing analysis: for each scan tree and each diagonal charge-state
threshold level, identify which nodes first become worth using an item at.

Suppression: once a node fires (its continue_time exceeds the threshold),
its descendants are not reported — the item was already used at that node
and the player never reaches the descendants with an unused item.

A node is "new" at threshold level X if:
  - Its continue_time is in (tau_X, tau_prev] — it first fires at this level.
  - No ancestor fires at tau_X — it is not suppressed.
"""

from data import load_scans
from frequencies import realistic_scan_frequencies


def f0_node(node, spot_freqs):
    """
    Expected ticks to finish the scan from this node, assuming no item is
    used at this node or any descendant.

    Weighted bottom-up over the child branches by their conditional
    probabilities (computed from spot_freqs).

    Inputs:
      node       — ScanNode to evaluate.  Leaves return 0.0 directly.
      spot_freqs — {(x, y): probability} for every dig spot in this scan,
                   normalised so the root's rc sums to 1.  Used to reweight
                   child branch probabilities at each node.
    """
    if node.is_leaf:
        return 0.0
    total_w = sum(spot_freqs.get(s, 0.0) for s in node.rc)
    result = 0.0
    for _pulse, child in node.children:
        w = sum(spot_freqs.get(s, 0.0) for s in child.rc) / total_w
        travel = child.tick - node.tick
        result += w * (travel + f0_node(child, spot_freqs))
    return result


def _dfs(node, sf, tau, events, path):
    """
    DFS from node. If this node's continue_time exceeds tau, record it and stop
    (children suppressed). Otherwise recurse into children.

    Inputs:
      node   — ScanNode currently being inspected.  Leaves return immediately.
      sf     — spot frequencies {(x, y): probability}; passed through to f0_node.
      tau    — threshold; nodes whose f0 exceeds tau fire (item used here).
      events — output list mutated in place; each firing node appends
               ("internal", ct, path, tick).
      path   — human-readable breadcrumb from the root: concatenated "pN->"
               pulse labels.  Extended by recursive calls.
    """
    if node.is_leaf:
        return
    ct = f0_node(node, sf)
    if ct > tau:
        events.append(("internal", ct, path, node.tick))
        return
    for pulse, child in node.children:
        _dfs(child, sf, tau, events, path + f"p{pulse}->")


def find_active_events(tree, tau):
    """
    All jacketing events for this tree at threshold tau, with suppression applied.

    For root.tick > 0: pre-root decision fires if f0_preroot > tau; if so, no
    internal events (item already used). Otherwise DFS from root.

    For root.tick == 0: first pulse is free. Each root branch is decided
    independently after observing the result. If a branch fires, its subtree
    is suppressed. Otherwise DFS into the branch's children.

    Inputs:
      tree — ScanTree whose firing nodes should be listed.  Its spot_freqs,
             root, and f0_preroot are consulted.
      tau  — threshold (ticks).  A node fires when its continue_time > tau.

    Returns a list of (kind, ct, path, tick) tuples, where kind is one of
    "pre-root", "branch", or "internal".
    """
    events = []
    sf = tree.spot_freqs

    if tree.root.tick > 0:
        if tree.f0_preroot > tau:
            events.append(("pre-root", tree.f0_preroot, "pre-root", 0))
        else:
            _dfs(tree.root, sf, tau, events, "")
    else:
        for pulse, child in tree.root.children:
            branch_ct = child.tick + f0_node(child, sf)
            if branch_ct > tau:
                events.append(("branch", branch_ct, f"p{pulse}->", child.tick))
            else:
                _dfs(child, sf, tau, events, f"p{pulse}->")

    return events


def run_analysis():
    """
    Print one section per threshold level listing the scan nodes that first
    become worth using an item at.  Thresholds are the per-diagonal-cell
    effective min(A_j, A_b) values taken from results.md §4.

    Inputs: none.  Reads scan.json via load_scans (avoid_difficult_powerbursts
    is hard-coded True so it stays consistent with the policy tables) and uses
    realistic_scan_frequencies to reweight by actual play frequency.
    """
    scan_trees_raw = load_scans("scan.json", avoid_difficult_powerbursts=True)
    scan_freqs = realistic_scan_frequencies(scan_trees_raw)
    scan_trees = load_scans("scan.json", scan_frequencies=scan_freqs,
                            avoid_difficult_powerbursts=True)

    # THRESHOLDS should be provided by selecting states from a run of main.py.
    THRESHOLDS = [
        ("(1,1)", 37.50),
        ("(2,2)", 33.22),
        ("(3,3)", 31.88),
        ("(4,3)", 24.27),
    ]

    print("Scan jacketing/backpacking by threshold level")
    print("(continue_time > tau triggers item use; subtree suppressed once fired)\n")

    for idx, (label, tau) in enumerate(THRESHOLDS):
        tau_prev = THRESHOLDS[idx - 1][1] if idx > 0 else float("inf")

        new_events = []
        for tree in scan_trees:
            name = f"clue {tree.clue_id} ({tree.scantext[:28].strip()})"
            for kind, ct, path, tick in find_active_events(tree, tau):
                if ct <= tau_prev:  # first fires at this level, not a higher one
                    desc = (f"f0={ct:.2f}" if kind == "pre-root"
                            else f"{path} (tick={tick:.0f}, ct={ct:.2f})")
                    new_events.append((ct, name, kind, desc))

        new_events.sort(key=lambda e: -e[0])
        print(f"=== {label}  tau={tau:.2f} ===")
        if not new_events:
            print("  (none)")
        else:
            for ct, name, kind, desc in new_events:
                print(f"  [{kind:9s}]  {name}  --  {desc}")
        print()


if __name__ == "__main__":
    run_analysis()

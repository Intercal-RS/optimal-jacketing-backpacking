"""
Data loading and shared domain constants.

Parses compass.json and scan.json into structures the MDP can consume:
  - load_compass: one (time, freq) pair per unique dig spot.
  - load_scans:   one ScanTree per unique clue id, after deduplication.

Both loaders accept a frequency override (default: uniform) and an
avoid_difficult_powerbursts flag that skips methods requiring tight
powerburst timing most players opt out of.
"""

import json
from dataclasses import dataclass

# Methods excluded when avoid_difficult_powerbursts=True.
# These require precise powerburst timing that many players skip.
_DIFFICULT_POWERBURST_COMPASS = frozenset([
    "Phoenix Lair Teleport (Powerburst)",   # spot (2185, 3637)
])
_DIFFICULT_POWERBURST_SCAN = {
    358: "Big Book o'Piracy Start",         # Mos Le'Harmless
}

# Empirical per-step type probabilities for elite clues.
P_COMPASS = 482 / 1059
P_SCAN    = 577 / 1059

# Time cost (in ticks) of using a jacket teleport.  A jacket skips all
# remaining travel and delivers the player directly to the dig spot in 3 ticks.
JACKET_TICKS = 3
BP_TICKS = 2



@dataclass
class ScanNode:
    # Cumulative ticks elapsed from the pre-root position to reach this node.
    # tick=0 at the root means the player is already standing at the scan
    # position when the clue is opened (no travel required before first pulse).
    tick: float

    # Remaining candidates: dig spots still consistent with the pulses observed
    # on the path from the root to this node.  [(x, y), ...]
    rc: list

    # Branches from this node: one per distinct pulse value observed here.
    # Each entry is (pulse, child_node) where pulse in {1, 2, 3}.
    children: list  # [(pulse, ScanNode), ...]

    @property
    def is_leaf(self):
        # A leaf has no children: the correct dig spot has been uniquely
        # identified and the player should walk directly to it.
        return len(self.children) == 0


@dataclass
class ScanTree:
    # Numeric identifier matching details.clue.id in scan.json; used for deduplication.
    clue_id: int

    # Human-readable scan area description from details.clue.scantext.
    scantext: str

    # Root of the parsed decision tree (first node where a pulse is taken).
    root: ScanNode

    # Total ticks from the pre-root position to each dig spot, following the
    # optimal decision-tree path.  Keyed by (x, y).  Built from
    # timing_analysis.spots in the root node's JSON.
    # Note that timing_analysis is not included by default in the exported
    # JSON.
    root_timing: dict

    # Expected scan time with no jacket available: weighted average of
    # root_timing values over spot_freqs.  Precomputed at load time.
    f0_preroot: float

    # Normalised probability that this scan area is selected as a clue step,
    # after deduplication.  Set by load_scans after all entries are processed.
    freq: float

    # Normalised probability of each dig spot being the answer, conditioned on
    # this scan area being selected.  Keyed by (x, y); sums to 1 over root.rc.
    spot_freqs: dict


def _parse_node(raw):
    """
    Recursively convert a JSON node dict into a ScanNode tree.

    Inputs:
      raw — dict from scan.json describing one node: expects keys
            "path.post_state.tick" (cumulative ticks), "remaining_candidates"
            (list of {"x","y"} dicts), and optional "children" (each with
            "key.pulse" and "value" subtree).
    """
    tick = raw["path"]["post_state"]["tick"]
    rc = [(c["x"], c["y"]) for c in raw["remaining_candidates"]]
    children = [
        (child["key"]["pulse"], _parse_node(child["value"]))
        for child in raw.get("children", [])
    ]
    return ScanNode(tick=tick, rc=rc, children=children)


def _build_root_timing(root_raw):
    """
    Build {(x, y): total_ticks_from_preroot} from the root node's timing_analysis.

    The total time to each dig spot, as pre-computed by the scan tree analyzer
    and stored on the root node.  Used to derive f0(N) for any node N by
    subtracting tick(N) from each entry.

    Inputs:
      root_raw — dict for the root node from scan.json; expects
                 timing_analysis.spots, a list of {"spot": {"x","y"},
                 "timings":[{"ticks": float}, ...]} entries.
    """
    timing = {}
    for entry in root_raw["timing_analysis"]["spots"]:
        spot = entry["spot"]
        timing[(spot["x"], spot["y"])] = entry["timings"][0]["ticks"]
    return timing


def _normalize_spot_freqs(rc, raw_freqs):
    """
    Normalise raw spot weights over the candidate list rc so they sum to 1.

    Falls back to uniform if no weights are provided or all are zero.

    Inputs:
      rc        — list of (x, y) candidate dig spots for this scan.
      raw_freqs — dict {(x, y): weight}; missing keys get weight 0.0 and
                  unknown-to-rc keys are ignored.
    """
    total = sum(raw_freqs.get(s, 0.0) for s in rc)
    if total == 0.0:
        n = len(rc)
        return {s: 1.0 / n for s in rc}
    return {s: raw_freqs.get(s, 0.0) / total for s in rc}


def _f0_preroot(rc, root_timing, spot_freqs):
    """
    Expected scan time from the pre-root position with no item use.

    Weighted average of total travel times over the spot frequency distribution.
    This is the baseline scan cost used to decide whether to jacket/backpack
    before the first pulse.

    Inputs:
      rc          — list of (x, y) candidate dig spots for this scan.
      root_timing — {(x, y): total_ticks_from_preroot} built from root raw JSON.
      spot_freqs  — {(x, y): probability} for spots in rc, summing to 1.
    """
    return sum(spot_freqs[s] * root_timing[s] for s in rc)


def load_compass(path, spot_frequencies=None, avoid_difficult_powerbursts=False):
    """
    Load compass clue data and return a list of (travel_time, freq) pairs.

    Each pair represents one distinct dig spot: travel_time is the minimum
    expected travel time across all compass methods that lead to that spot,
    and freq is its normalised probability of appearing as the clue target.

    Triangulation overhead (the time spent before the player knows which spot
    they drew) is not included here; it is attributed as a look-ahead cost on
    the preceding step inside the MDP (see _build_step_cache and
    compute_scan_step).

    Inputs:
      path                        — path to compass.json.
      spot_frequencies            — dict {(x, y): weight} of relative spot weights.
                                    Default: uniform over all spots in the file.
                                    Normalised internally so only the ratios matter.
      avoid_difficult_powerbursts — if True, exclude methods listed in
                                    _DIFFICULT_POWERBURST_COMPASS before selecting
                                    the best time, so the next-fastest method is
                                    used instead.
    """
    with open(path) as f:
        data = json.load(f)

    # For each dig spot, keep only the fastest known compass approach.
    best = {}
    for entry in data:
        if avoid_difficult_powerbursts and entry["name"] in _DIFFICULT_POWERBURST_COMPASS:
            continue
        key = (entry["for"]["spot"]["x"], entry["for"]["spot"]["y"])
        t = entry["expected_time"]
        if key not in best or t < best[key]:
            best[key] = t

    if spot_frequencies is None:
        spot_frequencies = {k: 1.0 for k in best}

    total = sum(spot_frequencies.get(k, 0.0) for k in best)
    result = []
    for k, t in best.items():
        freq = spot_frequencies.get(k, 0.0) / total
        if freq > 0.0:
            result.append((t, freq))
    return result


def load_scans(path, scan_frequencies=None, scan_spot_frequencies=None,
               avoid_difficult_powerbursts=False):
    """
    Load scan clue data and return a list of ScanTree objects, one per
    unique clue id.

    When scan.json contains multiple entries for the same clue id (alternative
    scan methods), only the entry with the lowest f0_preroot is retained; the
    others are discarded entirely and excluded from frequency weighting.

    Inputs:
      path                        — path to scan.json.
      scan_frequencies            — dict {clue_id: weight} of relative frequency of
                                    each scan area appearing as a clue step.
                                    Default: uniform over all deduplicated entries.
      scan_spot_frequencies       — dict {clue_id: dict {(x, y): weight}} of relative
                                    frequency of each dig spot being correct within
                                    a given scan area.  Default: uniform within each.
      avoid_difficult_powerbursts — if True, exclude methods listed in
                                    _DIFFICULT_POWERBURST_SCAN before selecting the
                                    best entry, so the next-fastest method is used.
    """
    with open(path) as f:
        data = json.load(f)

    if scan_spot_frequencies is None:
        scan_spot_frequencies = {}

    # Deduplicate by clue_id, keeping the entry with the lowest f0_preroot.
    # f0_preroot depends on spot frequencies, so it is computed here using the
    # caller-supplied scan_spot_frequencies before any deduplication decision.
    best = {}  # clue_id -> ScanTree with lowest f0_preroot seen so far

    for entry in data:
        details = entry["details"]
        clue_id = details["clue"]["id"]
        if (avoid_difficult_powerbursts
                and entry.get("name") == _DIFFICULT_POWERBURST_SCAN.get(clue_id)):
            continue
        root_raw = details["root_node"]

        root = _parse_node(root_raw)
        root_timing = _build_root_timing(root_raw)
        spot_freqs = _normalize_spot_freqs(
            root.rc, scan_spot_frequencies.get(clue_id, {})
        )
        f0 = _f0_preroot(root.rc, root_timing, spot_freqs)

        scantext = details["clue"].get("scantext", "")

        if clue_id not in best or f0 < best[clue_id].f0_preroot:
            best[clue_id] = ScanTree(
                clue_id=clue_id,
                scantext=scantext,
                root=root,
                root_timing=root_timing,
                f0_preroot=f0,
                freq=0.0,       # populated below after deduplication
                spot_freqs=spot_freqs,
            )

    survivors = list(best.values())

    # Assign normalised scan frequencies after deduplication so that discarded
    # alternative methods do not contribute to the probability mass.
    if scan_frequencies is None:
        n = len(survivors)
        for tree in survivors:
            tree.freq = 1.0 / n
    else:
        total = sum(scan_frequencies.get(t.clue_id, 0.0) for t in survivors)
        for tree in survivors:
            tree.freq = scan_frequencies.get(tree.clue_id, 0.0) / total

    return survivors

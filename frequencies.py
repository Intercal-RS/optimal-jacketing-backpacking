"""
'Realistic' compass/scan frequency weights.

Default-uniform frequencies over-weight rarely-visited spots and under-weight
popular ones.  These builders return spot-frequency / scan-frequency dicts that
are an attempt at a more realistic distribution.

  - Wilderness compass spots are downsampled.
  - A small set of high-traffic compass spots are upsampled.
  - The deep wild scan is extremely rare (conditional probability 1/10580) so
    is weighted accordingly; all other scans stay uniform. Ideally wilderness
    volcano woudl be downweighted too, but we have no method available for it.

The dicts are intended to be passed into load_compass / load_scans.

"""

import json

# Bounding polygon for the "wilderness" region of the RS3 map.
# Compass dig spots inside this polygon are downsampled because elite clue
# players typically avoid or rarely visit wilderness areas.
# Vertices listed in order; the polygon is closed implicitly (last -> first).
_WILDERNESS_POLYGON = [
    (2944, 3532), (3257, 3532), (3257, 3578), (3337, 3578),
    (3337, 3570), (3381, 3570), (3381, 3895), (3431, 3965),
    (2944, 3965),
]

# Specific compass dig spots that appear more frequently than the uniform
# baseline suggests, e.g. due to popular farming routes or known hotspots.
# These are upsampled relative to their default weight of 1.0.
_UPSAMPLE_SPOTS = frozenset([
    (3432, 3754), (3437, 3682), (3433, 3639), (3289, 3528), (2901, 3356),
    (3383, 3893),
])

# Multiplicative adjustment factors applied to the raw weights.
_WILDERNESS_FACTOR = 10.0   # wilderness spots get weight 1/10 of baseline
_UPSAMPLE_FACTOR = 10.0     # high-frequency spots get weight 10x baseline


def _point_in_polygon(x, y, polygon):
    """
    Ray-casting point-in-polygon test.

    Casts a horizontal ray from (x, y) to the right and counts how many polygon
    edges it crosses.  An odd count means the point is inside.  Works correctly
    for simple (non-self-intersecting) polygons.

    Inputs:
      x, y    — query point in the same coordinate space as polygon vertices.
      polygon — list of (x, y) vertices in order.  Closed implicitly
                (last -> first); last == first is not required.
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Check whether the ray from (x, y) rightward crosses edge (j -> i).
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def realistic_compass_frequencies(path):
    """
    Return a spot_frequencies dict suitable for passing to load_compass.

    Assigns a raw weight to each compass dig spot in the file:
      - Wilderness spots (inside _WILDERNESS_POLYGON): weight 1/10
      - High-frequency spots (_UPSAMPLE_SPOTS): weight x10
      - Spots that are both wilderness and upsampled: both factors apply (x1)
      - All others: weight 1.0

    load_compass normalises these weights to a probability distribution.

    Inputs:
      path — path to compass.json.  Every entry's for.spot.{x,y} is collected
             into the deduplicated set of dig spots this function weights.
    """
    with open(path) as f:
        data = json.load(f)

    spots = set()
    for entry in data:
        key = (entry["for"]["spot"]["x"], entry["for"]["spot"]["y"])
        spots.add(key)

    freqs = {}
    for (x, y) in spots:
        w = 1.0
        if _point_in_polygon(x, y, _WILDERNESS_POLYGON):
            w /= _WILDERNESS_FACTOR
        if (x, y) in _UPSAMPLE_SPOTS:
            w *= _UPSAMPLE_FACTOR
        freqs[(x, y)] = w
    return freqs


def realistic_scan_frequencies(scan_trees):
    """
    Return a scan_frequencies dict suitable for passing to load_scans.

    Clue id 36 is extremely rare in practice: its conditional probability given
    a scan step is 1/10580.  All other scan clues are equally likely.

    The weights are set so that after load_scans normalises them:
        P(clue 36 | scan)    = 1 / 10580
        P(any other | scan)  = (10579 / 10580) / (N - 1)
    where N is the total number of deduplicated scan clues.

    Inputs:
      scan_trees — list of ScanTree objects (typically from load_scans called
                   without a frequency argument).  Only the clue_id of each
                   tree is used; freq values are ignored.
    """
    other_ids = [t.clue_id for t in scan_trees if t.clue_id != 36]
    # Assign weight 1 to clue 36.  For the remaining (N-1) clues to collectively
    # hold probability 10579/10580, each needs weight 10579/(N-1) so that the
    # total weight sums to 10580 and normalisation yields the target probabilities.
    w_other = (10580 - 1) / len(other_ids)
    freqs = {cid: w_other for cid in other_ids}
    freqs[36] = 1.0
    return freqs



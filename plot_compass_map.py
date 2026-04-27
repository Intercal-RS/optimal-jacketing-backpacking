"""
Plot filtered compass spots onto gielinor.png, color-coded by the diagonal
load-balanced jacket/backpack threshold at which each spot first gets jacketed.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from frequencies import realistic_compass_frequencies

# --- Map boundaries (RS coords) ---
RS_X_MIN, RS_X_MAX = 2048, 3712
RS_Y_MIN, RS_Y_MAX = 2752, 3968

# --- Load-balanced thresholds at relevant (j, b) states ---
# tau = min(A_j, A_b) at the diagonal charge state from the constrained
# load-balanced MDP (avoid_difficult_powerbursts=True, realistic frequencies).
# A compass spot is jacketed/backpacked when its step time exceeds tau.
THRESHOLDS = [
    (37.50, 1, 1),
    (33.22, 2, 2),
    (31.88, 3, 3),
    (24.27, 4, 3),
]

DIFFICULT_POWERBURST_NAMES = frozenset([
    "Phoenix Lair Teleport (Powerburst)",
])

BAND_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
GREY = "#aaaaaa"


def rs_to_pixel(x, y, img_w, img_h):
    """
    Convert RS3 in-game coordinates to pixel coordinates on gielinor.png.

    The map image spans the RS_X/Y_MIN..MAX rectangle; x increases east, y
    increases north in RS coords, so py is flipped.

    Inputs:
      x, y         — RS3 in-game coordinates of the dig spot.
      img_w, img_h — pixel dimensions of gielinor.png; determine the scale
                     factor between RS units and pixels.
    """
    px = (x - RS_X_MIN) / (RS_X_MAX - RS_X_MIN) * img_w
    py = (RS_Y_MAX - y) / (RS_Y_MAX - RS_Y_MIN) * img_h
    return px, py


def assign_band(t):
    """
    Return the index of the first THRESHOLDS entry whose tau is below t,
    or None if t is below every threshold (the spot is never item-worthy).

    A lower index means the spot is item-worthy at an earlier (lower-charge)
    state; index 0 means it fires already at (1, 1).

    Inputs:
      t — expected step time (ticks) for this spot, compared against each
          THRESHOLDS tau in order.
    """
    for i, (tau, j, b) in enumerate(THRESHOLDS):
        if t > tau:
            return i
    return None


def main():
    """
    Render gielinor.png with compass dig spots overlaid, coloured by the
    earliest (j, b) state at which each spot becomes item-worthy.

    Spots are sized by their realistic frequency; never-item-worthy spots
    appear as small grey dots.

    Reads compass.json (for spot locations and expected_time) and gielinor.png
    (for the background map).  Uses realistic_compass_frequencies for marker
    sizing.  Output path is given by --output (default: compass_map.png).
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="compass_map.png",
                        help="Output PNG file path (default: compass_map.png)")
    args = parser.parse_args()

    with open("compass.json", encoding="utf-8") as f:
        raw = json.load(f)

    best = {}
    for entry in raw:
        if entry.get("name") in DIFFICULT_POWERBURST_NAMES:
            continue
        spot = entry["for"]["spot"]
        key = (spot["x"], spot["y"])
        t = entry["expected_time"]
        if key not in best or t < best[key]:
            best[key] = t

    freqs_raw = realistic_compass_frequencies("compass.json")
    total_w = sum(freqs_raw.values())
    freqs = {k: v / total_w for k, v in freqs_raw.items()}

    bands = {i: [] for i in range(len(THRESHOLDS))}
    unjacketed = []
    for (x, y), t in best.items():
        band = assign_band(t)
        freq = freqs.get((x, y), 0.0)
        if band is not None:
            bands[band].append((x, y, t, freq))
        else:
            unjacketed.append((x, y, t, freq))

    for i, (tau, j, b) in enumerate(THRESHOLDS):
        print(f"  (j={j}, b={b}) tau={tau}: {len(bands[i])} spots")
    print(f"  Never jacketed (t <= {THRESHOLDS[-1][0]}): {len(unjacketed)} spots")

    img = Image.open("gielinor.png").convert("RGBA")
    img_w, img_h = img.size

    fig, ax = plt.subplots(figsize=(img_w / 150, img_h / 150), dpi=150)
    ax.imshow(img, extent=[0, img_w, img_h, 0], aspect="equal")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")

    # Unjacketed: small grey dots, no labels
    for x, y, t, freq in unjacketed:
        px, py = rs_to_pixel(x, y, img_w, img_h)
        ax.plot(px, py, "o", color=GREY, markersize=2.0, alpha=0.4, zorder=2)

    # Jacketed: plot lowest band first so higher bands render on top
    for i in reversed(range(len(THRESHOLDS))):
        color = BAND_COLORS[i]
        for x, y, t, freq in bands[i]:
            px, py = rs_to_pixel(x, y, img_w, img_h)
            msize = 4 + 20 * freq
            ax.plot(px, py, "o", color=color, markersize=msize,
                    markeredgecolor="black", markeredgewidth=0.4,
                    alpha=0.9, zorder=4)
            ax.text(px + 3, py - 2, f"{t:.0f}", fontsize=5.5,
                    color="white", fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])

    # --- Legend (upper left, matching compass_jacketed_map.png style) ---
    legend_patches = []
    for i, (tau, j, b) in enumerate(THRESHOLDS):
        prev_tau = THRESHOLDS[i - 1][0] if i > 0 else None
        if i == 0:
            label = f"j={j}, b={b} (tau={tau})"
        else:
            label = f"j={j}, b={b} (tau={tau})"
        legend_patches.append(mpatches.Patch(color=BAND_COLORS[i], label=label))

    legend = ax.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=6.5,
        framealpha=0.9,
        edgecolor="#888888",
        title="First jacketed/backpacked at (j, b)",
        title_fontsize=7,
    )

    plt.tight_layout(pad=0.1)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

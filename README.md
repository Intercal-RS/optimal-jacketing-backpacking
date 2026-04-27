# optimal-jacketing-backpacking
Tools to determine the optimal thresholds for jacketing/backpacking clues in RS3. Currently only elite clues are supported.

See main.py's file-level comment for the overall structure.
Timings per clue step are taken from modified JSON exports of standard Clue Trainer method packs.

Some analysis tools for interpreting the timing thresholds are provided as well:
1. plot_compass_map.py will display all compass locations on a map of Gielinor that are above various input thresholds.
2. scan_analysis.py will display all scan routes (or partial scan routes) that are above various input thresholds.

These analysis tools can take in arbitrary thresholds and be run independently of the model outputs from main.py.

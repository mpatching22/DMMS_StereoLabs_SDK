"""Read a manifest produced by multicam_gnss_prototype.py and show what happened.

    python inspect_manifest.py out/manifest.csv
    python inspect_manifest.py out/hz5/manifest.csv --rows 20

Separate from the capture script on purpose: capture should not be re-run just
to look at its output again, and a 133 MB SVO takes 40 seconds to decode.
"""
import argparse
import collections
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--rows", type=int, default=10, help="how many sample rows to print")
ap.add_argument("--speed-kmh", type=float, default=60.0,
                help="used to convert timing error into metres on the ground")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
if not rows:
    raise SystemExit("empty manifest")
speed = args.speed_kmh / 3.6

print(f"{args.manifest}   {len(rows)} capture sets\n")

# ---- how each set got its position
methods = collections.Counter(r["gnss_method"] for r in rows)
print("association method")
for m, n in methods.most_common():
    print(f"  {m:<22} {n:>6}  ({n/len(rows)*100:.1f}%)")

# ---- did the cameras agree
spreads = [float(r["max_camera_spread_ms"]) for r in rows if r["max_camera_spread_ms"]]
if spreads:
    print(f"\ncamera spread ms        min {min(spreads):.2f}  "
          f"mean {sum(spreads)/len(spreads):.2f}  max {max(spreads):.2f}")
    if max(spreads) == 0.0:
        print("  ^ all zero. Same recording supplied more than once, so this is an")
        print("    artefact of the test setup, NOT a synchronisation measurement.")

# ---- what interpolation bought
interp = {r["lat"] for r in rows if r["lat"]}
near = {r["lat_nearest_only"] for r in rows if r["lat_nearest_only"]}
gaps = [float(r["nearest_gap_ms"]) for r in rows if r["nearest_gap_ms"]]
print(f"\ndistinct positions      {len(interp)} interpolated  vs  {len(near)} nearest-only")
if gaps:
    print(f"nearest-only error      mean {sum(gaps)/len(gaps)/1000*speed:.2f} m, "
          f"worst {max(gaps)/1000*speed:.2f} m at {args.speed_kmh:g} km/h")

# ---- sample rows, so the stair-step is visible rather than asserted
print(f"\nfirst {args.rows} sets")
print(f"{'set':>5} {'gap ms':>8} {'method':>14} {'lat interpolated':>20} {'lat nearest':>20}")
for r in rows[:args.rows]:
    lat = f"{float(r['lat']):.9f}" if r["lat"] else "-"
    ln = f"{float(r['lat_nearest_only']):.9f}" if r["lat_nearest_only"] else "-"
    gap = f"{float(r['gnss_gap_ms']):.1f}" if r["gnss_gap_ms"] else "-"
    print(f"{r['set_id']:>5} {gap:>8} {r['gnss_method']:>14} {lat:>20} {ln:>20}")

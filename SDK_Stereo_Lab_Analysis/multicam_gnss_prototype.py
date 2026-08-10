"""
Multi-camera + GNSS capture prototype  -  EagleEye / Project P104
================================================================

Answers the four questions Marcus asked, using two SVO recordings standing in
for two cameras, so it runs with no hardware.

    1. How are multiple cameras initialised?      -> CameraWorker.open()
    2. How are several read simultaneously?       -> one thread per camera, bounded queues
    3. How is GNSS accessed and used?             -> GnssSource (synthetic here, real receiver later)
    4. How do they work together?                 -> FrameMatcher pairs by timestamp, then attaches GNSS

Output: manifest.csv and manifest.json, one row per matched frame set.

Run:
    python multicam_gnss_prototype.py --svo a.svo2 b.svo2 --out ./out

Notes
-----
* Frames are matched by TIMESTAMP TOLERANCE, never by equality. Even hardware
  synced cameras report timestamps a few milliseconds apart, because the stamp
  is applied when the frame lands in the deserializer buffer, not at exposure.
* GNSS runs at a different rate to the cameras, so association is nearest-in-time
  with a staleness limit, not a one-to-one join.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Windows: since Python 3.8 the loader ignores PATH when resolving DLLs for
# extension modules, so pyzed cannot find the SDK's DLLs even when the SDK's
# bin folder is on PATH. Register it explicitly before the import.
if os.name == "nt":
    for _cand in (os.environ.get("ZED_SDK_ROOT"),
                  r"C:\Program Files (x86)\ZED SDK",
                  r"C:\Program Files\ZED SDK"):
        if _cand and os.path.isdir(os.path.join(_cand, "bin")):
            os.add_dll_directory(os.path.join(_cand, "bin"))
            break

try:
    import pyzed.sl as sl
except ImportError:
    raise SystemExit(
        "pyzed not found.\n"
        "  1. Install the ZED SDK for your CUDA version from stereolabs.com/developers\n"
        "  2. Run the bundled get_python_api.py to install pyzed\n"
        "  3. Check with:  python -c \"import pyzed.sl as sl; print(sl.Camera().get_sdk_version())\""
    )

# --------------------------------------------------------------------------- config

MATCH_TOLERANCE_MS = 10.0    # frames within this window are one capture set
# NOTE: there is deliberately no fixed staleness constant here. An absolute
# threshold silently encodes an assumed GNSS rate and rejects every slower
# receiver as faulty. GnssSource measures the receiver's own cadence instead.
GNSS_RATE_HZ = 10.0          # what a real receiver gives us. See note at the bottom.
QUEUE_DEPTH = 8              # bounded, so a slow consumer cannot exhaust memory


@dataclass
class Frame:
    device: str
    svo_index: int
    timestamp_ns: int
    dropped_so_far: int


@dataclass
class GnssFix:
    timestamp_ns: int
    lat: float
    lon: float
    alt: float
    fix_type: str
    speed_mps: float


# --------------------------------------------------------------------- camera worker

class CameraWorker(threading.Thread):
    """One thread per camera. Grabs and enqueues only. No processing in here,
    because anything slow in this loop shows up as dropped frames."""

    def __init__(self, svo_path: Path, device_id: str, out_q: queue.Queue, stop: threading.Event):
        super().__init__(name=f"cam-{device_id}", daemon=True)
        self.svo_path = svo_path
        self.device_id = device_id
        self.out_q = out_q
        self.stop = stop
        self.cam = sl.Camera()
        self.opened = False
        # Three separate counters, because one number cannot distinguish
        # "captured" from "kept". grabbed == enqueued + backpressure_drops.
        self.grabbed = 0
        self.enqueued = 0
        self.backpressure_drops = 0
        self.dropped_total = 0        # the SDK's own acquisition drop counter
        self.error: str | None = None

    def open(self) -> bool:
        init = sl.InitParameters()
        init.set_from_svo_file(str(self.svo_path))
        init.svo_real_time_mode = False        # process as fast as possible
        init.depth_mode = sl.DEPTH_MODE.NONE   # not needed; saves a lot of GPU work
        init.coordinate_units = sl.UNIT.METER

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.error = f"open failed: {status}"
            return False

        self.opened = True
        info = self.cam.get_camera_information()
        print(f"  [{self.device_id}] {self.svo_path.name}  "
              f"model={info.camera_model}  serial={info.serial_number}  "
              f"frames={self.cam.get_svo_number_of_frames()}")
        return True

    def run(self):
        # try/finally is not decoration. Without it, any exception from the SDK
        # kills this thread before the sentinel is enqueued, and main()'s drain
        # loop then waits for a sentinel that will never arrive: a silent hang
        # with no error message, mid-survey.
        try:
            rt = sl.RuntimeParameters()
            while not self.stop.is_set():
                err = self.cam.grab(rt)
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                if err != sl.ERROR_CODE.SUCCESS:
                    self.error = f"grab failed: {err}"
                    self.stop.set()          # a dead camera invalidates the whole set
                    break

                f = Frame(
                    device=self.device_id,
                    svo_index=self.cam.get_svo_position(),
                    timestamp_ns=self.cam.get_timestamp(
                        sl.TIME_REFERENCE.IMAGE).get_nanoseconds(),
                    dropped_so_far=self.cam.get_frame_dropped_count(),
                )
                self.grabbed += 1
                try:
                    self.out_q.put(f, timeout=1.0)
                    self.enqueued += 1
                except queue.Full:
                    # The frame existed and was thrown away. Count it, do not
                    # merely print it, or it vanishes from every statistic.
                    self.backpressure_drops += 1
                    print(f"  [{self.device_id}] queue full, dropped frame {f.svo_index}")
        except Exception as exc:                     # noqa: BLE001 - deliberate
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop.set()
        finally:
            # Read the drop count BEFORE closing. Asking a closed camera returns
            # 0, which reads as "nothing dropped" when it means "nobody home".
            try:
                self.dropped_total = self.cam.get_frame_dropped_count()
            except Exception:                        # noqa: BLE001
                self.dropped_total = -1              # -1 = could not be read
            self.out_q.put(None)      # sentinel, guaranteed on every exit path
            try:
                self.cam.close()
            except Exception:                        # noqa: BLE001
                pass


# ------------------------------------------------------------------------ GNSS source

class GnssSource:
    """Synthetic GNSS along a straight line at constant speed.

    Replace generate() with a real reader (pyserial + pynmea2, or gpsd) and the
    rest of this file does not change. That is the point of keeping it behind
    a class with one method.
    """

    def __init__(self, start_ns: int, duration_s: float, rate_hz: float,
                 speed_kmh: float = 60.0, curve_radius_m: float = 0.0):
        self.fixes: list[GnssFix] = []
        self.nominal_rate_hz = rate_hz
        speed = speed_kmh / 3.6
        step_ns = int(1e9 / rate_hz)
        lat0, lon0 = -33.8688, 151.2093        # Sydney

        # Derive metres-per-degree from the SAME sphere radius haversine_m uses.
        # Using 111_320 here while measuring with R=6371000 put a 0.11 percent
        # units mismatch between the track we generate and the track we measure,
        # which showed up as ~1 m missing from a 996 m synthetic run.
        # A real receiver reports WGS84 ellipsoidal coordinates, where the
        # spherical approximation is ~0.2 percent out at this latitude. That is
        # acceptable for short relative baselines and would need a local ENU
        # projection for production absolute accuracy.
        m_per_deg_lat = math.radians(1.0) * EARTH_R_M
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))

        n = int(duration_s * rate_hz) + 2
        for i in range(n):
            t = start_ns + i * step_ns
            travelled = speed * (i / rate_hz)
            if curve_radius_m > 0:
                # Constant-radius arc, so heading sweeps instead of sitting at a
                # single value. A straight track cannot exercise heading at all.
                theta = travelled / curve_radius_m
                north = curve_radius_m * math.sin(theta)
                east = curve_radius_m * (1.0 - math.cos(theta))
            else:
                north, east = travelled, 0.0
            self.fixes.append(GnssFix(
                timestamp_ns=t,
                lat=lat0 + north / m_per_deg_lat,
                lon=lon0 + east / m_per_deg_lon,
                alt=25.0,
                fix_type="RTK_FIX",
                speed_mps=speed,
            ))
        self._measure_cadence()

    def _measure_cadence(self) -> None:
        """Work out this receiver's normal fix interval from the fixes themselves.

        A real receiver's rate is whatever it is, and it can change mid-survey.
        Measuring it beats hardcoding a constant that quietly assumes one rate
        and rejects every slower receiver as faulty.
        """
        # Cache the timestamp array once. Rebuilding it inside position_at()
        # defeats the entire point of using bisect there.
        self._ts = [f.timestamp_ns for f in self.fixes]

        raw = [b - a for a, b in zip(self._ts, self._ts[1:])]
        self.out_of_order_fixes = sum(1 for g in raw if g <= 0)
        if self.out_of_order_fixes:
            # bisect silently returns nonsense on an unsorted list, so this must
            # be loud. A real NMEA stream does deliver out-of-order sentences.
            print(f"  WARNING: {self.out_of_order_fixes} GNSS fixes are not in "
                  f"ascending time order; interpolation results are unreliable")

        gaps = sorted(raw)
        if not gaps:
            # No cadence can be measured. Fall back to the declared rate rather
            # than to zero, which would make every span "too large" and produce
            # a complete manifest containing no positions at all.
            self.median_interval_ns = int(1e9 / self.nominal_rate_hz) if self.nominal_rate_hz else 0
            self.cadence_measured = False
        else:
            self.median_interval_ns = gaps[len(gaps) // 2]
            self.cadence_measured = True
        # Allow interpolation across a modest hiccup, but not a real dropout.
        self.max_span_ns = int(self.median_interval_ns * 2.5)

    def position_at(self, ts_ns: int) -> tuple[GnssFix | None, float, str]:
        """Estimate the position at an exact frame timestamp.

        Returns (fix, gap_ms, method).

        Interpolates between the two fixes that bracket ts_ns rather than
        snapping to whichever is closer. The vehicle is moving continuously, so
        a point between two fixes is a far better estimate than either end.

        Why it matters: at 10 Hz a fix arrives every 100 ms. Nearest-neighbour
        can therefore be up to 50 ms wrong, which at 60 km/h is 0.83 m of
        position error, larger than the 0.5 m capture interval we are trying to
        resolve. Interpolation removes that error almost entirely, because the
        residual is only the vehicle's deviation from constant velocity over
        one 100 ms window.

        gap_ms is still reported: it is the distance to the nearer bracketing
        fix, and it is the honest measure of how much we are inferring.
        """
        if not self.fixes:
            return None, float("inf"), "no_fixes"

        ts = self._ts                       # cached; do not rebuild per call
        i = bisect.bisect_left(ts, ts_ns)

        # An exact hit is not an edge case and must not be routed into the
        # clamp branch. main() starts the GNSS track at min(all_ts), so the
        # earliest frame's timestamp equals the first fix exactly, every run.
        if i < len(ts) and ts[i] == ts_ns:
            return self.fixes[i], 0.0, "exact"

        # Outside the recorded track. Do not extrapolate; say so instead.
        if i == 0 or i == len(self.fixes):
            edge = self.fixes[0] if i == 0 else self.fixes[-1]
            gap_ms = abs(edge.timestamp_ns - ts_ns) / 1e6
            if gap_ms * 1e6 > self.max_span_ns / 2:
                return None, gap_ms, "out_of_range"
            return edge, gap_ms, "edge_clamp"

        before, after = self.fixes[i - 1], self.fixes[i]
        span_ns = after.timestamp_ns - before.timestamp_ns
        gap_ms = min(ts_ns - before.timestamp_ns, after.timestamp_ns - ts_ns) / 1e6

        if span_ns <= 0:
            return before, gap_ms, "duplicate_timestamp"
        if span_ns > self.max_span_ns:
            # A hole in the GNSS record, e.g. a tunnel or a lost fix. This is a
            # gap relative to THIS receiver's normal cadence, not an absolute
            # threshold, so a 1 Hz receiver is not condemned for behaving like
            # a 1 Hz receiver.
            return None, gap_ms, "gnss_gap_too_large"

        t = (ts_ns - before.timestamp_ns) / span_ns      # 0.0 at before, 1.0 at after

        def lerp(a, b):
            return a + (b - a) * t

        fix = GnssFix(
            timestamp_ns=ts_ns,                          # the FRAME's time, not the fix's
            lat=lerp(before.lat, after.lat),
            lon=lerp(before.lon, after.lon),
            alt=lerp(before.alt, after.alt),
            fix_type=before.fix_type if before.fix_type == after.fix_type else "MIXED",
            speed_mps=lerp(before.speed_mps, after.speed_mps),
        )
        return fix, gap_ms, "interpolated"

    def nearest(self, ts_ns: int) -> tuple[GnssFix | None, float]:
        """Nearest fix in time, plus the gap in ms. Returns (None, inf) if too stale.

        Returns (None, inf) ONLY when there are no fixes at all. There is no
        staleness filter, deliberately: filtering would discard exactly the
        worst cases and make the naive method look better than it is. The point
        of this method is to show honestly what it costs.

        Kept so the two methods can be compared directly. This is the naive
        approach and it stair-steps: several frames share one position.
        """
        if not self.fixes:
            return None, float("inf")
        ts = self._ts
        j = bisect.bisect_left(ts, ts_ns)
        cands = [c for c in (j - 1, j) if 0 <= c < len(ts)]
        best = min(cands, key=lambda c: abs(ts[c] - ts_ns))
        return self.fixes[best], abs(ts[best] - ts_ns) / 1e6


# -------------------------------------------------------------- geodesy and track maths

EARTH_R_M = 6_371_000.0
STATIONARY_M = 0.01      # below this, direction of travel is not meaningful


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres.

    Not flat-earth: a degree of longitude shrinks as cos(latitude), and at Sydney's
    -33.87 deg that is a 17 percent error if ignored. Over 0.5 m steps the absolute
    error is small, but it accumulates across a 10 km survey.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing, degrees clockwise from true north, 0 to 360.

    atan2(dlon, dlat) would be WRONG. Degrees of longitude are shorter than degrees
    of latitude everywhere except the equator, so the naive version skews the angle
    by the cos(latitude) factor that appears in the x term below.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def add_track_metrics(rows: list[dict]) -> None:
    """Add cumulative along-track distance and heading to each positioned row.

    Distance is the running sum of point-to-point great-circle steps. Note this
    measures the CHORD path through the sampled points, so on a curve it slightly
    under-measures the true arc. With interpolated points every frame the sampling
    is dense and the error is negligible; with sparse points it would not be.

    Heading uses a CENTRAL difference, i.e. the bearing from the previous position
    to the next one, which is centred on the current frame rather than lagging it
    by half a sample. Endpoints fall back to a one-sided difference.

    A stationary vehicle has no direction of travel. Rather than emit a spurious
    heading from GNSS jitter, the last known heading is held and the row is marked,
    so a consumer can tell a real heading from a held one.
    """
    pos = [i for i, r in enumerate(rows) if r.get("lat") is not None]

    cum = 0.0
    prev = None
    for i in pos:
        r = rows[i]
        if prev is not None:
            cum += haversine_m(rows[prev]["lat"], rows[prev]["lon"], r["lat"], r["lon"])
        r["track_m"] = round(cum, 4)
        prev = i

    last_heading = None
    for n, i in enumerate(pos):
        r = rows[i]
        a = rows[pos[n - 1]] if n > 0 else r            # central difference where
        b = rows[pos[n + 1]] if n < len(pos) - 1 else r  # possible, one-sided at ends
        step = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
        if step < STATIONARY_M:
            r["heading_deg"] = last_heading
            r["heading_source"] = "held_stationary" if last_heading is not None else "unknown"
        else:
            h = bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])
            r["heading_deg"] = round(h, 3)
            r["heading_source"] = "derived"
            last_heading = round(h, 3)

    for r in rows:
        r.setdefault("track_m", None)
        r.setdefault("heading_deg", None)
        r.setdefault("heading_source", "no_position")


# ------------------------------------------------------------------- distance trigger

def select_at_interval(rows: list[dict], interval_m: float,
                       tolerance_m: float) -> tuple[list[dict], dict]:
    """Choose the frame nearest each multiple of interval_m along the track.

    This is POST HOC SELECTION from a fixed-rate recording, not a hardware trigger.
    It answers "could a capture at every N metres have been served by this frame
    rate?" It does not demonstrate commanding a shutter. A real distance trigger
    needs a pulse into the camera's sync input, driven by a DMI or a GNSS-derived
    distance accumulator, and that cannot be shown from a recording.

    Nearest-frame selection is used rather than first-frame-past-the-mark because
    it halves the worst-case spacing error: past-the-mark can only ever be late,
    while nearest can be early or late and is therefore bounded by half the frame
    spacing instead of a whole one.

    A mark whose nearest frame is farther than tolerance_m is reported as MISSED
    rather than quietly filled with a distant frame. When the frame rate is too low
    for the speed, one frame can also be the best answer for two adjacent marks;
    that is counted as aliasing, because the second mark has no independent image.
    """
    positioned = [r for r in rows if r.get("track_m") is not None]
    if len(positioned) < 2:
        return [], {"marks": 0, "reason": "insufficient positioned frames"}

    track = [r["track_m"] for r in positioned]
    total = track[-1]
    n_marks = int(total // interval_m) + 1

    selected, used = [], {}
    hits = misses = aliased = 0
    errors = []

    for k in range(n_marks):
        target = k * interval_m
        j = bisect.bisect_left(track, target)
        # bisect gives the insertion point; the nearest is that one or the one before
        cands = [c for c in (j - 1, j) if 0 <= c < len(track)]
        best = min(cands, key=lambda c: abs(track[c] - target))
        err = abs(track[best] - target)

        r = positioned[best]
        is_alias = best in used
        if err <= tolerance_m:
            hits += 1
        else:
            misses += 1
        if is_alias:
            aliased += 1
        errors.append(err)
        used[best] = used.get(best, 0) + 1

        selected.append({
            "mark_index": k,
            "target_m": round(target, 4),
            "set_id": r["set_id"],
            "actual_m": r["track_m"],
            "error_m": round(err, 4),
            "within_tolerance": err <= tolerance_m,
            "reuses_earlier_frame": is_alias,
            "timestamp_ns": r["timestamp_ns"],
            "lat": r["lat"], "lon": r["lon"],
            "heading_deg": r.get("heading_deg"),
        })

    # Real measured frame-to-frame spacing, from the positions themselves.
    # A single mean would hide the tail, and the tail is what decides whether
    # every 0.5 m mark can be served. Report the distribution.
    steps = sorted(track[i] - track[i - 1] for i in range(1, len(track)))
    def pct(p):
        return steps[min(len(steps) - 1, int(len(steps) * p))] if steps else float("nan")

    stats = {
        "marks": n_marks,
        "interval_m": interval_m,
        "tolerance_m": tolerance_m,
        "track_length_m": round(total, 2),
        "hits": hits,
        "misses": misses,
        "aliased": aliased,
        "distinct_frames_used": len(used),
        "mean_error_m": round(sum(errors) / len(errors), 4) if errors else None,
        "max_error_m": round(max(errors), 4) if errors else None,
        "frame_spacing_mean_m": round(sum(steps) / len(steps), 4) if steps else None,
        "frame_spacing_p95_m": round(pct(0.95), 4) if steps else None,
        "frame_spacing_max_m": round(steps[-1], 4) if steps else None,
        "steps_exceeding_interval": sum(1 for s in steps if s > interval_m),
    }
    return selected, stats


# ---------------------------------------------------------------------- frame matcher

def match_frames(per_device: dict[str, list[Frame]], tolerance_ms: float) -> list[dict]:
    """Pair frames across devices into capture sets by timestamp proximity.

    Uses the first device as the reference and finds the closest frame on every
    other device.

    Three things here are deliberate and were each a defect at some point:

    1. SPREAD IS RECORDED BEFORE THE TOLERANCE TEST. Recording it inside the
       `gap <= tolerance` branch bounds the reported maximum by the tolerance
       itself, so the headline "max camera spread" becomes a restatement of the
       filter rather than a measurement. It could never exceed the threshold no
       matter how badly the cameras were desynchronised.

    2. A CANDIDATE IS CONSUMED. Without exclusivity the same frame can be the
       nearest match for many reference frames, so one image gets logged at many
       different GNSS positions, and frames that are nearest to nothing vanish
       from the manifest silently. A claimed frame is refused and the set is
       marked incomplete with a reason.

    3. BISECT, NOT min(). A linear scan per reference frame is O(R x D x F). At
       the deliverable's scale (30 min, 60 fps, 5 cameras) that is ~5e10 Python
       comparisons, i.e. it cannot be run on real data at all.
    """
    devices = list(per_device)
    if not devices:
        return []
    ref_name = devices[0]
    others = devices[1:]

    # Sorted timestamp arrays once, not once per reference frame.
    ts_by_dev = {n: [f.timestamp_ns for f in per_device[n]] for n in others}
    claimed: dict[str, set[int]] = {n: set() for n in others}
    sets = []

    for ref in per_device[ref_name]:
        row = {"ref_device": ref_name, "ref_index": ref.svo_index,
               "timestamp_ns": ref.timestamp_ns, "members": {ref_name: ref.svo_index}}
        spreads, reasons = [], []
        complete = True

        for name in others:
            ts = ts_by_dev[name]
            if not ts:
                complete = False
                reasons.append(f"{name}:no_frames")
                continue
            j = bisect.bisect_left(ts, ref.timestamp_ns)
            cands = [c for c in (j - 1, j) if 0 <= c < len(ts)]
            best = min(cands, key=lambda c: abs(ts[c] - ref.timestamp_ns))
            gap_ms = abs(ts[best] - ref.timestamp_ns) / 1e6

            # Unconditional: this is the measurement, not the filter.
            spreads.append(gap_ms)

            if best in claimed[name]:
                complete = False
                reasons.append(f"{name}:frame_already_claimed")
            elif gap_ms <= tolerance_ms:
                row["members"][name] = per_device[name][best].svo_index
                claimed[name].add(best)
            else:
                complete = False
                reasons.append(f"{name}:outside_tolerance")

        # With a single device there is nothing to match against, so "complete"
        # would otherwise be trivially true for every row and the tool would
        # report 100 percent synchronisation for a one-camera run.
        row["complete"] = complete if others else None
        row["max_spread_ms"] = round(max(spreads), 3) if spreads else None
        row["match_notes"] = ";".join(reasons)
        sets.append(row)
    return sets


# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svo", nargs="+", required=True, help="two or more SVO files")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--gnss-hz", type=float, default=GNSS_RATE_HZ)
    ap.add_argument("--speed-kmh", type=float, default=60.0)
    ap.add_argument("--tolerance-ms", type=float, default=MATCH_TOLERANCE_MS)
    ap.add_argument("--interval-m", type=float, default=0.5,
                    help="distance-trigger interval in metres (client default 0.5)")
    ap.add_argument("--interval-tolerance-m", type=float, default=None,
                    help="a mark counts as hit within this distance. "
                         "Defaults to half the interval.")
    ap.add_argument("--curve-radius-m", type=float, default=0.0,
                    help="synthetic track curve radius in metres. 0 = straight. "
                         "A straight track holds heading constant and therefore "
                         "cannot exercise the heading calculation at all.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Opening cameras")
    stop = threading.Event()
    workers, queues = [], {}
    for i, path in enumerate(args.svo):
        dev = f"cam{i}"
        q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        w = CameraWorker(Path(path), dev, q, stop)
        if not w.open():
            raise SystemExit(f"  [{dev}] {w.error}")
        workers.append(w)
        queues[dev] = q

    print("\nCapturing, one thread per camera")
    t0 = time.time()
    for w in workers:
        w.start()

    per_device: dict[str, list[Frame]] = {w.device_id: [] for w in workers}
    finished = set()
    while len(finished) < len(workers):
        for dev, q in queues.items():
            if dev in finished:
                continue
            try:
                item = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                finished.add(dev)
            else:
                per_device[dev].append(item)

    for w in workers:
        w.join(timeout=5)
    elapsed = time.time() - t0

    print(f"\nCaptured in {elapsed:.1f}s")
    for w in workers:
        line = (f"  [{w.device_id}] grabbed {w.grabbed}, enqueued {w.enqueued}, "
                f"backpressure-dropped {w.backpressure_drops}, "
                f"SDK acquisition-drop counter {w.dropped_total}")
        if w.error:
            line += f"\n           ERROR: {w.error}"
        if w.is_alive():
            line += "\n           WARNING: thread did not exit within 5 s"
        print(line)

    # A camera that died mid-run must not yield a clean-looking deliverable.
    # Previously the error was printed and the run still exited 0 with a full
    # manifest.csv, which is exactly how a partial survey gets accepted as good.
    failed = [(w.device_id, w.error) for w in workers if w.error]
    degraded = failed or any(w.backpressure_drops for w in workers)

    counts = {w.device_id: len(per_device[w.device_id]) for w in workers}
    if len(counts) > 1 and max(counts.values()) - min(counts.values()) > 2:
        print(f"\n  WARNING: uneven frame counts across cameras {counts}. "
              f"Capture-set totals are bounded by the reference camera.")
        degraded = True

    all_ts = [f.timestamp_ns for fs in per_device.values() for f in fs]
    if not all_ts:
        raise SystemExit("No frames captured. Check the SVO paths.")
    if not per_device[workers[0].device_id]:
        raise SystemExit(f"Reference device {workers[0].device_id} produced no frames "
                         f"while others did; there is nothing to match against.")
    start_ns, end_ns = min(all_ts), max(all_ts)
    duration_s = (end_ns - start_ns) / 1e9

    print(f"\nGenerating synthetic GNSS at {args.gnss_hz} Hz, {args.speed_kmh} km/h")
    gnss = GnssSource(start_ns, duration_s, args.gnss_hz, args.speed_kmh,
                      curve_radius_m=args.curve_radius_m)
    print(f"  {len(gnss.fixes)} fixes over {duration_s:.2f}s")

    print(f"\nMatching frames, tolerance {args.tolerance_ms} ms")
    sets = match_frames(per_device, args.tolerance_ms)

    rows = []
    for s in sets:
        fix, gap_ms, method = gnss.position_at(s["timestamp_ns"])
        # Also compute the naive result, purely so the manifest shows what
        # interpolation bought us. Drop this column in production.
        near, near_gap = gnss.nearest(s["timestamp_ns"])
        rows.append({
            "set_id": len(rows),
            "timestamp_ns": s["timestamp_ns"],
            "complete_set": s["complete"],
            "max_camera_spread_ms": s["max_spread_ms"],
            "frames": json.dumps(s["members"]),
            "gnss_method": method,
            "gnss_gap_ms": round(gap_ms, 3) if fix else None,
            "lat": fix.lat if fix else None,
            "lon": fix.lon if fix else None,
            "alt": fix.alt if fix else None,
            "fix_type": fix.fix_type if fix else "NO_FIX",
            "speed_mps": fix.speed_mps if fix else None,
            "lat_nearest_only": near.lat if near else None,
            "lon_nearest_only": near.lon if near else None,
            "nearest_gap_ms": round(near_gap, 3) if near else None,
        })

    # Track distance and heading must exist before the distance trigger can run.
    add_track_metrics(rows)
    tol_m = (args.interval_tolerance_m if args.interval_tolerance_m is not None
             else args.interval_m / 2.0)
    marks, mark_stats = select_at_interval(rows, args.interval_m, tol_m)

    # Name the file after its trustworthiness. A degraded run must not produce
    # something a downstream consumer would pick up as a normal manifest.
    stem = "manifest.PARTIAL" if degraded else "manifest"
    csv_path, json_path = out / f"{stem}.csv", out / f"{stem}.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "distance_trigger": mark_stats,
                   "sets": rows}, fh, indent=1)

    # The distance-triggered selection is the deliverable's actual output: one
    # capture per interval, not one per frame. Written separately so it can be
    # handed over without the diagnostic columns.
    if marks:
        marks_path = out / "captures_at_interval.csv"
        with open(marks_path, "w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(marks[0]))
            wtr.writeheader()
            wtr.writerows(marks)

    complete = sum(1 for r in rows if r["complete_set"])
    spreads = [r["max_camera_spread_ms"] for r in rows if r["max_camera_spread_ms"] is not None]
    positioned = sum(1 for r in rows if r["lat"] is not None)

    print(f"\nResult")
    print(f"  capture sets            {len(rows)}")
    print(f"  complete sets           {complete}  ({complete/len(rows)*100:.1f}%)")
    if spreads:
        print(f"  camera spread ms        min {min(spreads):.2f}  "
              f"mean {sum(spreads)/len(spreads):.2f}  max {max(spreads):.2f}")
    print(f"  sets with a position    {positioned}  ({positioned/len(rows)*100:.1f}%)")

    # What interpolation actually bought us, in metres on the ground.
    uniq_interp = len({(r["lat"], r["lon"]) for r in rows if r["lat"] is not None})
    uniq_near = len({(r["lat_nearest_only"], r["lon_nearest_only"]) for r in rows
                     if r["lat_nearest_only"] is not None})
    near_gaps = [r["nearest_gap_ms"] for r in rows if r["nearest_gap_ms"] is not None]
    if near_gaps:
        speed_ms = args.speed_kmh / 3.6
        worst_m = max(near_gaps) / 1000.0 * speed_ms
        mean_m = (sum(near_gaps) / len(near_gaps)) / 1000.0 * speed_ms
        print(f"\n  GNSS association")
        print(f"    distinct positions      {uniq_interp} interpolated  vs  {uniq_near} nearest-only")
        print(f"    nearest-only error      mean {mean_m:.2f} m, worst {worst_m:.2f} m "
              f"at {args.speed_kmh:g} km/h")
        print(f"    interpolated residual   limited by how much the vehicle deviates from")
        print(f"                            constant velocity over one {1000/args.gnss_hz:.0f} ms window")
    # ---- heading
    derived = sum(1 for r in rows if r.get("heading_source") == "derived")
    held = sum(1 for r in rows if r.get("heading_source") == "held_stationary")
    hd = [r["heading_deg"] for r in rows if r.get("heading_deg") is not None]
    if hd:
        print(f"\n  Heading")
        print(f"    derived {derived}, held while stationary {held}")
        print(f"    range {min(hd):.1f} to {max(hd):.1f} deg true")

    # ---- distance trigger
    if mark_stats.get("marks"):
        s = mark_stats
        print(f"\n  Distance trigger, every {s['interval_m']:g} m "
              f"(tolerance {s['tolerance_m']:g} m)")
        print(f"    track length          {s['track_length_m']:g} m")
        print(f"    marks required        {s['marks']}")
        print(f"    within tolerance      {s['hits']}  ({s['hits']/s['marks']*100:.1f}%)")
        print(f"    outside tolerance     {s['misses']}")
        print(f"    reused an earlier frame {s['aliased']}   "
              f"(distinct frames {s['distinct_frames_used']})")
        print(f"    spacing error         mean {s['mean_error_m']:g} m, "
              f"max {s['max_error_m']:g} m")
        print(f"    frame spacing         mean {s['frame_spacing_mean_m']:g} m, "
              f"p95 {s['frame_spacing_p95_m']:g} m, max {s['frame_spacing_max_m']:g} m")
        print(f"    steps wider than {s['interval_m']:g} m: {s['steps_exceeding_interval']}")
        need_fps = (args.speed_kmh / 3.6) / s["interval_m"]
        print(f"    rate needed for {s['interval_m']:g} m at {args.speed_kmh:g} km/h: "
              f"{need_fps:.1f} fps")
        if s["aliased"]:
            print(f"    ^ frame rate is too low for this interval and speed. "
                  f"{s['aliased']} marks have no independent image.")

    print(f"\n  {csv_path}")
    print(f"  {json_path}")
    if marks:
        print(f"  {out / 'captures_at_interval.csv'}")

    # Deliberately NOT args.speed_kmh * duration. That reported the assumed
    # command-line speed back as though it were a measurement, and would keep
    # doing so unchanged once a real receiver replaced GnssSource. Distance now
    # comes from the interpolated positions themselves, via track_m, and is
    # printed in the distance-trigger block above.

    if degraded:
        print("\n  RUN DEGRADED. Manifest written with a PARTIAL name.")
        for dev, err in failed:
            print(f"    [{dev}] {err}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

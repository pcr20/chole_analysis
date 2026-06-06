#!/usr/bin/env python3
r"""
crcd_batch_angular_path.py  (v2 - per-instrument, +roll, +ROM bounds)
---------------------------------------------------------------------
Cycle through every CRCD procedure and, for each PSM, break the recording down
BY INSTRUMENT (tools are swapped mid-procedure) and report, per (arm, instrument):

  * total angular distance travelled (total variation) for PITCH, YAW, ROLL
  * the 95% range-of-motion envelope (lower & upper percentile bounds) for each

Angles are the dVRK PSM joint positions (joint space):
    shaft ROLL  = "outer_roll"        (joint 3)
    wrist PITCH = "outer_wrist_pitch" (joint 4)
    wrist YAW   = "outer_wrist_yaw"   (joint 5)
Joints are resolved by NAME when /PSMx/measured_js carries names, else by index.

Instrument identity comes from the dVRK tool-type topic (a std_msgs/String that
changes when a tool is swapped). It is auto-detected, can be overridden with
--tool-topic, and you can list everything in a bag with --list-topics. If no tool
topic exists in the bag, the whole arm is reported as one instrument = "all".

Total travel is summed within each contiguous run of an instrument (so the jump
across a swap-out / swap-back gap is NOT counted). ROM bounds pool all samples of
that instrument.

Reader: `rosbags` (pure Python, no ROS install).  Download modes unchanged from v1
(--local recommended; --box-url [--box-token] for automated pulls).

    pip install rosbags numpy pandas requests
    python crcd_batch_angular_path.py --local .\data --list-topics      # inspect one bag
    python crcd_batch_angular_path.py --local .\data                    # full run
"""

import argparse
import os
import re
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# dVRK classic PSM joint order (0-indexed) — defaults if names are absent
DEFAULT_IDX = {"roll": 3, "pitch": 4, "yaw": 5}
BOX_URL_DEFAULT = "https://uofi.box.com/s/p3aocj6yzq4ctwc0s635a2dfyk9zdv5j"


# =============================================================== math
def total_variation_deg(theta_rad, deadband_deg=0.0):
    if len(theta_rad) < 2:
        return 0.0
    th = np.unwrap(np.asarray(theta_rad, dtype=float))
    d = np.degrees(np.diff(th))
    if deadband_deg > 0:
        d = d[np.abs(d) > deadband_deg]
    return float(np.sum(np.abs(d)))


def rom_bounds_deg(theta_rad, pct=95.0):
    """Lower/upper percentile bounds spanning `pct`% of the motion, in degrees."""
    if len(theta_rad) == 0:
        return (np.nan, np.nan, np.nan)
    th = np.degrees(np.unwrap(np.asarray(theta_rad, dtype=float)))
    lo_q = (100.0 - pct) / 2.0
    hi_q = 100.0 - lo_q
    lo, hi = np.percentile(th, [lo_q, hi_q])
    return (float(lo), float(hi), float(hi - lo))


def abs_speed_deg_s(theta_rad, t):
    """Per-sample |angular velocity| in deg/s within ONE contiguous run.
    Uses the actual timestamps (handles sampling jitter). Direction-agnostic."""
    if len(theta_rad) < 2:
        return np.array([])
    th = np.degrees(np.unwrap(np.asarray(theta_rad, dtype=float)))
    dt = np.diff(np.asarray(t, dtype=float))
    dth = np.abs(np.diff(th))
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = dth / dt
    return spd[np.isfinite(spd) & (dt > 0)]


def abs_step_deg(theta_rad):
    """Per-sample |Δθ| in degrees (the per-step increments that sum to travel)."""
    if len(theta_rad) < 2:
        return np.array([], dtype=np.float32)
    th = np.unwrap(np.asarray(theta_rad, dtype=float))
    return np.abs(np.degrees(np.diff(th))).astype(np.float32)


def count_actuations(jaw_rad, min_amp_deg=10.0):
    """Count grasp/cut actuations = jaw open->close transitions, with hysteresis.
    Returns (n_actuations, jaw_amplitude_deg). A non-jawed tool (e.g. cautery hook)
    has near-flat jaw -> amplitude below min_amp_deg -> 0 actuations."""
    if len(jaw_rad) < 3:
        return 0, 0.0
    jaw = np.degrees(np.asarray(jaw_rad, dtype=float))
    lo, hi = np.percentile(jaw, [5, 95])
    amp = float(hi - lo)
    if amp < min_amp_deg:                      # jaw essentially static -> not actuating
        return 0, amp
    close_thr = lo + 0.20 * amp                # hysteresis band avoids jitter recounts
    open_thr = lo + 0.60 * amp
    n, state = 0, None
    for v in jaw:
        if state is None:
            state = "open" if v >= open_thr else ("closed" if v <= close_thr else None)
        elif state == "open" and v <= close_thr:
            n += 1; state = "closed"           # one grasp/cut
        elif state == "closed" and v >= open_thr:
            state = "open"
    return n, amp


def resolve_joint_indices(names):
    """Map roll/pitch/yaw to joint indices using names; fall back to defaults."""
    idx = dict(DEFAULT_IDX)
    if names:
        for i, n in enumerate(names):
            nl = str(n).lower()
            if "roll" in nl:
                idx["roll"] = i
            elif "wrist" in nl and "pitch" in nl:
                idx["pitch"] = i
            elif "wrist" in nl and "yaw" in nl:
                idx["yaw"] = i
    return idx


# =============================================================== bag reading
def list_topics(bag_path):
    from rosbags.highlevel import AnyReader
    rows = []
    with AnyReader([Path(bag_path)]) as reader:
        for c in reader.connections:
            rows.append((c.topic, c.msgtype, c.msgcount))
    return sorted(set(rows))


def find_tool_topic(topics, arm, override):
    if override:
        return override
    cands = [t for (t, mt, _) in topics
             if mt.endswith("String")
             and arm.lower() in t.lower()
             and ("tool" in t.lower() or "instrument" in t.lower())]
    # prefer ones that look like a "type"/"name" channel
    cands.sort(key=lambda t: (("type" not in t.lower()) and ("name" not in t.lower()), len(t)))
    return cands[0] if cands else None


def read_arm(bag_path, arm, tool_topic):
    """Return per-sample arrays for one arm: t, roll, pitch, yaw (rad), the jaw
    angle series (jaw_t, jaw), and the tool-change series (tool_t, tool_name)."""
    from rosbags.highlevel import AnyReader

    js_topic = f"/{arm}/measured_js"
    jaw_topic = f"/{arm}/jaw/measured_js"
    t, roll, pitch, yaw = [], [], [], []
    jaw_t, jaw = [], []
    tool_t, tool_name = [], []
    idx = None

    with AnyReader([Path(bag_path)]) as reader:
        want = {js_topic, jaw_topic}
        if tool_topic:
            want.add(tool_topic)
        conns = [c for c in reader.connections if c.topic in want]
        if not any(c.topic == js_topic for c in conns):
            return None
        for con, ts_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)
            if con.topic == js_topic:
                pos = np.asarray(msg.position, dtype=float)
                if idx is None:
                    idx = resolve_joint_indices(list(getattr(msg, "name", []) or []))
                if len(pos) <= max(idx.values()):
                    continue
                roll.append(pos[idx["roll"]])
                pitch.append(pos[idx["pitch"]])
                yaw.append(pos[idx["yaw"]])
                t.append(ts_ns / 1e9)
            elif con.topic == jaw_topic:
                pos = np.asarray(msg.position, dtype=float)
                if pos.size:
                    jaw.append(pos[0]); jaw_t.append(ts_ns / 1e9)
            else:  # tool-type string
                tool_t.append(ts_ns / 1e9)
                tool_name.append(str(getattr(msg, "data", "")).strip())

    return {"t": np.array(t), "roll": np.array(roll),
            "pitch": np.array(pitch), "yaw": np.array(yaw),
            "jaw_t": np.array(jaw_t), "jaw": np.array(jaw),
            "tool_t": np.array(tool_t), "tool_name": tool_name,
            "joint_idx": idx}


# =============================================================== segmentation
def label_instruments(t, tool_t, tool_name):
    """Per-sample instrument label using the most recent tool message <= t."""
    if len(t) == 0:
        return np.array([], dtype=object)
    if len(tool_t) == 0:
        return np.array(["all"] * len(t), dtype=object)
    order = np.argsort(tool_t)
    tt = tool_t[order]
    names = np.array(tool_name, dtype=object)[order]
    pos = np.searchsorted(tt, t, side="right") - 1
    labels = np.where(pos >= 0, names[np.clip(pos, 0, len(names) - 1)], "unknown_pre")
    return labels.astype(object)


def segment_metrics(d, deadband, rom_pct, speed_pct, jaw_min_amp, downsample):
    """Return {instrument: metrics dict} for a single arm's data dict."""
    t = d["t"]
    if downsample > 1:
        sl = slice(None, None, downsample)
        t = t[sl]; roll = d["roll"][sl]; pitch = d["pitch"][sl]; yaw = d["yaw"][sl]
    else:
        roll, pitch, yaw = d["roll"], d["pitch"], d["yaw"]
    labels = label_instruments(t, d["tool_t"], d["tool_name"])

    # jaw stream is separate (own timestamps); label it independently
    jaw_t, jaw = d.get("jaw_t", np.array([])), d.get("jaw", np.array([]))
    jaw_labels = label_instruments(jaw_t, d["tool_t"], d["tool_name"])
    jaw_acc = {}
    if len(jaw_t):
        jch = np.where(jaw_labels[1:] != jaw_labels[:-1])[0] + 1
        for s, e in zip(np.concatenate(([0], jch)), np.concatenate((jch, [len(jaw_labels)]))):
            n, amp = count_actuations(jaw[s:e], jaw_min_amp)
            a = jaw_acc.setdefault(jaw_labels[s], {"n": 0, "amp": []})
            a["n"] += n
            a["amp"].append(jaw[s:e])

    results = {}
    pool = {}
    if len(t) == 0:
        return results, pool

    # contiguous runs (so travel is not counted across a swap-out/in gap)
    change = np.where(labels[1:] != labels[:-1])[0] + 1
    run_starts = np.concatenate(([0], change))
    run_ends = np.concatenate((change, [len(labels)]))

    for s, e in zip(run_starts, run_ends):
        name = labels[s]
        acc = results.setdefault(name, {
            "pitch_travel": 0.0, "yaw_travel": 0.0, "roll_travel": 0.0,
            "dur": 0.0, "n": 0,
            "pitch_vals": [], "yaw_vals": [], "roll_vals": [],
            "pitch_spd": [], "yaw_spd": [], "roll_spd": [],
            "pitch_step": [], "yaw_step": [], "roll_step": []})
        acc["pitch_travel"] += total_variation_deg(pitch[s:e], deadband)
        acc["yaw_travel"] += total_variation_deg(yaw[s:e], deadband)
        acc["roll_travel"] += total_variation_deg(roll[s:e], deadband)
        if e - s >= 1:
            acc["dur"] += float(t[e - 1] - t[s])
        acc["n"] += (e - s)
        acc["pitch_vals"].append(pitch[s:e])
        acc["yaw_vals"].append(yaw[s:e])
        acc["roll_vals"].append(roll[s:e])
        acc["pitch_spd"].append(abs_speed_deg_s(pitch[s:e], t[s:e]))
        acc["yaw_spd"].append(abs_speed_deg_s(yaw[s:e], t[s:e]))
        acc["roll_spd"].append(abs_speed_deg_s(roll[s:e], t[s:e]))
        acc["pitch_step"].append(abs_step_deg(pitch[s:e]))
        acc["yaw_step"].append(abs_step_deg(yaw[s:e]))
        acc["roll_step"].append(abs_step_deg(roll[s:e]))

    def pctl(chunks, q):
        arr = np.concatenate(chunks) if chunks else np.array([])
        return float(np.percentile(arr, q)) if arr.size else np.nan

    out = {}
    for name, a in results.items():
        pv = np.concatenate(a["pitch_vals"]); yv = np.concatenate(a["yaw_vals"]); rv = np.concatenate(a["roll_vals"])
        pl, pu, ps = rom_bounds_deg(pv, rom_pct)
        yl, yu, ys = rom_bounds_deg(yv, rom_pct)
        rl, ru, rs = rom_bounds_deg(rv, rom_pct)
        ja = jaw_acc.get(name, {"n": 0, "amp": []})
        jaw_amp = (float(np.degrees(np.percentile(np.concatenate(ja["amp"]), 95)
                                    - np.percentile(np.concatenate(ja["amp"]), 5)))
                   if ja["amp"] and np.concatenate(ja["amp"]).size else np.nan)
        out[name] = {
            "duration_s": round(a["dur"], 1), "n_samples": int(a["n"]),
            "jaw_actuations": int(ja["n"]),
            "jaw_amp_deg": round(jaw_amp, 1) if jaw_amp == jaw_amp else np.nan,
            "has_jaw": bool(ja["n"] > 0),
            "pitch_travel_deg": round(a["pitch_travel"], 1),
            "yaw_travel_deg": round(a["yaw_travel"], 1),
            "roll_travel_deg": round(a["roll_travel"], 1),
            "pitch_lo_deg": round(pl, 1), "pitch_hi_deg": round(pu, 1), "pitch_rom_deg": round(ps, 1),
            "yaw_lo_deg": round(yl, 1), "yaw_hi_deg": round(yu, 1), "yaw_rom_deg": round(ys, 1),
            "roll_lo_deg": round(rl, 1), "roll_hi_deg": round(ru, 1), "roll_rom_deg": round(rs, 1),
            "pitch_speed_p%g_degs" % speed_pct: round(pctl(a["pitch_spd"], speed_pct), 1),
            "yaw_speed_p%g_degs" % speed_pct: round(pctl(a["yaw_spd"], speed_pct), 1),
            "roll_speed_p%g_degs" % speed_pct: round(pctl(a["roll_spd"], speed_pct), 1),
        }
        # per-segment raw-sample pool for true global percentiles later (float32 to save RAM)
        def _cat32(chunks):
            return (np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32))
        pool[name] = {
            "pitch_angle_deg": np.degrees(np.unwrap(pv)).astype(np.float32),
            "yaw_angle_deg":   np.degrees(np.unwrap(yv)).astype(np.float32),
            "roll_angle_deg":  np.degrees(np.unwrap(rv)).astype(np.float32),
            "pitch_step_deg":  _cat32(a["pitch_step"]),
            "yaw_step_deg":    _cat32(a["yaw_step"]),
            "roll_step_deg":   _cat32(a["roll_step"]),
            "pitch_speed_degs": _cat32(a["pitch_spd"]),
            "yaw_speed_degs":   _cat32(a["yaw_spd"]),
            "roll_speed_degs":  _cat32(a["roll_spd"]),
        }
    return out, pool


def proc_name_from_path(p, root):
    base = os.path.basename(p)
    m = re.search(r"kinematics[_-](.+?)\.bag$", base, re.I)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
    rootbase = os.path.basename(os.path.abspath(root))
    if parent and parent != rootbase:
        return parent
    return os.path.splitext(base)[0]


# =============================================================== Box download (unchanged from v1)
def download_box(box_url, dest, token=None):
    import requests
    os.makedirs(dest, exist_ok=True)
    if token:
        return _box_api_download(box_url, dest, token, requests)
    print("[box] No --box-token; attempting anonymous public-share scrape "
          "(use --local if it fails).")
    try:
        return _box_anon_download(box_url, dest, requests)
    except Exception as e:
        print(f"[box] Anonymous download failed: {e}\n"
              f"[box] Open the link, click Download, unzip, then use --local.")
        return []


def _box_api_download(box_url, dest, token, requests):
    H = {"Authorization": f"Bearer {token}", "BoxApi": f"shared_link={box_url}"}
    root = requests.get("https://api.box.com/2.0/shared_items", headers=H, timeout=30).json()
    if root.get("type") != "folder":
        raise RuntimeError("Shared link is not a folder")
    paths = []

    def walk(folder_id, sub):
        offset = 0
        while True:
            r = requests.get(f"https://api.box.com/2.0/folders/{folder_id}/items", headers=H,
                             params={"limit": 1000, "offset": offset, "fields": "id,name,type"},
                             timeout=30).json()
            for it in r.get("entries", []):
                if it["type"] == "folder":
                    walk(it["id"], os.path.join(sub, it["name"]))
                elif it["type"] == "file" and re.search(r"kinematics.*\.bag$", it["name"], re.I):
                    outdir = os.path.join(dest, sub); os.makedirs(outdir, exist_ok=True)
                    out = os.path.join(outdir, it["name"])
                    if not os.path.exists(out):
                        print(f"[box] downloading {sub}/{it['name']}")
                        with requests.get(f"https://api.box.com/2.0/files/{it['id']}/content",
                                          headers=H, stream=True, timeout=600) as resp:
                            resp.raise_for_status()
                            with open(out, "wb") as f:
                                for chunk in resp.iter_content(1 << 20):
                                    f.write(chunk)
                    paths.append(out)
            offset += len(r.get("entries", []))
            if offset >= r.get("total_count", 0):
                break

    walk(root["id"], "")
    return paths


def _box_anon_download(box_url, dest, requests):
    s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0"
    r = s.get(box_url, timeout=30, allow_redirects=True)
    shared_name = box_url.rstrip("/").split("/")[-1]
    host = r.url.split("/")[2]
    m = re.search(r'"requestToken":"([^"]+)"', r.text) or re.search(r"requestToken\s*[:=]\s*'([^']+)'", r.text)
    if not m:
        raise RuntimeError("could not find requestToken on share page")
    rt = m.group(1)
    api = f"https://{host}/app-api/enduserapp/shared-folder"
    hdr = {"Request-Token": rt, "X-Request-Token": rt,
           "X-Box-Client-Name": "enduserapp", "Content-Type": "application/json"}
    paths = []

    def walk(folder_id, sub):
        body = {"shared_name": shared_name, "folder_id": folder_id, "offset": 0, "limit": 1000}
        j = s.post(api, json=body, headers=hdr, timeout=30).json()
        for it in j.get("items", j.get("entries", [])):
            tid = str(it.get("id") or it.get("typedID", "")).split("_")[-1]
            name = it.get("name", "")
            if it.get("type") == "folder":
                walk(tid, os.path.join(sub, name))
            elif re.search(r"kinematics.*\.bag$", name, re.I):
                outdir = os.path.join(dest, sub); os.makedirs(outdir, exist_ok=True)
                out = os.path.join(outdir, name)
                if not os.path.exists(out):
                    print(f"[box] downloading {sub}/{name}")
                    url = (f"https://{host}/index.php?rm=box_download_shared_file"
                           f"&shared_name={shared_name}&file_id=f_{tid}")
                    with s.get(url, stream=True, timeout=600) as resp:
                        resp.raise_for_status()
                        with open(out, "wb") as f:
                            for chunk in resp.iter_content(1 << 20):
                                f.write(chunk)
                paths.append(out)

    walk("0", "")
    return paths


# =============================================================== driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local", help="dir of kinematics bags (searched recursively)")
    src.add_argument("--box-url", nargs="?", const=BOX_URL_DEFAULT, help="Box shared-folder URL")
    ap.add_argument("--box-token")
    ap.add_argument("--dest", default="./crcd_bags")
    ap.add_argument("--arms", default="PSM1,PSM2")
    ap.add_argument("--tool-topic", help="override the instrument-type topic name")
    ap.add_argument("--deadband", type=float, default=0.10, help="noise gate, deg/sample")
    ap.add_argument("--rom-pct", type=float, default=95.0, help="central pct for ROM bounds")
    ap.add_argument("--speed-pct", type=float, default=95.0,
                    help="one-sided upper pct for |angular speed| (default 95)")
    ap.add_argument("--jaw-min-amp", type=float, default=10.0,
                    help="min jaw swing (deg) to count as an actuating tool (else 0)")
    ap.add_argument("--summary-pct", type=float, default=95.0,
                    help="upper percentile for the across-procedures summary (default 95)")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--out", default="crcd_angular_path_by_instrument.csv")
    ap.add_argument("--list-topics", action="store_true",
                    help="print topics of the first bag (find the tool topic) and exit")
    args = ap.parse_args()

    if args.local:
        bags = sorted(glob.glob(os.path.join(args.local, "**", "*kinematics*.bag"), recursive=True)) \
               or sorted(glob.glob(os.path.join(args.local, "**", "*.bag"), recursive=True))
        root_dir = args.local
    else:
        bags = sorted(download_box(args.box_url, args.dest, args.box_token))
        root_dir = args.dest
    if not bags:
        sys.exit("No kinematics bags found. See the DOWNLOAD notes in the header.")

    if args.list_topics:
        print(f"Topics in {bags[0]}:\n")
        for topic, mt, n in list_topics(bags[0]):
            flag = ("  <-- candidate tool topic"
                    if (mt.endswith("String")
                        and ("tool" in topic.lower() or "instrument" in topic.lower())) else "")
            print(f"  {n:>8}  {mt:<34} {topic}{flag}")
        return

    arms = [a.strip() for a in args.arms.split(",")]
    print(f"Found {len(bags)} bag(s). arms={arms} deadband={args.deadband} "
          f"rom={args.rom_pct}% downsample={args.downsample}x\n")

    rows = []
    global_pool = {}        # (arm, instrument) -> {key: [chunks of float32]}
    pool_keys = ["pitch_angle_deg", "yaw_angle_deg", "roll_angle_deg",
                 "pitch_step_deg", "yaw_step_deg", "roll_step_deg",
                 "pitch_speed_degs", "yaw_speed_degs", "roll_speed_degs"]
    for bag in bags:
        proc = proc_name_from_path(bag, root_dir)
        topics = list_topics(bag)
        for arm in arms:
            tool_topic = find_tool_topic(topics, arm, args.tool_topic)
            try:
                d = read_arm(bag, arm, tool_topic)
            except Exception as e:
                print(f"  ! {proc}/{arm}: {e}"); continue
            if d is None:
                continue
            seg, seg_pool = segment_metrics(d, args.deadband, args.rom_pct, args.speed_pct,
                                            args.jaw_min_amp, args.downsample)
            sp = args.speed_pct
            for instr, m in seg.items():
                rows.append({"procedure": proc, "arm": arm, "instrument": instr,
                             "tool_topic": tool_topic or "", **m})
                key = (instr, arm)
                gp = global_pool.setdefault(key, {k: [] for k in pool_keys})
                for k in pool_keys:
                    gp[k].append(seg_pool[instr][k])
                print(f"  {proc:>6} {arm} {str(instr):<24} dur {m['duration_s']:6.1f}s | "
                      f"actuations {m['jaw_actuations']:4d} (jaw {m['jaw_amp_deg']}\u00b0) | "
                      f"travel P/Y/R {m['pitch_travel_deg']:7.0f}/{m['yaw_travel_deg']:7.0f}/"
                      f"{m['roll_travel_deg']:7.0f} | "
                      f"v{sp:g} P/Y/R {m['pitch_speed_p%g_degs' % sp]:5.0f}/"
                      f"{m['yaw_speed_p%g_degs' % sp]:5.0f}/{m['roll_speed_p%g_degs' % sp]:5.0f} deg/s")
            if tool_topic is None:
                print(f"        (no tool topic for {arm}; reported as 'all'. "
                      f"Run --list-topics to check.)")

    if not rows:
        sys.exit("No data extracted.")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} rows)\n")

    pd.set_option("display.width", 220)
    sp = args.speed_pct
    sm = args.summary_pct
    spd_cols = ["pitch_speed_p%g_degs" % sp, "yaw_speed_p%g_degs" % sp, "roll_speed_p%g_degs" % sp]

    # ---- 1) Means across procedures (linear -> safe to aggregate from per-procedure rows)
    mean_cols = ["jaw_actuations",
                 "pitch_travel_deg", "yaw_travel_deg", "roll_travel_deg",
                 "pitch_rom_deg", "yaw_rom_deg", "roll_rom_deg",
                 *spd_cols, "duration_s"]
    mean_tbl = df.groupby(["instrument", "arm"])[mean_cols].mean().round(1)

    # ---- 2) p_sm across procedures, RESTRICTED to per-procedure scalars where
    #         this calculation is legitimate (travel and duration). ROM/speed/step
    #         require pooling, so they go in table 3; actuations also go there.
    scalar_cols = ["pitch_travel_deg", "yaw_travel_deg", "roll_travel_deg", "duration_s"]
    proc_p95_tbl = df.groupby(["instrument", "arm"])[scalar_cols].quantile(sm / 100.0).round(1)

    # ---- 3) p_sm computed on the FULL pooled sample stream per (instrument, arm).
    #         Per-sample data points -> single percentile -> avoids nested-histogram error.
    p95_rows = []
    act_p95 = df.groupby(["instrument", "arm"])["jaw_actuations"].quantile(sm / 100.0)
    for (instr, arm), gp in global_pool.items():
        row = {"instrument": instr, "arm": arm}
        for axis in ("pitch", "yaw", "roll"):
            step = np.concatenate(gp[f"{axis}_step_deg"]) if gp[f"{axis}_step_deg"] else np.array([])
            ang  = np.concatenate(gp[f"{axis}_angle_deg"]) if gp[f"{axis}_angle_deg"] else np.array([])
            spd  = np.concatenate(gp[f"{axis}_speed_degs"]) if gp[f"{axis}_speed_degs"] else np.array([])
            row[f"{axis}_step_deg"]    = round(float(np.percentile(step, sm)), 2) if step.size else np.nan
            row[f"{axis}_rom_hi_deg"]  = round(float(np.percentile(ang,  sm)), 1) if ang.size  else np.nan
            row[f"{axis}_speed_degs"]  = round(float(np.percentile(spd,  sm)), 1) if spd.size  else np.nan
        row["actuations_per_proc"]    = round(float(act_p95.get((instr, arm), np.nan)), 1)
        p95_rows.append(row)
    pool_p95_tbl = pd.DataFrame(p95_rows).set_index(["instrument", "arm"]).sort_index()

    # ---- print to stdout
    print("Mean per (instrument, arm) across procedures:")
    print(mean_tbl.to_string())
    print(f"\n{sm:g}th-percentile upper bound per (instrument, arm) across procedures "
          f"(per-procedure scalars only):")
    print(proc_p95_tbl.to_string())
    print(f"\n{sm:g}th-percentile upper bound, computed on the POOLED raw sample stream "
          f"per (instrument, arm):")
    print(pool_p95_tbl.to_string())
    print(f"  (step_deg = per-sample |Δθ|, the distribution that sums to travel;"
          f" rom_hi_deg = upper bound of pooled angle values;"
          f" speed_degs = pooled |dθ/dt|.")
    print(f"   actuations is the one exception — it's a per-procedure event count "
          f"so its p{sm:g} is across the {df.groupby(['instrument','arm']).size().max()} per-procedure counts.)")

    # ---- append the three tables to the CSV, with 2 blank rows before they start
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        f.write("\n\n")                                       # 2 blank rows
        f.write("# Mean per (instrument, arm) across procedures\n")
        mean_tbl.to_csv(f)
        f.write(f"\n# {sm:g}th-percentile upper bound across procedures "
                f"(per-procedure scalars only: travel + duration)\n")
        proc_p95_tbl.to_csv(f)
        f.write(f"\n# {sm:g}th-percentile upper bound, POOLED raw sample stream "
                f"per (instrument, arm)\n")
        f.write("# step_deg=per-sample |dtheta|; rom_hi_deg=upper of pooled angle values; "
                "speed_degs=pooled |dtheta/dt|; actuations_per_proc=p"
                f"{sm:g} of per-procedure counts\n")
        pool_p95_tbl.to_csv(f)
    print(f"\n(summary tables appended to {args.out})")


if __name__ == "__main__":
    main()

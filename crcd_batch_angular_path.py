#!/usr/bin/env python3
"""
crcd_batch_angular_path.py
--------------------------
Cycle through EVERY CRCD procedure (~21 kinematics recordings, surgeons A-G x3)
and compute the TOTAL ANGULAR DISTANCE travelled by each PSM instrument tip in
PITCH and YAW, then write a summary CSV.

For each bag and each arm (PSM1, PSM2) it reports BOTH interpretations:
  * wrist_joints : EndoWrist distal joints from /PSMx/measured_js
                   (dVRK classic order -> position[4]=wrist_pitch, position[5]=wrist_yaw)
  * tip_frame    : tip pointing direction from the quaternion in
                   /PSMx/custom/setpoint_cp (PSM tip in ECM-tip frame, g_et)

"Total angular distance" = total variation = sum_i |theta(i+1)-theta(i)|
(accumulated angular path, NOT the net start->end change).

Reader: `rosbags` (pure Python; no ROS install required).

--------------------------------------------------------------------------------
DOWNLOAD (read this — it's the fiddly part)
--------------------------------------------------------------------------------
The raw rosbags live in a Box shared FOLDER:
    https://uofi.box.com/s/p3aocj6yzq4ctwc0s635a2dfyk9zdv5j
Box shared folders have no stable anonymous per-file URLs, so fully unattended
download is unreliable. Three modes, most reliable first:

  (1) --local DIR        Process bags already on disk. RECOMMENDED.
                         One-time: open the Box link in a browser, hit
                         "Download" (folder downloads as a .zip), unzip, point
                         --local at the unzipped tree. The script finds every
                         *kinematics*.bag recursively.

  (2) --box-url URL --box-token TOKEN
                         Reliable automated download via the Box API. Get a
                         60-min developer token: developer.box.com -> create a
                         (free) Custom App -> "Generate Developer Token". The
                         script resolves the shared folder, walks it, and pulls
                         every *kinematics*.bag.

  (3) --box-url URL      Best-effort ANONYMOUS scrape of the public share (no
                         token). May break if Box changes its web app; if it
                         finds nothing it prints manual instructions and exits.

Examples
    pip install rosbags numpy pandas requests
    python crcd_batch_angular_path.py --local ./crcd_bags
    python crcd_batch_angular_path.py --box-url https://uofi.box.com/s/XXXX --box-token YYYY --dest ./crcd_bags
"""

import argparse
import os
import re
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

WRIST_PITCH_IDX, WRIST_YAW_IDX = 4, 5          # dVRK classic PSM joint order
WRIST_PITCH_NAME, WRIST_YAW_NAME = "wrist_pitch", "wrist_yaw"
BOX_URL_DEFAULT = "https://uofi.box.com/s/p3aocj6yzq4ctwc0s635a2dfyk9zdv5j"


# =============================================================== math
def quat_to_euler_zyx(x, y, z, w):
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw, pitch


def total_variation_deg(theta_rad, deadband_deg=0.0):
    if len(theta_rad) < 2:
        return 0.0
    th = np.unwrap(np.asarray(theta_rad, dtype=float))
    d = np.degrees(np.diff(th))
    if deadband_deg > 0:
        d = d[np.abs(d) > deadband_deg]
    return float(np.sum(np.abs(d)))


# =============================================================== bag reading
def read_arm(bag_path, arm):
    """Return dict of arrays: wrist_pitch, wrist_yaw (rad), tip_pitch, tip_yaw (rad), dur."""
    from rosbags.highlevel import AnyReader

    js_topic = f"/{arm}/measured_js"
    cp_topic = f"/{arm}/custom/setpoint_cp"
    wp, wy, tp, ty, t_js, t_cp = [], [], [], [], [], []

    with AnyReader([Path(bag_path)]) as reader:
        want = {js_topic, cp_topic}
        conns = [c for c in reader.connections if c.topic in want]
        if not conns:                       # arm absent -> don't let rosbags read ALL topics
            return {"wrist_pitch": np.array([]), "wrist_yaw": np.array([]),
                    "tip_pitch": np.array([]), "tip_yaw": np.array([]),
                    "n_js": 0, "n_cp": 0, "dur": 0.0}
        for con, ts_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)
            if con.topic == js_topic:
                names = list(getattr(msg, "name", []) or [])
                pos = np.asarray(msg.position, dtype=float)
                if WRIST_PITCH_NAME in names and WRIST_YAW_NAME in names:
                    wp.append(pos[names.index(WRIST_PITCH_NAME)])
                    wy.append(pos[names.index(WRIST_YAW_NAME)])
                elif len(pos) > WRIST_YAW_IDX:
                    wp.append(pos[WRIST_PITCH_IDX]); wy.append(pos[WRIST_YAW_IDX])
                t_js.append(ts_ns / 1e9)
            elif con.topic == cp_topic:
                q = msg.transform.rotation
                y_, p_ = quat_to_euler_zyx(q.x, q.y, q.z, q.w)
                tp.append(p_); ty.append(y_); t_cp.append(ts_ns / 1e9)

    allt = t_js + t_cp
    dur = (max(allt) - min(allt)) if allt else 0.0
    return {"wrist_pitch": np.array(wp), "wrist_yaw": np.array(wy),
            "tip_pitch": np.array(tp), "tip_yaw": np.array(ty),
            "n_js": len(t_js), "n_cp": len(t_cp), "dur": dur}


def proc_name_from_path(p):
    m = re.search(r"kinematics[_-](.+?)\.bag$", os.path.basename(p), re.I)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(p))[0]


# =============================================================== Box download
def download_box(box_url, dest, token=None):
    """Download every *kinematics*.bag from a Box shared folder into dest.
    Returns list of local bag paths. Best-effort; see module docstring."""
    import requests
    os.makedirs(dest, exist_ok=True)

    if token:
        return _box_api_download(box_url, dest, token, requests)
    print("[box] No --box-token given; attempting anonymous public-share scrape.")
    print("[box] If this fails, download the folder once via the browser and use --local.")
    try:
        return _box_anon_download(box_url, dest, requests)
    except Exception as e:
        print(f"[box] Anonymous download failed: {e}")
        print("[box] Fall back to: open the Box link, click Download, unzip, then "
              "re-run with --local <unzipped_dir>.")
        return []


def _box_api_download(box_url, dest, token, requests):
    H = {"Authorization": f"Bearer {token}", "BoxApi": f"shared_link={box_url}"}
    root = requests.get("https://api.box.com/2.0/shared_items", headers=H, timeout=30).json()
    if root.get("type") != "folder":
        raise RuntimeError("Shared link is not a folder")
    paths = []

    def walk(folder_id):
        offset = 0
        while True:
            r = requests.get(f"https://api.box.com/2.0/folders/{folder_id}/items",
                             headers=H, params={"limit": 1000, "offset": offset,
                                                "fields": "id,name,type"}, timeout=30).json()
            for it in r.get("entries", []):
                if it["type"] == "folder":
                    walk(it["id"])
                elif it["type"] == "file" and re.search(r"kinematics.*\.bag$", it["name"], re.I):
                    out = os.path.join(dest, it["name"])
                    if not os.path.exists(out):
                        print(f"[box] downloading {it['name']}")
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

    walk(root["id"])
    return paths


def _box_anon_download(box_url, dest, requests):
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    r = s.get(box_url, timeout=30, allow_redirects=True)
    shared_name = box_url.rstrip("/").split("/")[-1]
    host = r.url.split("/")[2]                       # e.g. uofi.app.box.com
    m = re.search(r'"requestToken":"([^"]+)"', r.text) or re.search(r"requestToken\s*[:=]\s*'([^']+)'", r.text)
    if not m:
        raise RuntimeError("could not find requestToken on share page")
    rt = m.group(1)
    api = f"https://{host}/app-api/enduserapp/shared-folder"
    hdr = {"Request-Token": rt, "X-Request-Token": rt,
           "X-Box-Client-Name": "enduserapp", "Content-Type": "application/json"}
    paths = []

    def walk(folder_id):
        body = {"shared_name": shared_name, "folder_id": folder_id, "offset": 0, "limit": 1000}
        j = s.post(api, json=body, headers=hdr, timeout=30).json()
        for it in j.get("items", j.get("entries", [])):
            tid = str(it.get("id") or it.get("typedID", "")).split("_")[-1]
            name = it.get("name", "")
            if it.get("type") == "folder":
                walk(tid)
            elif re.search(r"kinematics.*\.bag$", name, re.I):
                out = os.path.join(dest, name)
                if not os.path.exists(out):
                    print(f"[box] downloading {name}")
                    url = (f"https://{host}/index.php?rm=box_download_shared_file"
                           f"&shared_name={shared_name}&file_id=f_{tid}")
                    with s.get(url, stream=True, timeout=600) as resp:
                        resp.raise_for_status()
                        with open(out, "wb") as f:
                            for chunk in resp.iter_content(1 << 20):
                                f.write(chunk)
                paths.append(out)

    walk("0")
    return paths


# =============================================================== driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local", help="dir containing the kinematics bags (searched recursively)")
    src.add_argument("--box-url", nargs="?", const=BOX_URL_DEFAULT,
                     help="Box shared-folder URL (default: the CRCD raw link)")
    ap.add_argument("--box-token", help="Box developer token (reliable API download)")
    ap.add_argument("--dest", default="./crcd_bags", help="where to save downloaded bags")
    ap.add_argument("--arms", default="PSM1,PSM2")
    ap.add_argument("--deadband", type=float, default=0.10, help="noise gate, degrees/sample")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--out", default="crcd_angular_path_summary.csv")
    args = ap.parse_args()

    if args.local:
        bags = sorted(glob.glob(os.path.join(args.local, "**", "*kinematics*.bag"), recursive=True))
        if not bags:  # fall back to any .bag
            bags = sorted(glob.glob(os.path.join(args.local, "**", "*.bag"), recursive=True))
    else:
        bags = sorted(download_box(args.box_url, args.dest, args.box_token))

    if not bags:
        sys.exit("No kinematics bags found. See the DOWNLOAD section in this file's header.")

    print(f"Found {len(bags)} bag(s). Processing arms={args.arms} "
          f"deadband={args.deadband} downsample={args.downsample}x\n")

    rows = []
    arms = [a.strip() for a in args.arms.split(",")]
    for bag in bags:
        proc = proc_name_from_path(bag)
        for arm in arms:
            try:
                d = read_arm(bag, arm)
            except Exception as e:
                print(f"  ! {proc}/{arm}: {e}")
                continue
            ds = args.downsample
            wp = d["wrist_pitch"][::ds]; wy = d["wrist_yaw"][::ds]
            tp = d["tip_pitch"][::ds]; ty = d["tip_yaw"][::ds]
            row = {
                "procedure": proc, "arm": arm,
                "duration_s": round(d["dur"], 1),
                "n_js": d["n_js"], "n_cp": d["n_cp"],
                "wrist_pitch_deg": round(total_variation_deg(wp, args.deadband), 1),
                "wrist_yaw_deg":   round(total_variation_deg(wy, args.deadband), 1),
                "tip_pitch_deg":   round(total_variation_deg(tp, args.deadband), 1),
                "tip_yaw_deg":     round(total_variation_deg(ty, args.deadband), 1),
            }
            row["wrist_pitch+yaw_deg"] = round(row["wrist_pitch_deg"] + row["wrist_yaw_deg"], 1)
            row["tip_pitch+yaw_deg"] = round(row["tip_pitch_deg"] + row["tip_yaw_deg"], 1)
            rows.append(row)
            print(f"  {proc:>6} {arm}: dur {row['duration_s']:6.1f}s | "
                  f"wrist P/Y {row['wrist_pitch_deg']:8.0f}/{row['wrist_yaw_deg']:8.0f} deg | "
                  f"tip P/Y {row['tip_pitch_deg']:8.0f}/{row['tip_yaw_deg']:8.0f} deg")

    if not rows:
        sys.exit("No data extracted from any bag.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} rows)\n")

    # quick aggregate view
    pd.set_option("display.width", 140)
    print("Per-arm means across procedures (degrees):")
    print(df.groupby("arm")[["wrist_pitch_deg", "wrist_yaw_deg",
                             "tip_pitch_deg", "tip_yaw_deg",
                             "duration_s"]].mean().round(0).to_string())


if __name__ == "__main__":
    main()

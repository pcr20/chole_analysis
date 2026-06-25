#!/usr/bin/env python3
"""
maude_robotic_pull.py  (v2 - full-dataset pull)
===============================================
Pulls robotic surgical instrument adverse-event reports from FDA MAUDE via the
openFDA device/event API. v2 retrieves the COMPLETE record set over the window
(not just a capped harm subset), plus the same aggregate breakdowns as before.

Designed to be run locally, then upload the output directory back to Claude.

KEY CONSTRAINTS HANDLED
-----------------------
* 25,000-record skip ceiling per search  -> records are pulled in MONTHLY chunks
  (each month here is <~5k records, well under the ceiling). If any month ever
  exceeds 25k it auto-falls back to day-by-day pagination.
* 120,000 requests/day + 240 requests/min (keyed)  -> a global request counter,
  a --max-requests budget guard, and a min-interval throttle keep you inside both.
  (A full single-product pull is only a few hundred requests, so the daily cap is
  not normally binding; the guard mainly protects multi-product / repeat runs.)
* Resumability -> completed months are recorded in records_manifest.json. Re-run
  with the same --outdir to continue where a stopped/interrupted run left off.
* Output size -> full records stream to records.jsonl.gz (gzip) by default.

USAGE
-----
  pip install requests
  export OPENFDA_API_KEY=xxxx           # https://open.fda.gov/apis/authentication/
  python maude_robotic_pull.py                      # full pull, NAY, 5 yrs
  python maude_robotic_pull.py --product-codes NAY OLO
  python maude_robotic_pull.py --discover           # list codes by manufacturer
  python maude_robotic_pull.py --no-gzip            # plain .jsonl output
  python maude_robotic_pull.py --aggregates-only    # skip the full record pull
  python maude_robotic_pull.py --max-requests 50000 # stop early, resume tomorrow

CAVEATS FOR ANALYSIS
--------------------
MAUDE is passive surveillance: under-reporting, duplicate/supplemental MDRs,
unverified causality. Counts are reports, not validated events. event_type
"Injury"/"Death" is the harm signal; "Malfunction" alone is not. Bucketing is on
date_received (most complete); use --date-field date_of_event to switch.
"""

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from calendar import monthrange
from collections import defaultdict
from datetime import date, timedelta

try:
    import requests
except ImportError:
    sys.exit("This script needs 'requests'.  Install it with:  pip install requests")

BASE = "https://api.fda.gov/device/event.json"
DEFAULT_PRODUCT_CODES = ["NAY"]  # System, Surgical, Computer Controlled Instrument
DISCOVER_MANUFACTURERS = ["intuitive surgical", "medtronic", "cmr surgical",
                          "asensus", "stryker", "johnson"]
SKIP_CEILING = 25000          # openFDA max skip
PAGE = 1000                   # openFDA max limit per call

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "maude-robotic-pull/2.0"})

# --- global request governance ------------------------------------------------
REQ_COUNT = 0
REQ_BUDGET = 115000           # set from --max-requests
MIN_INTERVAL = 0.26           # seconds between calls (~230/min < 240 cap)
_last_call = [0.0]


def budget_left():
    return REQ_BUDGET - REQ_COUNT


# --------------------------------------------------------------------------- #
# Low-level API
# --------------------------------------------------------------------------- #
def fda_get(params, api_key=None, max_retries=5):
    """Throttled, retrying GET. Increments the global request counter."""
    global REQ_COUNT
    if api_key:
        params = {**params, "api_key": api_key}
    # throttle
    gap = time.time() - _last_call[0]
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    for attempt in range(max_retries):
        try:
            r = SESSION.get(BASE, params=params, timeout=60)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  ! network error ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        finally:
            _last_call[0] = time.time()
        REQ_COUNT += 1
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return {"meta": {"results": {"total": 0}}, "results": []}
        if r.status_code in (429, 403):
            if r.status_code == 403 and not api_key:
                print("  ! 403 without an API key — anonymous budget exhausted. "
                      "Get a free key at https://open.fda.gov/apis/authentication/ "
                      "and re-run with --api-key.", file=sys.stderr)
                return None
            wait = 5 * (attempt + 1)
            print(f"  ! HTTP {r.status_code}; backing off {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        print(f"  ! HTTP {r.status_code} for {params}\n    {r.text[:300]}", file=sys.stderr)
        return None
    print("  ! exhausted retries", file=sys.stderr)
    return None


def total_for(search, api_key=None):
    data = fda_get({"search": search, "limit": 1}, api_key)
    if not data:
        return None
    return data.get("meta", {}).get("results", {}).get("total", 0)


def count_field(search, field, api_key=None, limit=1000):
    data = fda_get({"search": search, "count": field, "limit": limit}, api_key)
    if not data or "results" not in data:
        return []
    return data["results"]


def paginate(search, api_key=None):
    """Yield every record for `search`, up to the skip ceiling."""
    skip = 0
    while skip < SKIP_CEILING:
        want = min(PAGE, SKIP_CEILING - skip)
        data = fda_get({"search": search, "limit": want, "skip": skip}, api_key)
        if not data:
            return
        results = data.get("results", [])
        if not results:
            return
        for rec in results:
            yield rec
        skip += len(results)
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        if skip >= min(total, SKIP_CEILING):
            return


# --------------------------------------------------------------------------- #
# Search helpers / record flattening
# --------------------------------------------------------------------------- #
def pc_clause(product_codes):
    if len(product_codes) == 1:
        return f"device.device_report_product_code:{product_codes[0]}"
    joined = " ".join(product_codes)
    return f"device.device_report_product_code:({joined})"


def with_year(base, year, date_field):
    return f"{base} AND {date_field}:[{year}0101 TO {year}1231]"


def narrative_text(rec):
    out = []
    for t in rec.get("mdr_text", []) or []:
        txt = (t.get("text") or "").strip()
        if txt:
            out.append(f"[{t.get('text_type_code','')}] {txt}")
    return "\n".join(out)


def flatten(rec):
    dev = (rec.get("device") or [{}])[0]
    outcomes = []
    for p in rec.get("patient") or []:
        o = p.get("sequence_number_outcome")
        if isinstance(o, list):
            outcomes.extend(o)
        elif o:
            outcomes.append(o)
    return {
        "mdr_report_key": rec.get("mdr_report_key"),
        "report_number": rec.get("report_number"),
        "date_received": rec.get("date_received"),
        "date_of_event": rec.get("date_of_event"),
        "event_type": rec.get("event_type"),
        "product_code": dev.get("device_report_product_code"),
        "brand_name": dev.get("brand_name"),
        "generic_name": dev.get("generic_name"),
        "manufacturer": dev.get("manufacturer_d_name"),
        "product_problems": rec.get("product_problems"),
        "patient_outcomes": outcomes,
        "narrative": narrative_text(rec),
    }


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def month_chunks(years):
    """Yield (start_yyyymmdd, end_yyyymmdd, label) per month up to today."""
    today = date.today()
    for y in years:
        for m in range(1, 13):
            start = date(y, m, 1)
            if start > today:
                return
            last = monthrange(y, m)[1]
            end = date(y, m, last)
            yield (start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), f"{y}-{m:02d}")


def day_chunks(start_yyyymmdd, end_yyyymmdd):
    s = date(int(start_yyyymmdd[:4]), int(start_yyyymmdd[4:6]), int(start_yyyymmdd[6:8]))
    e = date(int(end_yyyymmdd[:4]), int(end_yyyymmdd[4:6]), int(end_yyyymmdd[6:8]))
    d = s
    while d <= e:
        ds = d.strftime("%Y%m%d")
        yield (ds, ds)
        d += timedelta(days=1)


def load_manifest(path):
    if os.path.exists(path):
        try:
            return set(json.load(open(path)))
        except Exception:
            return set()
    return set()


def save_manifest(path, done):
    json.dump(sorted(done), open(path, "w"), indent=0)


# --------------------------------------------------------------------------- #
# Aggregates (cheap count queries)
# --------------------------------------------------------------------------- #
def write_aggregates(base, df, years, outdir, api_key):
    print("[aggregates] yearly totals + event-type split ...")
    by_year, et_by_year, et_seen = [], [], set()
    for y in years:
        s = with_year(base, y, df)
        total = total_for(s, api_key)
        by_year.append({"year": y, "total": total})
        ev = {r["term"]: r["count"] for r in count_field(s, "event_type.exact", api_key, 20)}
        et_seen.update(ev); ev["year"] = y; et_by_year.append(ev)
        print(f"   {y}: total={total:,}" if isinstance(total, int) else f"   {y}: total=?")
    event_types = sorted(et_seen)
    with open(os.path.join(outdir, "summary_by_year.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "total"]); w.writeheader(); w.writerows(by_year)
    with open(os.path.join(outdir, "event_type_by_year.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year"] + event_types, restval=0, extrasaction="ignore")
        w.writeheader(); w.writerows(et_by_year)

    print("[aggregates] device-problem codes (failure modes) ...")
    win = f"{base} AND {df}:[{years[0]}0101 TO {years[-1]}1231]"
    overall = count_field(win, "product_problems.exact", api_key)
    with open(os.path.join(outdir, "device_problems_overall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term", "count"]); w.writeheader()
        w.writerows({"term": r["term"], "count": r["count"]} for r in overall)
    prob_by_year = defaultdict(dict)
    for y in years:
        for r in count_field(with_year(base, y, df), "product_problems.exact", api_key, 50):
            prob_by_year[r["term"]][y] = r["count"]
    with open(os.path.join(outdir, "device_problems_by_year.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["problem"] + years)
        for term, yd in sorted(prob_by_year.items(), key=lambda kv: -sum(kv[1].values())):
            w.writerow([term] + [yd.get(y, 0) for y in years])

    print("[aggregates] patient-outcome codes (harm taxonomy) ...")
    oc = count_field(win, "patient.sequence_number_outcome.exact", api_key)
    with open(os.path.join(outdir, "patient_outcomes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term", "count"]); w.writeheader()
        w.writerows({"term": r["term"], "count": r["count"]} for r in oc)


# --------------------------------------------------------------------------- #
# Full record pull (streamed, chunked, resumable, budgeted)
# --------------------------------------------------------------------------- #
def pull_full(base, df, years, outdir, api_key, gzip_out, max_records):
    recs_path = os.path.join(outdir, "records.jsonl.gz" if gzip_out else "records.jsonl")
    manifest_path = os.path.join(outdir, "records_manifest.json")
    done = load_manifest(manifest_path)
    opener = gzip.open if gzip_out else open
    mode = "at" if done else "wt"
    if done:
        print(f"[records] resuming — {len(done)} months already complete")
    fout = opener(recs_path, mode, encoding="utf-8")

    written = 0
    stopped_for_budget = False
    try:
        for start, end, label in month_chunks(years):
            if label in done:
                continue
            search = f"{base} AND {df}:[{start} TO {end}]"
            total = total_for(search, api_key)
            if total is None:
                print(f"   {label}: query failed — stopping (resume later)")
                stopped_for_budget = True
                break
            if total == 0:
                done.add(label); save_manifest(manifest_path, done); continue

            est = 1 + math.ceil(total / PAGE)
            if est > budget_left():
                print(f"   {label}: need ~{est} requests, only {budget_left()} left in "
                      f"budget — stopping. Re-run to resume.")
                stopped_for_budget = True
                break

            n = 0
            if total <= SKIP_CEILING:
                for rec in paginate(search, api_key):
                    fout.write(json.dumps(flatten(rec), ensure_ascii=False) + "\n"); n += 1
                    if max_records and written + n >= max_records:
                        break
            else:  # rare: split the month into days
                print(f"   {label}: {total:,} > skip ceiling, splitting by day")
                for ds, de in day_chunks(start, end):
                    daysearch = f"{base} AND {df}:[{ds} TO {de}]"
                    for rec in paginate(daysearch, api_key):
                        fout.write(json.dumps(flatten(rec), ensure_ascii=False) + "\n"); n += 1
                    if max_records and written + n >= max_records:
                        break

            written += n
            done.add(label); save_manifest(manifest_path, done)
            fout.flush()
            print(f"   {label}: wrote {n:,}  (cumulative {written:,}; requests used {REQ_COUNT:,})")
            if max_records and written >= max_records:
                print(f"   reached --max-records {max_records:,}, stopping"); break
    finally:
        fout.close()

    complete = (len(done) >= len(list(month_chunks(years)))) and not stopped_for_budget
    return recs_path, written, complete


# --------------------------------------------------------------------------- #
def discover(api_key):
    print("Product codes by manufacturer (robotic-relevant):\n")
    for mfr in DISCOVER_MANUFACTURERS:
        rows = count_field(f'device.manufacturer_d_name:"{mfr}"',
                           "device.device_report_product_code.exact", api_key, 25)
        print(f"  {mfr}:")
        for r in rows[:15]:
            print(f"    {r['term']:6}  {r['count']:>8,}")
        if not rows:
            print("    (none)")
        print()


def main():
    global REQ_BUDGET, MIN_INTERVAL
    ap = argparse.ArgumentParser(description="Full MAUDE robotic-surgery pull from openFDA.")
    ap.add_argument("--product-codes", nargs="+", default=DEFAULT_PRODUCT_CODES)
    ap.add_argument("--years", type=int, default=5, help="Full calendar years back (default 5).")
    ap.add_argument("--date-field", default="date_received",
                    choices=["date_received", "date_of_event"])
    ap.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY"))
    ap.add_argument("--max-requests", type=int, default=115000,
                    help="Daily request budget guard (default 115000; cap is 120000).")
    ap.add_argument("--min-interval", type=float, default=0.26,
                    help="Seconds between API calls (default 0.26 ~ 230/min).")
    ap.add_argument("--max-records", type=int, default=0,
                    help="Optional cap on records pulled (0 = unlimited / full set).")
    ap.add_argument("--no-gzip", action="store_true", help="Write plain .jsonl (default gzip).")
    ap.add_argument("--aggregates-only", action="store_true", help="Skip the full record pull.")
    ap.add_argument("--outdir", default="maude_output")
    ap.add_argument("--discover", action="store_true")
    args = ap.parse_args()

    REQ_BUDGET = args.max_requests
    MIN_INTERVAL = args.min_interval

    if args.discover:
        discover(args.api_key); return

    os.makedirs(args.outdir, exist_ok=True)
    today = date.today()
    years = list(range(today.year - args.years, today.year + 1))
    base = pc_clause(args.product_codes)

    print("openFDA MAUDE full pull")
    print(f"  product codes : {', '.join(args.product_codes)}")
    print(f"  window        : {years[0]}-{years[-1]}  (by {args.date_field})")
    print(f"  api key       : {'yes' if args.api_key else 'NO'}")
    print(f"  req budget     : {REQ_BUDGET:,}/day  | throttle {MIN_INTERVAL}s/call")
    if not args.api_key:
        print("\n  !! No API key — heavier queries will 403. Get one (30s) at "
              "https://open.fda.gov/apis/authentication/\n")
    print()

    write_aggregates(base, args.date_field, years, args.outdir, args.api_key)

    recs_path, written, complete = (None, 0, True)
    if not args.aggregates_only:
        print("\n[records] full record pull (monthly chunks) ...")
        recs_path, written, complete = pull_full(
            base, args.date_field, years, args.outdir,
            args.api_key, gzip_out=not args.no_gzip, max_records=args.max_records)

    meta = {
        "generated": today.isoformat(),
        "product_codes": args.product_codes,
        "date_field": args.date_field,
        "years": years,
        "base_search": base,
        "records_file": os.path.basename(recs_path) if recs_path else None,
        "records_written_this_run": written,
        "full_pull_complete": complete,
        "requests_used": REQ_COUNT,
        "caveats": "MAUDE = passive surveillance; counts are reports not events; "
                   "duplicates/supplements exist; causality unverified.",
    }
    json.dump(meta, open(os.path.join(args.outdir, "run_meta.json"), "w"), indent=2)

    print(f"\nDone. Requests used: {REQ_COUNT:,}/{REQ_BUDGET:,}")
    print(f"Output dir: {os.path.abspath(args.outdir)}")
    if recs_path:
        status = "COMPLETE" if complete else "PARTIAL — re-run same --outdir to resume"
        print(f"  records: {os.path.basename(recs_path)}  ({written:,} this run, {status})")
    print("  + summary_by_year / event_type_by_year / device_problems_* / "
          "patient_outcomes CSVs")
    print("\nUpload the output dir back to chat for analysis. If PARTIAL, just "
          "re-run tomorrow with the same --outdir; completed months are skipped.")


if __name__ == "__main__":
    main()

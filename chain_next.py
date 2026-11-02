#!/usr/bin/env python3
"""
chain_next.py  -  keeps the paper sim running through the market day on GitHub Actions.

GitHub's cron scheduler is best-effort and can go quiet for hours, so each workflow run also hands
off to the next one itself: wait until the next quarter-hour slot (:07, :22, :37, :52 UTC, matching
the cron), confirm no other run is already queued, then trigger the workflow by workflow_dispatch.
One run anywhere in the day keeps the chain going until 4:45pm New York, when it stops on its own.

Used only by .github/workflows/sim.yml. Needs GH_TOKEN, GITHUB_REPOSITORY and GITHUB_RUN_ID.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SLOTS = (7, 22, 37, 52)          # minutes past the hour, UTC
LAST_PASS = dt.time(16, 45)      # New York; the sim's settle window ends here
LEAD_SECONDS = 25                # dispatch a little before the slot so the next run starts on it


def gh(*args):
    out = subprocess.run(["gh", "api", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or out.stdout.strip())
    return out.stdout


def still_trading(now_ny):
    return now_ny.weekday() < 5 and now_ny.time() <= LAST_PASS


def seconds_to_next_slot(now_utc):
    for m in SLOTS:
        cand = now_utc.replace(minute=m, second=0, microsecond=0)
        if cand > now_utc + dt.timedelta(seconds=LEAD_SECONDS + 5):
            return (cand - now_utc).total_seconds() - LEAD_SECONDS
    cand = (now_utc + dt.timedelta(hours=1)).replace(minute=SLOTS[0], second=0, microsecond=0)
    return (cand - now_utc).total_seconds() - LEAD_SECONDS


def other_runs_pending(repo, workflow, own_id):
    n = 0
    for status in ("queued", "in_progress", "waiting", "pending"):
        data = json.loads(gh(f"repos/{repo}/actions/workflows/{workflow}/runs?status={status}&per_page=20"))
        n += sum(1 for r in data.get("workflow_runs", []) if str(r.get("id")) != str(own_id))
    return n


def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    own_id = os.environ.get("GITHUB_RUN_ID", "")
    workflow = os.environ.get("WORKFLOW_FILE", "sim.yml")
    if not repo or not os.environ.get("GH_TOKEN"):
        print("chain: missing GITHUB_REPOSITORY or GH_TOKEN; not chaining.")
        return
    now_ny = dt.datetime.now(NY)
    if not still_trading(now_ny):
        print(f"chain: {now_ny.strftime('%a %H:%M')} New York, past the last pass; chain ends for today.")
        return
    wait = max(0, seconds_to_next_slot(dt.datetime.now(dt.timezone.utc)))
    print(f"chain: sleeping {int(wait)}s until the next slot.")
    time.sleep(wait)
    now_ny = dt.datetime.now(NY)
    if not still_trading(now_ny):
        print("chain: past the last pass after waiting; chain ends for today.")
        return
    try:
        pending = other_runs_pending(repo, workflow, own_id)
    except Exception as ex:
        print(f"chain: could not list runs ({ex}); dispatching anyway.")
        pending = 0
    if pending:
        print(f"chain: {pending} other run(s) already queued or running; they will carry the chain.")
        return
    gh("-X", "POST", f"repos/{repo}/actions/workflows/{workflow}/dispatches", "-f", "ref=main")
    print("chain: next pass dispatched.")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"chain: failed ({ex}). The cron schedule or a manual run will pick it up.")
        sys.exit(0)

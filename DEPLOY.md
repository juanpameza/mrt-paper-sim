# Running the paper sim live

Two free ways. Both use the same files; pick one.

Files that belong in the folder (or repo root):

```
index.html          the tracker page (rename mrt-tracker.html to this)
mrt_scan.py         fills the Scan tab (your version that writes scan.js is fine)
mrt_sim.py          the rules engine
requirements.txt
```

The sim writes `sim_state.json` (the record), `sim.js` and `sim.json` (what the page reads),
`sim_trades.csv`, and `.earnings_cache.json`. Keep them all together.

---

## A. On your own computer

Two terminal windows in that folder:

```
python mrt_sim.py --loop
```

Keeps running. During market hours (9:30 to 4:00 New York, so 6:30am to 1pm Pacific) it does a pass
every 15 minutes: marks every open spread, exits on the rules, looks for new entries. Outside those
hours it idles. Ctrl-C stops it. Add `--every 5` for faster passes, though Yahoo's option quotes are
15 minutes delayed so there is little to gain.

```
python -m http.server 8000
```

Then open http://localhost:8000/ in a browser. Served this way, the page re-reads `sim.json` every
minute, so the Trades tab updates by itself. Opening the file directly also works but won't refresh
until you reload.

Keep your 6pm scan task; it only feeds the Scan tab. The sim computes its own indicators on every pass.

The computer has to be awake during market hours. On a Mac, `caffeinate -i python mrt_sim.py --loop`
keeps it from sleeping while the loop runs.

---

## B. On GitHub, nothing running at home (free)

1. Create a **public** repository (public gets unlimited Actions minutes and free Pages). Paper-trade
   data is all that lives in it.
2. Add the files above plus the workflow at `.github/workflows/sim.yml` and commit.
3. Settings → Actions → General → Workflow permissions: **Read and write permissions**. Save.
4. Settings → Pages → Build and deployment → Source: **Deploy from a branch**, branch `main`, folder `/ (root)`. Save.
   The tracker is live at `https://<your-user>.github.io/<repo>/` a minute later.
5. Actions tab → "Paper sim" → **Run workflow** once by hand to confirm it goes green. After that it
   fires on its own every 15 minutes during market hours, commits the results, and Pages redeploys.

The page on Pages is served over https, so it refreshes from `sim.json` on its own every minute.

Things to know:

- GitHub's scheduler is best-effort. Passes can run a few minutes late and the occasional one is
  skipped under load. For a daily-bar method that's fine.
- Yahoo sometimes rate-limits cloud IPs. If runs start failing with "Too Many Requests", the sim
  state is safe in the repo; switch to option A for a while, or add `--no-earnings` to the scan step
  to cut the request count.
- Each pass commits a few files, so the repo history will be busy. That is the point: `sim_state.json`
  at any commit is the full record as of that moment.
- To change a rule, edit the RULES block in `mrt_sim.py` and commit; the next pass uses it. Trades
  already open keep the exits they were opened with.
- To start over: delete `sim_state.json`, `sim.js`, `sim.json`, `sim_trades.csv` and commit.

---

## What "live" changes versus the once-a-day run

- Entries happen on the pass where the setup first passes every rule, at that moment's option mid
  (plus slippage), instead of at the close.
- Stops and targets are checked every pass, so a 50% stop fires mid-session instead of on a close
  mark that may have gapped through it. Expect realized losses closer to −50% than the daily run's −65%.
- A name trades at most once per day. Exits are checked every pass regardless.
- The Rule log records a skip only when a name's reason changes that day, so it stays readable.

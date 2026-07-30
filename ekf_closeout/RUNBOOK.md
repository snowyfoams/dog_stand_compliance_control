# EKF Close-Out Runbook

Two short hardware sessions that finish the EKF validation without a motion
capture room: a **distance walk** (x/y accuracy) and a **height cross-check**
(z scale).  Everything else in this folder is offline tooling that was already
run and passed — see *Status* at the bottom.

Nothing in the existing codebase was modified.  This folder layers on top of
`crawl_hw2.0/walk1_hw.py` and `state_estimator/` by subclassing.

```
V=/home/robot01/Documents/can_motor_control/.venv/bin/python   # hardware (python-can)
cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/ekf_closeout
```
Offline replay/tests run under the system `python3` (numpy only).

---

## Session A — distance walk (x/y drift)

**What it answers:** how far does the EKF's horizontal position estimate
diverge from reality over a real walk?  The paper's expectation is ~10 % of
distance travelled; this measures ours.

**Setup**
1. Chalk or tape a straight reference line on the floor, ~1 m long.
2. Dog on the floor, aligned with the line, IMU and CAN up as usual.
3. Pick a fixed reference point on the trunk (a corner, a bolt) and note how
   you will drop a plumb line from it to the floor.

**Run** — 160 mm forward, 4 cycles:
```
sudo chrt -f 50 $V walk_cmd_hw.py --goto 0.16 0 --raw-log walk_A1.npz
```
The command layer prints the plan (heading, cycles, step length, quantization
residual, estimated duration) and refuses before arming if the heading is
infeasible.  It writes `walk_A1.cmd.json` next to the log so the replay knows
what was commanded.

4. Press ENTER at `WAIT_CROUCH` to stand.  **When the dog reaches HOLD4 and
   settles, mark the floor** under the trunk reference point — that is the
   measurement origin, and it is the same instant the replay uses as its
   displacement origin.
5. The walk runs hands-free from there (`--auto` is added automatically).
   Watch the usual lines; X stops, P parks.
6. At the final HOLD4, **let it stand still for ~5 s** (the replay averages
   the last second), then mark the floor again and press X.
7. Tape-measure from mark to mark: **X** = travel along the line, **Y** =
   lateral offset (left positive, matching the dog's +y).

**Check**
```
python3 replay_full.py walk_A1.npz --gait --measured "158,4"
```
**Pass:** `EKF vs tape` error ≤ max(20 mm, 10 % of measured distance).
Repeat 2–3 times; the run-to-run spread matters as much as any single number.

**Expected from the existing log** (`walk_0729_1748.npz`, 1 cycle): the EKF
reports +40.0 mm forward / −2.7 mm lateral against a commanded 40 mm — so a
clean 160 mm run should land within a few mm if the tape agrees.

---

## Session B — height cross-check (z scale)

**What it answers:** the static logs show z *drift* is small (5.9 mm / 46 s),
but nothing has yet checked the z *scale* — that a commanded 40 mm rise is a
real 40 mm.

**Run** — stand only, no steps, at three heights:
```
$V walk_cmd_hw.py --goto 0 0 --height 0.155 --raw-log h_155.npz
$V walk_cmd_hw.py --goto 0 0 --height 0.175 --raw-log h_175.npz
$V walk_cmd_hw.py --goto 0 0 --height 0.190 --raw-log h_190.npz
```
Zero displacement means **STAND ONLY**: crouch → stand → HOLD4, no stepping.
(The stepping validator is skipped for these — H = 0.19 stands fine but cannot
take a 40 mm step.)

For each run:
1. With the dog settled in the **crouch**, tape-measure the trunk reference
   height above the floor.
2. Press ENTER, let it stand, wait ~5 s at HOLD4, measure the same point again.
3. Press X.  Δh = stand − crouch, in mm.

**Check**
```
python3 replay_full.py h_175.npz --measured-dz 62
```
**Pass:** |EKF Δz − tape Δz| ≤ 10 mm on each, and the three Δz values increase
monotonically with the commanded height.

---

## Reading the report

`replay_full.py` prints two blocks:

* **close-out gates** — the corrected ones: health over the post-re-anchor
  window, static |v| on the 95th percentile, displacement/height comparisons.
* **standard hw_replay gates** — the existing attitude / innovation / per-leg
  offset / bias / touchdown checks, unchanged, so the record stays comparable
  with earlier runs.

Useful flags: `--health-detail` (dumps every unhealthy cluster with the term
that failed), `--static` (for a still log), `--gait` (walking log).

---

## Honest limits (read before commanding anything)

* **Forward only, in practice.** The validated heading envelope is a narrow
  forward cone — 0…+2° at 40 mm steps, 0…+4° at 30 mm.  Anything else fails
  as *FL outside soft limits* (the FL abduction wall that `walk2_fl_hw.py`
  exists for).  Dropping the pre-lift cap to 20 mm only widens it to about
  −4…+6°.  The planner accepts any (dx, dy) and refuses infeasible ones
  before the bus is armed; `walk_cmd_hw.py --self-test` prints the fan.
* **No turning.** There is no yaw gait; the trunk never rotates, so "world
  frame" is simply the startup frame for the whole run.
* **Distance is quantized** to one cycle (~40 mm) and capped at 20 cycles
  (800 mm).
* **Speed is not commandable.** A cycle is ~75 s and gate-dominated, so mean
  |v| is ~0.5 mm/s.  `--vx/--vy --duration` exists to express a distance in
  velocity terms; the heading is exact, the speed is best-effort.
* **Keep runs short** (≲3 min): EKF yaw drifts, and the world frame drifts
  with it.
* During the contacts-off STAND rise the filter dead-reckons and slides
  150–550 mm; that window is excluded from every gate here by construction.
  Don't stretch the rise — σ_v reaches ~93 % of its health threshold in it.

---

## Offline verification (already run, all green)

```
python3 test_closeout.py --logs                    # 28 checks, all PASS
python3 replay_full.py ../../walk_0729_1748.npz --gait --health-detail   # PASS
python3 replay_full.py ../vmc/stand.npz --static                         # PASS
$V walk_cmd_hw.py --self-test                      # decomposition + fan + dry run
```

Unchanged upstream suites, re-run to prove this folder changes nothing:
`state_estimator/test_estimator.py`, `state_estimator/test_hw_replay_gait.py`,
`crawl_hw2.0/walk1_hw.py --self-test` — all still PASS.

## Status

| Item | State |
|---|---|
| min_eig health artefact | diagnosed + fixed in `estimator_health.py` (subclass) |
| Health gate window | fixed in `replay_full.py` |
| Static \|v\| 95th-pct hygiene | fixed in `replay_full.py` |
| x/y drift quantified | **needs Session A** |
| z scale verified | **needs Session B** |
| In-tree health fix (`dog5_state_estimator.outputs`) | deferred — roadmap item |

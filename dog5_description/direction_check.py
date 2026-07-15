#!/usr/bin/env python3
"""calibrate12.py -- observe / verify / hardware set-zero for ALL 12 motors (CAN id 1..12).

The 12-motor sibling of catersian_compliance_demo/calibrate.py. Built on the clean
motorbus.py library (MotorBus round-robin), which is the correct primitive for a
shared multi-motor bus -- the arm's calibrate.py uses the blocking LKMotor, which
mixes replies across ids and does not scale to 12.

Every mode commands all 12 motors ZERO torque throughout (fully back-drivable).
NOTHING is ever commanded to move; you pose the rig by hand. Start the tool FIRST,
then switch motor power on when prompted -- streaming keep-alives before power
means the input-signal-lost latch never sets in the first place. And if a motor
DID boot latched (0x80), it is cleared OVER CAN (0x9B, escalating to the t7
0x9B->0x88 ladder) -- so no power cycle is ever needed to get going.

The 0x9C encoder reads the MOTOR output angle directly: motoroutput =
encoder * 360/65535, NO 10:1 gear divide (same as the 2-DOF arm; this is the
OPPOSITE of the HIL dog code, which divides by GEAR=10 -- reconcile before mixing).

  1) OBSERVE  (direction / which-id-is-which, safe):
       python calibrate12.py --observe
     Live table of every motor's encoder, motoroutput, and delta since start. Turn
     ONE joint by hand and watch which id moves and which way it counts. Ctrl-C to
     stop.

  2) VERIFY  (confirm the zero, safe):
       python calibrate12.py --verify
     Pose the rig at its intended zero and read all 12. After a --set-zero +
     POWER-CYCLE, each motor at the zero pose should read motoroutput ~ 0.

  3) SET ZERO  (HARDWARE 0x19 write to ALL 12 at the current pose):
       python calibrate12.py --set-zero          # shows readings, then confirms
       python calibrate12.py --set-zero --yes    # skip the confirmation

     Sends set_zero_position (0x19) to every motor -> the current encoder position
     becomes each motor's zero. !! The vendor warns 0x19 affects the DRIVER
     LIFETIME -- use it rarely. It takes effect only after a POWER-CYCLE; then run
     --verify to confirm all 12 read ~0.

  4) CROSS VERIFY  (live hardware-to-MuJoCo mirror, safe):
       python calibrate12.py --cross-verify
       python calibrate12.py --cross-verify --motors 1 4

     Motors continuously receive ZERO torque. Rotate a physical joint by hand;
     the program reads motoroutput, calculates joint_angle = direction *
     motoroutput, and writes that angle into MuJoCo so the model leg mirrors it.

Exit status is non-zero if any motor never replied (or, with --tol, any motor is
off zero), so the tool is scriptable.
"""
import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

from dog5_hardware_map import MOTOR_JOINT_MAP
from motorbus import MotorBus, ENCODER_GAIN

IDS = sorted(MOTOR_JOINT_MAP)       # plain CAN ids 1..12, dir=+1 for all
DEFAULT_RATE_HZ = 250.0             # per-motor keep-alive period 1/250 = 4 ms << 10 ms

# Hardware zero was written at the MJCF zero pose, so no software offset and no
# 10:1 divide are applied. The confirmed ID/joint map and unchanged directions
# come from dog5_hardware_map.py:
#     joint_angle_deg = direction * motoroutput_deg

# The 0x9C encoder reading maps directly to the MOTOR output angle on this rig:
# motoroutput_deg = encoder * ENCODER_GAIN, with ENCODER_GAIN = 360/65535. No 10:1
# reducer divide -- same convention as the 2-DOF arm (demo_config ENCODER_GAIN
# comment: "encoder_value * gain -> OUTPUT-shaft degrees"). NOTE: this is the
# OPPOSITE of the HIL dog code (hil_joint_map GEAR=10), which divides the same
# register by 10; reconcile the two before mixing calibrations.


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------
def fold180(deg):
    """Fold a degree value into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def circ_mean_deg(samples):
    """Wrap-safe mean of degree samples (guards the 0/360 boundary)."""
    s = sum(math.sin(math.radians(d)) for d in samples)
    c = sum(math.cos(math.radians(d)) for d in samples)
    return math.degrees(math.atan2(s, c)) % 360.0


# ---------------------------------------------------------------------------
# Bus bring-up -- back-drivable, recover a 0x80 latch OVER CAN (no power cycle)
# ---------------------------------------------------------------------------
CLEAR_TRIES_BEFORE_ARM = 8      # 0x9B-only attempts on a latched motor before the ladder
LADDER_PERIOD = 8               # then re-apply the 0x9B->0x88 ladder every N attempts


def bring_up_limp(mb, rate_hz=DEFAULT_RATE_HZ, status_period_s=0.1,
                  timeout_s=None, quiet=False):
    """Bring every motor to 'replying, input-lost latch clear, encoder live' while
    every motor is commanded ZERO torque throughout (back-drivable -- you pose the
    rig by hand). Returns True once every motor is ready; False on timeout.

    Streams zero-torque keep-alives (0xA1 iq=0) round-robin -- each motor
    re-commanded every 1/rate s (4 ms at 250 Hz, inside the ~10 ms input-lost
    window) -- and substitutes a round-robin 0x9A status read so the error byte
    becomes visible.

    A motor that boots with the input-signal-lost latch set (status-1 error bit
    0x80 -- e.g. it was powered before the stream reached it, or it re-latched
    during a blocking print/prompt) is recovered OVER CAN, so NO power cycle is
    needed (this is the t7 result). Recovery is clear-first: send 0x9B (clear
    error flags) in that motor's slots, which leaves it STOPPED / limp. If a motor
    is not ready after CLEAR_TRIES_BEFORE_ARM attempts -- still latched, OR cleared
    but not yet reporting its encoder -- escalate to the t7-proven clear+run ladder
    (0x9B -> 0x88) via mb.recover(), re-applied every LADDER_PERIOD attempts so one
    lost 0x88 cannot strand it. A motor left RUN-enabled here is still only ever
    commanded iq=0 (back-drivable).

    Readiness is re-evaluated every visit (never latched sticky-True), so a motor
    that re-latches after being marked ready drops back into recovery instead of
    slipping through. Pass quiet=True to suppress the prints (used as an in-place
    re-arm before the set-zero write)."""
    ids = mb.ids
    n = len(ids)
    slot = mb.slot(rate_hz)
    visits_per_status = max(2, int(status_period_s * rate_hz))
    if not quiet:
        print(f"[bring-up] zero-torque keep-alive to {ids} at {rate_hz:.0f} Hz/motor"
              " -- motors held at ZERO torque (back-drivable).")
        print("           Switch 24 V ON now if it is off. A 0x80-latched motor is "
              "cleared over CAN -- no power cycle (Ctrl-C aborts).")
    probe_in = {mid: 1 + (k % visits_per_status) for k, mid in enumerate(ids)}
    attempts = {mid: 0 for mid in ids}        # consecutive not-ready recovery attempts
    armed = {mid: False for mid in ids}       # escalated to the 0x9B->0x88 ladder
    ready = {mid: False for mid in ids}
    deadline = time.perf_counter() + slot
    t0 = time.perf_counter()
    last_show = 0.0
    i = 0
    while not all(ready.values()):
        mid = ids[i % n]
        i += 1
        mb.poll()
        rec = mb.rec(mid)
        latched = rec.error is not None and (rec.error & 0x80)
        err_clear = rec.error is not None and not (rec.error & 0x80)
        have_enc = rec.encoder is not None
        ready[mid] = err_clear and have_enc               # non-sticky (re-checked)
        # Recover a motor that is latched, OR that cleared but is not yet reporting
        # its encoder (a stopped motor that only fully replies once run). A motor
        # merely awaiting its first status read (error unknown) just gets probed.
        needs_rec = latched or (err_clear and not have_enc)
        if needs_rec:
            a = attempts[mid]
            attempts[mid] += 1
            if a >= CLEAR_TRIES_BEFORE_ARM and (a - CLEAR_TRIES_BEFORE_ARM) % LADDER_PERIOD == 0:
                mb.recover(mid)               # 0x9B -> 0xA1 -> 0x88 (then iq=0)
                armed[mid] = True
            elif latched:
                mb.clear(mid)                 # 0x9B -- clears flags, stays stopped
            else:
                mb.keepalive(mid)             # cleared: fetch the encoder reply
        else:
            attempts[mid] = 0
            if probe_in[mid] <= 0:
                mb.status1_req(mid)           # 0x9A -- also resets the input timer
                probe_in[mid] = visits_per_status
            else:
                mb.keepalive(mid)             # 0xA1 iq=0 -- fed, no motion
                probe_in[mid] -= 1

        now = time.perf_counter()
        if not quiet and now - last_show > 0.5:
            pend = [m for m in ids if not ready[m]]
            latched_ids = [m for m in ids if (mb.rec(m).error or 0) & 0x80]
            msg = f"  waiting: {pend}"
            if latched_ids:
                armed_ids = [m for m in latched_ids if armed[m]]
                msg += f"  clearing 0x80 over CAN: {latched_ids}"
                if armed_ids:
                    msg += f" (escalated to clear+run: {armed_ids})"
            print(msg + "   ", end="\r")
            last_show = now
        if timeout_s is not None and now - t0 > timeout_s:
            stuck = [m for m in ids if not ready[m]]
            if not quiet:
                print(f"\n[bring-up] timeout; not ready: {stuck}. A motor stuck here "
                      "has a fault that\n           will not clear over CAN (e.g. "
                      "over-current/stall) -- power-cycle just it.")
            return False
        mb.pace(deadline)
        deadline += slot

    mb.poll()
    if not quiet:
        print(f"\n[bring-up] all {n} motors replying, no input-lost latch -- "
              "zero torque, back-drivable.")
    return True


# ---------------------------------------------------------------------------
# Sample every motor's encoder over a short settle window (limp stream running)
# ---------------------------------------------------------------------------
def sample_positions(mb, dur_s=0.4, rate_hz=DEFAULT_RATE_HZ):
    """Hold the zero-torque stream for dur_s and average each motor's encoder.
    Returns {mid: {"joint_deg": float|None, "enc": int|None, "err": int|None,
    "n": int}} where joint_deg = encoder * ENCODER_GAIN (already the joint angle,
    no gear divide)."""
    ids = mb.ids
    n = len(ids)
    slot = mb.slot(rate_hz)
    deadline = time.perf_counter() + slot
    t_end = time.perf_counter() + dur_s
    samples = {mid: [] for mid in ids}
    i = 0
    while time.perf_counter() < t_end:
        mid = ids[i % n]
        i += 1
        mb.poll()
        mb.keepalive(mid)
        if i % n == 0:                     # one snapshot per full round
            degs = mb.encoders_deg()
            for m in ids:
                if mb.rec(m).encoder is not None:
                    samples[m].append(degs[m])
        mb.pace(deadline)
        deadline += slot
    mb.poll()

    out = {}
    for mid in ids:
        rec = mb.rec(mid)
        if samples[mid]:
            out[mid] = {"joint_deg": circ_mean_deg(samples[mid]),
                        "enc": rec.encoder, "err": rec.error, "n": len(samples[mid])}
        else:
            out[mid] = {"joint_deg": None, "enc": None, "err": rec.error, "n": 0}
    return out


# ---------------------------------------------------------------------------
# OBSERVE -- live table so you can hand-check id + rotation direction
# ---------------------------------------------------------------------------
def observe(mb, rate_hz=DEFAULT_RATE_HZ):
    ids = mb.ids
    n = len(ids)
    slot = mb.slot(rate_hz)
    print("\nOBSERVE -- motors at ZERO torque (back-drivable). Turn ONE joint by "
          "hand and\nwatch which id moves and which way d_motoroutput counts. "
          "+ d_motoroutput\nand + motoroutput define this motor's 'positive'. "
          "Ctrl-C to stop.\n")
    base = {}                              # joint_deg at first good reading per id
    attempts = {m: 0 for m in ids}         # 0x80 self-heal escalation
    probe_in = {m: 1 + (k % 25) for k, m in enumerate(ids)}  # periodic status read
    printed = 0
    deadline = time.perf_counter() + slot
    last_show = 0.0
    i = 0
    try:
        while True:
            mid = ids[i % n]
            i += 1
            mb.poll()
            rec = mb.rec(mid)
            if rec.error is not None and (rec.error & 0x80):
                # Self-heal a latch over CAN (e.g. a blocking stdout write stalled
                # the stream past ~10 ms) -- same clear-first/escalate ladder as
                # bring-up, so a motor never freezes mid-observation.
                a = attempts[mid]
                attempts[mid] += 1
                if a >= CLEAR_TRIES_BEFORE_ARM and (a - CLEAR_TRIES_BEFORE_ARM) % LADDER_PERIOD == 0:
                    mb.recover(mid)
                else:
                    mb.clear(mid)
            elif probe_in[mid] <= 0:
                mb.status1_req(mid)        # 0x9A -- keep the err column live
                probe_in[mid] = 25
                attempts[mid] = 0
            else:
                mb.keepalive(mid)          # 0xA1 iq=0 -- fed, no motion
                probe_in[mid] -= 1
                attempts[mid] = 0
            if i % n == 0:
                degs = mb.encoders_deg()
                for m in ids:
                    if m not in base and mb.rec(m).encoder is not None:
                        base[m] = degs[m]

            now = time.perf_counter()
            if now - last_show > 0.4:
                degs = mb.encoders_deg()
                lines = [f"  {'id':>3}  {'encoder':>7}  {'motoroutput':>11}  "
                         f"{'d_motoroutput':>13}  {'dps':>7}  err"]
                for m in ids:
                    rec = mb.rec(m)
                    if rec.encoder is None:
                        lines.append(f"  {m:>3}  {'--':>7}  {'--':>11}  {'--':>13}  "
                                     f"{'--':>7}  no-reply")
                        continue
                    jd = degs[m]
                    dj = fold180(jd - base[m]) if m in base else 0.0
                    dps = mb.speeds_dps()[m]
                    err = rec.error or 0
                    etag = "ok" if not err else f"0x{err:02x}"
                    lines.append(f"  {m:>3}  {rec.encoder:>7}  {fold180(jd):>+11.2f}  "
                                 f"{dj:>+13.2f}  {dps:>+7.1f}  {etag}")
                if printed:
                    sys.stdout.write(f"\033[{printed}A")
                sys.stdout.write("\n".join(s.ljust(64) for s in lines) + "\n")
                sys.stdout.flush()
                printed = len(lines)
                last_show = now

            mb.pace(deadline)
            deadline += slot
    except KeyboardInterrupt:
        print("\n\nStopped. A motor whose d_motoroutput / motoroutput moved OPPOSITE "
              "to the "
              "way you\nturned it has dir=-1 for your convention (record it for your "
              "joint map).")


# ---------------------------------------------------------------------------
# CROSS VERIFY -- MuJoCo target angle vs live, calibrated motor output
# ---------------------------------------------------------------------------
class KeyPoller:
    """Non-blocking terminal keys while the zero-torque stream keeps running."""

    def __init__(self):
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("--cross-verify requires an interactive terminal")
        self._select = select
        self._termios = termios
        self.fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self.fd)
        self._closed = False
        tty.setcbreak(self.fd)

    def get(self):
        readable, _, _ = self._select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if readable else ""

    def close(self):
        if not self._closed:
            self._termios.tcsetattr(
                self.fd, self._termios.TCSADRAIN, self._old
            )
            self._closed = True


def print_cross_verify_map(motor_ids):
    """Print the fixed relationship before opening the time-critical CAN stream."""
    print("\nCROSS VERIFY mapping (hardware zero already calibrated):")
    print("  joint_angle_deg = direction * motoroutput_deg")
    print(f"  {'id':>3}  {'MuJoCo joint':<14}  {'dir':>4}  relationship")
    for mid in motor_ids:
        joint_name, direction = MOTOR_JOINT_MAP[mid]
        sign = "+" if direction > 0 else "-"
        print(f"  {mid:>3}  {joint_name:<14}  {direction:>+4d}  q = {sign}motoroutput")


def cross_verify(mb, motor_ids, rate_hz=DEFAULT_RATE_HZ, no_viewer=False):
    """Mirror hand-moved hardware joints into MuJoCo while commanding iq=0.

    For each selected motor, the calibrated raw output angle is folded to signed
    degrees, multiplied by its confirmed direction, and assigned directly to the
    corresponding MuJoCo qpos. Physics is never stepped and no motor receives a
    nonzero torque, speed, or position command.
    """
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(_HERE / "dog5.xml"))
    data = mujoco.MjData(model)
    qadr = {}
    for mid in motor_ids:
        joint_name, _ = MOTOR_JOINT_MAP[mid]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"MuJoCo joint not found: {joint_name}")
        qadr[mid] = model.jnt_qposadr[jid]

    viewer_context = (nullcontext(None) if no_viewer else
                      mujoco.viewer.launch_passive(model, data))
    key = None
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.08)
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.sync()

        # Opening the GUI can pause Python long enough to trip the ~10 ms motor
        # input watchdog. Re-read status and restore the zero-torque stream before
        # accepting any measurement.
        for mid in mb.ids:
            mb.rec(mid).error = None
        if not bring_up_limp(
            mb, rate_hz=rate_hz, timeout_s=5.0, quiet=True
        ):
            raise RuntimeError("could not restore zero-torque stream after viewer start")

        key = KeyPoller()
        try:
            ids = mb.ids
            n = len(ids)
            attempts = {mid: 0 for mid in ids}
            probe_in = {mid: 1 + (index % 25) for index, mid in enumerate(ids)}
            slot = mb.slot(rate_hz)
            deadline = time.perf_counter() + slot
            loop_index = 0
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            last_show = 0.0
            last_sync = 0.0
            printed = 0
            readings = {}

            print(
                "\nLIVE MIRROR -- ZERO TORQUE ONLY. Rotate joints by hand; "
                "MuJoCo follows.\nPress X in this terminal, close the viewer, "
                "or press Ctrl-C to stop.\n"
            )

            while True:
                mid = ids[loop_index % n]
                loop_index += 1
                mb.poll()
                rec = mb.rec(mid)
                latched = rec.error is not None and (rec.error & 0x80)
                if latched:
                    attempt = attempts[mid]
                    attempts[mid] += 1
                    if (attempt >= CLEAR_TRIES_BEFORE_ARM and
                            (attempt - CLEAR_TRIES_BEFORE_ARM) % LADDER_PERIOD == 0):
                        mb.recover(mid, settle_s=0.0, verify=False)
                    else:
                        mb.clear(mid)
                elif probe_in[mid] <= 0:
                    mb.status1_req(mid)
                    probe_in[mid] = 25
                    attempts[mid] = 0
                else:
                    # This is the only recurring motor command in live-mirror
                    # mode: 0xA1 with iq=0. It cannot drive the motor.
                    mb.keepalive(mid)
                    probe_in[mid] -= 1
                    attempts[mid] = 0

                now = time.perf_counter()
                if now - last_sync >= 1.0 / 20.0:
                    for selected_mid in motor_ids:
                        selected = mb.rec(selected_mid)
                        if selected.encoder is None:
                            readings[selected_mid] = None
                            continue
                        joint_name, direction = MOTOR_JOINT_MAP[selected_mid]
                        motoroutput = fold180(selected.encoder * ENCODER_GAIN)
                        joint_deg = fold180(direction * motoroutput)
                        data.qpos[qadr[selected_mid]] = math.radians(joint_deg)
                        readings[selected_mid] = (motoroutput, joint_deg)

                    # Forward kinematics only: no mj_step(), actuator control, or
                    # simulation dynamics. qpos comes directly from hand motion.
                    mujoco.mj_forward(model, data)
                    if viewer is not None:
                        if not viewer.is_running():
                            break
                        viewer.sync()
                    last_sync = now

                if now - last_show >= 1.0 / 5.0:
                    lines = [
                        f"  {'id':>3}  {'MuJoCo joint':<14}  {'dir':>4}  "
                        f"{'motoroutput':>11}  {'joint_deg':>10}  "
                        f"{'MuJoCo_q':>9}  status"
                    ]
                    for selected_mid in motor_ids:
                        joint_name, direction = MOTOR_JOINT_MAP[selected_mid]
                        selected = mb.rec(selected_mid)
                        reading = readings.get(selected_mid)
                        if reading is None:
                            lines.append(
                                f"  {selected_mid:>3}  {joint_name:<14}  "
                                f"{direction:>+4d}  {'--':>11}  {'--':>10}  "
                                f"{'--':>9}  no-reply"
                            )
                            continue
                        motoroutput, joint_deg = reading
                        mujoco_q_deg = math.degrees(data.qpos[qadr[selected_mid]])
                        status = ("LATCHED" if (selected.error or 0) & 0x80
                                  else "ok")
                        lines.append(
                            f"  {selected_mid:>3}  {joint_name:<14}  "
                            f"{direction:>+4d}  {motoroutput:>+11.2f}  "
                            f"{joint_deg:>+10.2f}  {mujoco_q_deg:>+9.2f}  {status}"
                        )
                    if printed:
                        sys.stdout.write(f"\033[{printed}A")
                    sys.stdout.write("\n".join(line.ljust(82) for line in lines) + "\n")
                    sys.stdout.flush()
                    printed = len(lines)
                    last_show = now

                if key.get() in ("x", "X"):
                    break

                mb.pace(deadline)
                deadline += slot
        finally:
            if key is not None:
                key.close()

    print("\nLIVE MIRROR stopped; all motors remained at zero commanded torque.")


# ---------------------------------------------------------------------------
# VERIFY -- read all 12 at the (posed) zero and report
# ---------------------------------------------------------------------------
def verify(mb, settle_s, rate_hz=DEFAULT_RATE_HZ, tol=None):
    pos = sample_positions(mb, dur_s=settle_s, rate_hz=rate_hz)
    print("\nVERIFY -- current motor output of every motor (fold to +/-180 deg).")
    print("At the ZERO pose after a --set-zero + power-cycle, motoroutput should be "
          "~ 0.\n")
    hdr = f"  {'id':>3}  {'encoder':>7}  {'motoroutput':>11}  err"
    if tol is not None:
        hdr += f"   verdict (|motoroutput| < {tol:g})"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n_missing = n_off = 0
    for mid in mb.ids:
        p = pos[mid]
        if p["joint_deg"] is None:
            print(f"  {mid:>3}  {'--':>7}  {'--':>11}  NO REPLY")
            n_missing += 1
            continue
        jd = fold180(p["joint_deg"])
        err = p["err"] or 0
        etag = "ok" if not err else f"0x{err:02x}"
        line = f"  {mid:>3}  {p['enc']:>7}  {jd:>+11.2f}  {etag}"
        if tol is not None:
            off = abs(jd) > tol
            n_off += off
            line += f"   {'OFF ZERO <--' if off else 'ok'}"
        print(line)
    print("  " + "-" * (len(hdr) - 2))
    if n_missing:
        print(f"  {n_missing} motor(s) NO REPLY -- check bus/power; result incomplete.")
    elif tol is not None and n_off:
        print(f"  {n_off} motor(s) OFF ZERO by more than {tol:g} deg. Re-pose to the"
              " zero\n  pose and re-check, or run --set-zero if the pose is right"
              " and the zero is stale.")
    else:
        print("  All motors replied." + (" PASS -- every motor within tolerance."
              if tol is not None else " (pass --tol to flag motors off zero.)"))
    return n_missing, n_off


# ---------------------------------------------------------------------------
# SET ZERO -- hardware 0x19 write to all 12 at the current pose
# ---------------------------------------------------------------------------
def set_zero(mb, settle_s, rate_hz=DEFAULT_RATE_HZ, assume_yes=False):
    pos = sample_positions(mb, dur_s=settle_s, rate_hz=rate_hz)
    missing = [mid for mid in mb.ids if pos[mid]["joint_deg"] is None]
    if missing:
        raise SystemExit(f"[set-zero] no reply from ids {missing} -- refusing to "
                         "zero a partial set. Fix the bus/power and retry.")

    print("\nAbout to hardware-zero (0x19) these motors at their CURRENT pose:\n")
    print(f"  {'id':>3}  {'encoder':>7}  {'motoroutput':>11}")
    for mid in mb.ids:
        jd = fold180(pos[mid]["joint_deg"])
        print(f"  {mid:>3}  {pos[mid]['enc']:>7}  {jd:>+11.2f}")
    print("\n!! 0x19 writes each driver's encoder offset. The vendor warns this "
          "affects\n   the DRIVER LIFETIME -- use it rarely. It takes effect only "
          "after a\n   POWER-CYCLE.")

    if not assume_yes:
        # This blocking prompt stops the keep-alive stream, so every motor will set
        # its input-lost latch (0x80) while it waits. We do NOT write 0x19 to a
        # latched motor -- a latched driver still answers config reads, so its
        # echoed offset would look "ok" while the lifetime-affecting write went to a
        # motor in a fault state. Instead we re-arm robustly below before writing.
        ans = input("\nType 'yes' to write the zero to ALL 12 motors > ").strip().lower()
        if ans != "yes":
            print("Aborted -- no zero written.")
            return

    # Re-arm before the write: the prompt (and the table print above) paused the
    # stream, so motors may be latched. A single 0x9B is not guaranteed to clear
    # 0x80, so re-run the full escalating recovery (clear -> clear+run ladder).
    # Invalidate the now-stale error telemetry first so it is re-read fresh rather
    # than trusting the pre-prompt value. Refuse to write if any motor stays
    # latched -- better no zero than a zero written to a faulted driver.
    for mid in mb.ids:
        mb.rec(mid).error = None
    if not bring_up_limp(mb, rate_hz=rate_hz, timeout_s=5.0, quiet=True):
        stuck = [m for m in mb.ids if (mb.rec(m).error or 0x80) & 0x80]
        raise SystemExit(f"[set-zero] could not clear ids {stuck} over CAN before "
                         "writing -- aborting so NO zero is written to a latched "
                         "motor. Power-cycle those and retry.")
    offsets = mb.set_zero_all(rate_hz=rate_hz)

    print("\n  id   new encoder_offset   ack")
    n_ack = 0
    for mid in mb.ids:
        off = offsets[mid]
        if off is None:
            print(f"  {mid:>3}   {'--':>16}   NO ACK <--")
        else:
            print(f"  {mid:>3}   {off:>16}   ok")
            n_ack += 1
    print(f"\n{n_ack}/12 motors acknowledged the zero write.")
    if n_ack < 12:
        print("Some motors did not ACK -- re-run --set-zero (a fresh keep-alive "
              "stream).")
    print("\nDONE. POWER-CYCLE all motors (the 0x19 zero only takes effect on "
          "restart),\nthen run `python calibrate12.py --verify` to confirm every "
          "motor reads\nmotoroutput ~ 0 at the zero pose.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--observe", action="store_true",
                   help="live zero-torque read of all 12 to hand-check id + direction")
    g.add_argument("--verify", action="store_true",
                   help="read all 12 at the zero pose and report (motoroutput ~ 0)")
    g.add_argument("--set-zero", action="store_true",
                   help="HARDWARE 0x19 set-zero of all 12 at the current pose")
    g.add_argument("--cross-verify", action="store_true",
                   help="zero-torque live hardware-to-MuJoCo joint mirror")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt for --set-zero")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ,
                    help=f"keep-alive stream rate per motor, Hz (default {DEFAULT_RATE_HZ:g})")
    ap.add_argument("--settle", type=float, default=0.4,
                    help="encoder averaging window for verify/set-zero, s (default 0.4)")
    ap.add_argument("--tol", type=float, default=None,
                    help="verify: flag motors whose |motoroutput| exceeds this (deg)")
    ap.add_argument("--motors", type=int, nargs="+", default=None,
                    help="cross-verify selected CAN IDs in MuJoCo (default: all)")
    ap.add_argument("--no-viewer", action="store_true",
                    help="cross-verify numeric live angles without the MuJoCo GUI")
    ap.add_argument("--timeout", type=float, default=None,
                    help="bring-up timeout in s (default: wait indefinitely)")
    args = ap.parse_args()

    cross_motor_ids = args.motors if args.motors is not None else IDS
    if args.cross_verify:
        invalid = [mid for mid in cross_motor_ids if mid not in MOTOR_JOINT_MAP]
        if invalid:
            ap.error(f"--motors contains invalid CAN IDs: {invalid}")
        if len(set(cross_motor_ids)) != len(cross_motor_ids):
            ap.error("--motors must not contain duplicate CAN IDs")
        if (not args.no_viewer and os.name != "nt" and
                not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
            ap.error(
                "MuJoCo viewer requested, but this shell has no DISPLAY or "
                "WAYLAND_DISPLAY. Reconnect with X11 forwarding (`ssh -Y "
                "robot01@monroe`) or pass --no-viewer for the numeric live mirror."
            )
        print_cross_verify_map(cross_motor_ids)

    print("Motors come up at ZERO torque (back-drivable) -- nothing is commanded to "
          "move; a 0x80-latched motor is cleared over CAN (no power cycle).")
    with MotorBus(IDS) as mb:
        if not bring_up_limp(mb, rate_hz=args.rate, timeout_s=args.timeout):
            raise SystemExit("[calibrate12] bring-up failed -- see above.")
        if args.observe:
            observe(mb, rate_hz=args.rate)
        elif args.verify:
            n_missing, n_off = verify(mb, args.settle,
                                      rate_hz=args.rate, tol=args.tol)
            if n_missing or n_off:
                raise SystemExit(1)
        elif args.set_zero:
            set_zero(mb, args.settle, rate_hz=args.rate, assume_yes=args.yes)
        else:
            cross_verify(
                mb,
                cross_motor_ids,
                rate_hz=args.rate,
                no_viewer=args.no_viewer,
            )
    # MotorBus.__exit__ -> stop_all() (0x81 + iq=0) -> shutdown(): motors stopped.
    print("[calibrate12] bus closed. The input-lost latch sets ~10 ms after the "
          "stream ends,\nbut the next run clears it over CAN -- no power cycle needed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl-C during bring-up or the set-zero confirm: MotorBus.__exit__ already
        # stopped the motors and closed the bus on the way out.
        print("\nAborted.")

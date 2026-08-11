"""Closed-loop MuJoCo harness: DOG5 trots in place under whole-body VMC with the
proprioceptive EKF in the feedback path.  This is CONTROL_ROADMAP Gate D1.

    Gate D1: trot in MuJoCo on the hardware-fidelity model
             (EKF in loop, not sim truth).

Pipeline:
    MuJoCo IMU + encoders + schedule contacts
        -> EKF predict/update  ->  outputs() (r, v, C, w_hat)
        -> trot schedule (contacts, EKF-placed swing targets)
        -> VMC body wrench -> foot-force distribution -> joint torques
        -> data.ctrl

Sequence: stand-up (`stand_dog5.Dog5Stand`) -> EKF init on a static hold ->
VMC four-foot hold -> TROT for N cycles -> hold.

Rates (this is the part `vmc_mujoco.py` does NOT do)
----------------------------------------------------
`vmc_mujoco.py` runs the whole loop once per physics tick -- 500 Hz -- which no
hardware ever will.  Here the loop is decimated to the real thing:

    physics    500 Hz  (dog5.xml timestep 0.002)
    VMC/control 250 Hz  one full 12-motor CAN sweep (stand3_dog5.CONTROL_HZ);
                        12 x 250 Hz x ~260 us = 78 % of a 1 Mbit/s bus
    EKF         100 Hz  ekf_runtime.DEFAULT_CONTROL_UPDATE_HZ, IMU batch-averaged
                        between ticks exactly as the hardware worker does

Measured at the default operating point (14 cycles, |v| error max):

    EKF  50 Hz -> 58.2 mm/s        control 125 Hz -> 26.7 mm/s
    EKF 100 Hz -> 26.9 mm/s        control 250 Hz -> 26.9 mm/s
    EKF 250 Hz -> 23.0 mm/s        control 500 Hz -> 26.4 mm/s
    EKF 500 Hz -> 25.6 mm/s

So the hardware's existing 100 Hz worker is sufficient with margin, and the trot
is not control-rate limited at 250 Hz.  `--ekf-per-sample` (integrate every
buffered IMU sample instead of collapsing the batch to its mean) makes no
measurable difference: 26.8 vs 26.9 mm/s.  See TROT_D1.md -- an earlier reading
that appeared to make 100 Hz the binding limit was taken while the wrench
controller was torque-saturated, and did not survive fixing that.

Torque is held between control ticks.  `--control-hz` / `--ekf-hz` reproduce any
other rate for A/B.  This matters far more for a trot than for a stand: a
0.164 s swing is only ~41 control ticks.

Privileged state discipline
---------------------------
Everything on the control path comes from the IMU sensors, the joint encoders
and the EKF.  MuJoCo's ground truth is read ONLY by `Oracle`, which grades the
run afterwards and never steers it -- the same contract `twostand_dog5.py` uses.

Run (mujoco lives in the project venv):
    .venv/bin/python trot_mujoco.py             # interactive viewer
    .venv/bin/python trot_mujoco.py --headless  # metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import mujoco

_DESC = Path(__file__).resolve().parent.parent / "dog5_description"
_EST = Path(__file__).resolve().parent.parent / "state_estimator"
for _p in (_DESC, _EST):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import dog5_vmc_core as vmc                                    # noqa: E402
import dog5_trot                                               # noqa: E402
from dog5_state_estimator import DOG5StateEstimator, quat_to_C  # noqa: E402
import stand_dog5                                              # noqa: E402

XML = str(_DESC / "dog5.xml")
LEGS = vmc.LEGS
PHYS_DT = 0.002                       # matches dog5.xml timestep

DEFAULT_CONTROL_HZ = 250.0            # one 12-motor CAN sweep
DEFAULT_EKF_HZ = 100.0                # ekf_runtime.DEFAULT_CONTROL_UPDATE_HZ

# phase durations (s)
T_STANDUP = stand_dog5.T_STAND + 1.0  # let Dog5Stand reach and settle standing
T_INIT = 0.5                          # static window to initialise the estimator
T_HOLD = 2.0                          # VMC four-foot hold before trotting
T_HOLD_END = 1.0
FALL_Z = 0.10                         # trunk height below this = fallen
FOOT_RADIUS_M = 0.020                 # dog5.xml foot sphere
SETTLE_CYCLES = 2                     # trot cycles excluded as hand-off transient


def wxyz_to_R(quat):
    """MuJoCo [w,x,y,z] (body->world) -> 3x3 rotation matrix."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


class TrotSim:
    """Thin MuJoCo accessor: joint state, IMU, ground truth, torque write."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.qadr, self.dadr, self.actadr = {}, {}, {}
        for leg in LEGS:
            names = [f"hip_abd_{leg}", f"hip_pitch_{leg}", f"knee_{leg}"]
            jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in names]
            self.qadr[leg] = np.array([model.jnt_qposadr[j] for j in jids])
            self.dadr[leg] = np.array([model.jnt_dofadr[j] for j in jids])
            self.actadr[leg] = np.array(
                [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   f"{n}_motor") for n in names])
        self._sadr = {}
        for name in ("imu_acc", "imu_gyro", "imu_quat", "trunk_pos",
                     "trunk_linvel"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._sadr[name] = (model.sensor_adr[sid], model.sensor_dim[sid])
        self.mass = float(model.body_mass.sum())

    def sensor(self, name):
        adr, dim = self._sadr[name]
        return self.d.sensordata[adr:adr + dim].copy()

    def joint_state(self):
        q = np.concatenate([self.d.qpos[self.qadr[leg]] for leg in LEGS])
        qd = np.concatenate([self.d.qvel[self.dadr[leg]] for leg in LEGS])
        return q, qd

    def alpha44(self):
        return np.stack([self.d.qpos[self.qadr[leg]] for leg in LEGS], axis=0)

    def imu(self):
        return self.sensor("imu_acc"), self.sensor("imu_gyro")

    def truth_rp(self):
        C = wxyz_to_R(self.sensor("imu_quat")).T     # I->B
        return vmc.attitude_error_rp(C)

    def trunk_z(self):
        return float(self.sensor("trunk_pos")[2])

    def trunk_xy(self):
        return self.sensor("trunk_pos")[:2].copy()

    def write_torque(self, tau12):
        for i, leg in enumerate(LEGS):
            self.d.ctrl[self.actadr[leg]] = tau12[3 * i:3 * i + 3]


class Oracle:
    """Privileged MuJoCo state -- grading ONLY, never on the control path."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.foot_gid = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in LEGS}
        self.foot_sid = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}") for leg in LEGS}
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    def foot_force(self, leg):
        """Total normal contact force on `leg`'s foot (N)."""
        gid = self.foot_gid[leg]
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if con.geom1 == gid or con.geom2 == gid:
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                total += abs(float(f6[0]))
        return total

    def foot_clearance(self, leg):
        """Height of the foot sphere's underside above the floor (m)."""
        return float(self.d.site_xpos[self.foot_sid[leg]][2]) - FOOT_RADIUS_M

    def tilt_deg(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def com_line_offset(self, stance):
        """Perpendicular distance from the true CoM to the stance support line.

        Only meaningful for a 2-foot stance -- this is `d` in the tip relation
        theta ~ 0.5*(m*g*d/I)*T_ss^2, so it is the quantity that decides whether
        a given single-support time is survivable.
        """
        if len(stance) != 2:
            return None
        a = self.d.site_xpos[self.foot_sid[stance[0]]][:2]
        b = self.d.site_xpos[self.foot_sid[stance[1]]][:2]
        com = np.asarray(self.d.subtree_com[self.trunk][:2])
        ab = b - a
        n = float(np.linalg.norm(ab))
        if n < 1e-9:
            return None
        return float(abs(np.cross(ab, com - a)) / n)


def _est_out(est, w_meas):
    out = est.outputs(last_w_meas=w_meas)     # w_hat = w_meas - gyro bias
    out["C"] = quat_to_C(out["q"])
    return out


class TrotPipeline:
    """stand -> hold -> trot state machine, one `step()` per PHYSICS tick."""

    def __init__(self, model, data, gains=None, cfg=None, params=None,
                 n_cycles=25, control_hz=DEFAULT_CONTROL_HZ,
                 ekf_hz=DEFAULT_EKF_HZ, placement=True,
                 batch_average=True):
        self.sim = TrotSim(model, data)
        self.oracle = Oracle(model, data)
        self.stand = stand_dog5.Dog5Stand(model)
        self.est = DOG5StateEstimator()
        self.gains = gains or vmc.VMCGains()
        self.cfg = cfg or vmc.VMCConfig()
        self.p = params or dog5_trot.TrotParams()
        self.placement = placement
        self.batch_average = batch_average
        if not placement:
            # Ablation: kill the closed-loop foothold term, keep everything else.
            self.p.k_place = 0.0
            self.p.place_clamp = 0.0

        self.n_ctrl = max(1, int(round(1.0 / (control_hz * PHYS_DT))))
        self.n_ekf = max(1, int(round(1.0 / (ekf_hz * PHYS_DT))))
        self.control_hz = 1.0 / (self.n_ctrl * PHYS_DT)
        self.ekf_hz = 1.0 / (self.n_ekf * PHYS_DT)

        self.t_trot = n_cycles * self.p.period
        self.total_t = (T_STANDUP + T_INIT + T_HOLD + self.t_trot + T_HOLD_END)

        # runtime state
        self.scheduler = None
        self._init_acc, self._init_gyro = [], []
        self._started = False
        self._phase = "HOLD"
        self._vmc_t0 = 0.0
        self._trot_t0 = None
        self._z_true0 = None
        self._xy_true0 = None
        self._tau = np.zeros(12)
        self._out = None
        self._frozen = None
        self._imu_batch = []
        self._k = 0
        # Contact flags as last published by the control sweep; the EKF reads
        # these, never its own. All-planted until the first sweep runs.
        self._contacts = np.ones(len(LEGS), dtype=bool)
        self._sched = {"contacts": self._contacts}
        self._diag = None

    # ---------------------------------------------------------------- helpers
    def _advance_phase(self, tv):
        """Monotonic HOLD -> TROT -> HOLD_END. Never revisits a phase.

        This must latch.  Deriving the phase from `scheduler.mode` instead lets
        the end-of-trot flip back to HOLD re-satisfy the HOLD->TROT condition on
        the very next tick, which restarts the gait and resets the gait clock --
        silently corrupting every windowed statistic downstream.
        """
        if self._phase == "HOLD" and tv >= T_HOLD:
            self._phase = "TROT"
            self._trot_t0 = tv
            self.scheduler.mode = "TROT"
        elif (self._phase == "TROT"
              and (tv - self._trot_t0) >= self.t_trot):
            self._phase = "HOLD_END"
            self.scheduler.mode = "HOLD"
        return self._phase

    def step(self, freeze_estimator=False, push=None):
        """Advance one physics tick (writes data.ctrl). Returns a record dict
        once the trot phase machine is live, else None. Does NOT call mj_step."""
        d = self.sim.d
        t = d.time

        # ---- stand up, then a static window to initialise the estimator ----
        if t < T_STANDUP:
            self.stand.control(d)
            return None
        if t < T_STANDUP + T_INIT:
            self.stand.control(d)
            a, g = self.sim.imu()
            self._init_acc.append(a)
            self._init_gyro.append(g)
            return None

        if not self._started:
            self.est.initialise(np.array(self._init_acc),
                                np.array(self._init_gyro),
                                self.sim.alpha44(), np.ones(4, dtype=bool))
            nominal = {leg: vmc.dog5_kinematics.foot_position(
                leg, self.sim.alpha44()[i]) for i, leg in enumerate(LEGS)}
            self.scheduler = dog5_trot.TrotScheduler(nominal, self.p,
                                                     mode="HOLD")
            self._z_true0 = self.sim.trunk_z()
            self._xy_true0 = self.sim.trunk_xy()
            self._vmc_t0 = t
            self._out = _est_out(self.est, self.sim.imu()[1])
            self._started = True

        tv = t - self._vmc_t0
        phase = self._advance_phase(tv)
        gt = 0.0 if self._trot_t0 is None else (tv - self._trot_t0)

        q, qd = self.sim.joint_state()
        f_meas, w_meas = self.sim.imu()
        self._imu_batch.append((f_meas, w_meas))

        foot_pos_body = [vmc.dog5_kinematics.foot_position(LEGS[i], q[3*i:3*i+3])
                         for i in range(len(LEGS))]

        # ---- EKF tick: batch-average the IMU exactly as ekf_runtime does ----
        # The filter consumes the contact flags the CONTROL loop last published,
        # never its own freshly-computed set.  That is the hardware contract:
        # the control sweep writes `shared.contacts`, the worker reads it
        # (ekf_runtime.py, and walk1_hw.py's per-sweep publish).  Recomputing the
        # schedule inside the filter would let the two disagree about which feet
        # are down, which is the one disagreement an estimator cannot survive.
        if self._k % self.n_ekf == 0 and self._imu_batch:
            if self.batch_average:
                # What `ekf_runtime.ekf_worker` does today: collapse the whole
                # buffered batch into one mean sample and take a single step.
                fb = np.mean([b[0] for b in self._imu_batch], axis=0)
                wb = np.mean([b[1] for b in self._imu_batch], axis=0)
                self.est.predict(fb, wb, len(self._imu_batch) * PHYS_DT,
                                 self._contacts)
            else:
                # Integrate every buffered IMU sample, but still pay for only one
                # measurement update per tick.  Averaging f and w across a 10 ms
                # batch is fine when omega barely changes -- it is not fine in a
                # trot, where the body rate reverses inside a single batch, and
                # the averaged step throws that away.  `predict` is cheap (no
                # solve); the 27x27 Joseph-form `update` is what costs, and it
                # stays at ekf_hz, so this buys accuracy at nearly no compute.
                for fb, wb in self._imu_batch:
                    self.est.predict(fb, wb, PHYS_DT, self._contacts)
            self.est.update(self.sim.alpha44(), self._contacts)
            self._imu_batch = []
            if freeze_estimator:
                if self._frozen is None:
                    self._frozen = _est_out(self.est, w_meas)
                self._out = self._frozen
            else:
                self._out = _est_out(self.est, w_meas)

        # ---- control tick: one full 12-motor CAN sweep ----
        if self._k % self.n_ctrl == 0:
            sched = self.scheduler.sched_state(gt, self._out, foot_pos_body)
            self._tau, self._diag = vmc.compute_vmc_torques(
                q, qd, self._out, sched, self.gains, self.cfg, self.sim.mass)
            self._sched = sched
            self._contacts = sched["contacts"]      # published for the EKF
        self.sim.write_torque(self._tau)
        self._k += 1

        if push is not None and push["t0"] <= tv < push["t0"] + push["dur"]:
            d.xfrc_applied[self.sim.trunk, :3] = push["force"]
        else:
            d.xfrc_applied[self.sim.trunk, :3] = 0.0

        contacts = self._sched["contacts"]
        stance = [LEGS[i] for i in range(len(LEGS)) if contacts[i]]
        return {
            "tv": tv, "gt": gt, "phase": phase,
            "z_est": float(self._out["r"][2]),
            "z_true": self.sim.trunk_z() - self._z_true0,
            "v_est": self._out["v"].copy(),
            "v_true": self.sim.sensor("trunk_linvel").copy(),
            "rp_est": vmc.attitude_error_rp(self._out["C"]),
            "rp_true": self.sim.truth_rp(),
            "healthy": bool(self._out["healthy"]),
            "fell": self.sim.trunk_z() < FALL_Z,
            "contacts": contacts.copy(),
            "xy_drift": self.sim.trunk_xy() - self._xy_true0,
            "tilt_true": self.oracle.tilt_deg(),
            "force": np.array([self.oracle.foot_force(l) for l in LEGS]),
            "clear": np.array([self.oracle.foot_clearance(l) for l in LEGS]),
            "tau_max": float(np.max(np.abs(self._tau))),
            "com_line_d": self.oracle.com_line_offset(stance),
        }


def run(n_cycles=25, params=None, gains=None, cfg=None, push=None,
        freeze_estimator=False, control_hz=DEFAULT_CONTROL_HZ,
        ekf_hz=DEFAULT_EKF_HZ, placement=True, com_bias=(0.0, 0.0),
        batch_average=True, quiet=True):
    """Headless run; returns metrics over the VMC phase."""
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if com_bias[0] or com_bias[1]:
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        model.body_ipos[trunk][0] += com_bias[0]
        model.body_ipos[trunk][1] += com_bias[1]

    pipe = TrotPipeline(model, data, gains=gains, cfg=cfg, params=params,
                        n_cycles=n_cycles, control_hz=control_hz,
                        ekf_hz=ekf_hz, placement=placement,
                        batch_average=batch_average)
    if not quiet:
        print(f"  control {pipe.control_hz:.0f} Hz  EKF {pipe.ekf_hz:.0f} Hz  "
              f"period {pipe.p.period:.3f}s  ds_frac {pipe.p.ds_frac:.2f}  "
              f"t_sw {pipe.p.t_sw:.3f}s  cycles {n_cycles}")

    hist, fell = [], False
    while data.time < pipe.total_t:
        rec = pipe.step(freeze_estimator=freeze_estimator, push=push)
        if rec is not None:
            hist.append(rec)
            fell = fell or rec["fell"]
            if fell:
                break
        mujoco.mj_step(model, data)

    return _summarise(hist, fell, pipe)


def _summarise(hist, fell, pipe):
    m = {"fell": fell, "n": len(hist), "final_z": pipe.sim.trunk_z(),
         "params": pipe.p, "n_cycles": pipe.t_trot / pipe.p.period,
         "control_hz": pipe.control_hz, "ekf_hz": pipe.ekf_hz}
    if not hist:
        return m
    trot_all = [r for r in hist if r["phase"] == "TROT"]
    m["n_trot_all"] = len(trot_all)
    if not trot_all:
        m["n_trot"] = 0
        return m
    # Exclude the hand-off transient.  Entering TROT from a four-foot hold is a
    # step change in the support pattern and the first cycle or two is a settling
    # response, not steady-state trotting.  The repo already draws this line the
    # same way elsewhere -- `replay_full.py` runs the health gate on the
    # post-re-anchor window only and excludes the rise/settle transient from the
    # displacement measurement.  Both windows are reported; nothing is hidden.
    t_settle = SETTLE_CYCLES * pipe.p.period
    trot = [r for r in trot_all if r["gt"] >= t_settle] or trot_all
    m["n_trot"] = len(trot)
    m["settle_cycles"] = SETTLE_CYCLES
    m["v_err_all"] = float(np.max(np.linalg.norm(
        np.array([r["v_est"] for r in trot_all])
        - np.array([r["v_true"] for r in trot_all]), axis=1)))
    m["tilt_all_max"] = float(np.max([r["tilt_true"] for r in trot_all]))

    z_est = np.array([r["z_est"] for r in trot])
    z_true = np.array([r["z_true"] for r in trot])
    v_est = np.array([r["v_est"] for r in trot])
    v_true = np.array([r["v_true"] for r in trot])
    rp_est = np.array([r["rp_est"] for r in trot])
    rp_true = np.array([r["rp_true"] for r in trot])

    m["z_err"] = np.abs(z_est - z_true)
    m["v_err"] = np.linalg.norm(v_est - v_true, axis=1)
    m["rp_err_deg"] = np.degrees(np.abs(rp_est - rp_true))
    m["healthy_frac"] = float(np.mean([r["healthy"] for r in trot]))
    m["tilt_true_deg"] = np.array([r["tilt_true"] for r in trot])
    m["tau_max"] = float(np.max([r["tau_max"] for r in trot]))
    m["gt"] = np.array([r["gt"] for r in trot])
    m["tv"] = np.array([r["tv"] for r in hist])

    # in-place drift: net and worst excursion of the true trunk xy
    drift = np.array([r["xy_drift"] for r in trot])
    m["drift"] = drift
    m["drift_net"] = float(np.linalg.norm(drift[-1] - drift[0]))
    m["drift_max"] = float(np.max(np.linalg.norm(drift - drift[0], axis=1)))
    # The steady-state question is not "how far did it end up" but "is it
    # walking away". Per-cycle rate answers that and is comparable across runs
    # of different length.
    n_cyc = max(1e-9, (m["gt"][-1] - m["gt"][0]) / pipe.p.period)
    m["drift_rate_mm_cycle"] = m["drift_net"] * 1e3 / n_cyc
    m["drift_total_net"] = float(np.linalg.norm(
        np.array(trot_all[-1]["xy_drift"]) - np.array(trot_all[0]["xy_drift"])))

    # per-leg swing quality: did every leg actually get off the ground?
    contacts = np.array([r["contacts"] for r in trot])
    force = np.array([r["force"] for r in trot])
    clear = np.array([r["clear"] for r in trot])
    m["clear_max"], m["force_at_apex"], m["swing_samples"] = {}, {}, {}
    for i, leg in enumerate(LEGS):
        air = ~contacts[:, i]
        m["swing_samples"][leg] = int(air.sum())
        if air.sum() == 0:
            m["clear_max"][leg] = 0.0
            m["force_at_apex"][leg] = float("inf")
            continue
        c = clear[air, i]
        j = int(np.argmax(c))
        m["clear_max"][leg] = float(c[j])
        m["force_at_apex"][leg] = float(force[air, i][j])

    # tilt growth across the run (divergence check): first vs last third
    n3 = max(1, len(trot) // 3)
    m["tilt_first"] = float(np.max(m["tilt_true_deg"][:n3]))
    m["tilt_last"] = float(np.max(m["tilt_true_deg"][-n3:]))

    d_line = [r["com_line_d"] for r in trot if r["com_line_d"] is not None]
    m["com_line_d_mean"] = float(np.mean(d_line)) if d_line else None
    m["com_line_d_max"] = float(np.max(d_line)) if d_line else None
    return m


def _print_metrics(m):
    print(f"fell={m['fell']}  final trunk z={m['final_z']:.3f}  "
          f"samples={m['n']} (trot {m.get('n_trot', 0)})")
    if not m.get("n_trot"):
        print("  no trot samples")
        return
    print(f"  control {m['control_hz']:.0f} Hz / EKF {m['ekf_hz']:.0f} Hz")
    print(f"  (settled window: first {m['settle_cycles']} cycles excluded as "
          f"hand-off transient; {m['n_trot']}/{m['n_trot_all']} samples)")
    print(f"  est vs truth: z_err max/mean = {m['z_err'].max()*1e3:.1f}/"
          f"{m['z_err'].mean()*1e3:.1f} mm")
    print(f"                |v|_err max/mean = {m['v_err'].max()*1e3:.1f}/"
          f"{m['v_err'].mean()*1e3:.1f} mm/s"
          f"   (incl. transient: {m['v_err_all']*1e3:.1f} mm/s)")
    print(f"                roll/pitch_err max = {m['rp_err_deg'].max():.2f} deg")
    print(f"  estimator healthy fraction = {m['healthy_frac']:.4f}")
    print(f"  drift rate (settled) = {m['drift_rate_mm_cycle']:.2f} mm/cycle")
    print(f"  trunk tilt (truth) max = {m['tilt_true_deg'].max():.2f} deg "
          f"(first third {m['tilt_first']:.2f}, last third {m['tilt_last']:.2f})")
    print(f"  in-place drift: net {m['drift_net']*1e3:.1f} mm, "
          f"max excursion {m['drift_max']*1e3:.1f} mm")
    print(f"  peak |tau| = {m['tau_max']:.2f} Nm")
    if m["com_line_d_mean"] is not None:
        print(f"  CoM offset from support line: mean "
              f"{m['com_line_d_mean']*1e3:.2f} mm, max "
              f"{m['com_line_d_max']*1e3:.2f} mm")
    print("  per-leg swing:")
    for leg in LEGS:
        print(f"    {leg}: max clearance {m['clear_max'][leg]*1e3:6.1f} mm, "
              f"force at apex {m['force_at_apex'][leg]:6.2f} N, "
              f"{m['swing_samples'][leg]} airborne samples")


def _live_loop(args):
    import mujoco.viewer
    import time
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    pipe = TrotPipeline(model, data, params=_params_from(args),
                        n_cycles=args.cycles, control_hz=args.control_hz,
                        ekf_hz=args.ekf_hz, placement=not args.no_placement,
                        batch_average=not args.ekf_per_sample)
    print("Viewer: stand -> VMC hold -> IN-PLACE TROT (estimator in the loop)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.perf_counter()
        while viewer.is_running() and data.time < pipe.total_t:
            pipe.step()
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = data.time - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:
                t0 = time.perf_counter() - data.time


def _params_from(args):
    p = dog5_trot.TrotParams()
    p.period = args.period
    p.ds_frac = args.ds_frac
    p.lift = args.lift
    if args.k_place is not None:
        p.k_place = args.k_place
    return p


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--period", type=float, default=0.40)
    ap.add_argument("--ds-frac", type=float, default=0.34)
    ap.add_argument("--lift", type=float, default=0.022)
    ap.add_argument("--k-place", type=float, default=None)
    ap.add_argument("--cycles", type=int, default=25)
    ap.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    ap.add_argument("--ekf-hz", type=float, default=DEFAULT_EKF_HZ)
    ap.add_argument("--no-placement", action="store_true",
                    help="ablate the EKF-driven foothold term")
    ap.add_argument("--com-bias-x", type=float, default=0.0)
    ap.add_argument("--com-bias-y", type=float, default=0.0)
    ap.add_argument("--ekf-per-sample", action="store_true",
                    help="integrate every buffered IMU sample instead of "
                         "collapsing the batch to its mean (A/B; measured to "
                         "make no difference -- the update rate is the limit)")
    return ap


def main():
    args = build_argparser().parse_args()
    if args.headless:
        m = run(n_cycles=args.cycles, params=_params_from(args),
                control_hz=args.control_hz, ekf_hz=args.ekf_hz,
                placement=not args.no_placement,
                com_bias=(args.com_bias_x, args.com_bias_y),
                batch_average=not args.ekf_per_sample, quiet=False)
        _print_metrics(m)
        return 0
    _live_loop(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

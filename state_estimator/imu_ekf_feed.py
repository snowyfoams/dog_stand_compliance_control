"""IMU -> EKF bridge: surfaces the DETA10 raw inertial packet in the dog FLU
frame and buffers it for the estimator's predict step.

`ImuDog` (IMU_sensor/imu_dog.py) only wires the AHRS (attitude) packet. The
state estimator needs the raw *specific force* and *angular rate* from the IMU
packet (type 0x40), which the DETA10 driver exposes via `on_imu` / `latest_imu`.
This bridge registers that callback on the same `DETA10` instance, converts
NED -> FLU, and keeps a buffer so the (slow) control loop can batch every IMU
sample into the EKF at its own rate.

Frame (settled Phase 0, see imu_dog.py):
    sensor = NED body (X fwd, Y right, Z down); at rest & level accel ~ (0,0,-9.81)
    dog    = FLU body (X fwd, Y left,  Z up);   NED -> FLU negates Y and Z
So FLU specific force at rest & level is ~ (0, 0, +9.81) -- exactly what the
estimator (dog5_state_estimator / plan.md eq. 3) expects.
"""
from __future__ import annotations

import collections
import threading
import time
from pathlib import Path
import sys

import numpy as np

# Locate IMU_sensor relative to this file (repo/dog_stand_compliance_control/
# state_estimator/imu_ekf_feed.py -> repo/IMU_sensor), NOT via $HOME -- $HOME is
# /root under sudo/chrt, which would break the import.
_IMU_DIR = Path(__file__).resolve().parents[2] / "IMU_sensor"
if str(_IMU_DIR) not in sys.path:
    sys.path.insert(0, str(_IMU_DIR))

from imu_dog import ImuDog, DEFAULT_PORT  # noqa: E402


def ned_to_flu(v):
    """NED sensor vector -> FLU dog vector: keep X, negate Y and Z."""
    return np.array([v[0], -v[1], -v[2]], dtype=float)


class ImuEkfFeed:
    """Buffered FLU raw-IMU feed built on top of ImuDog.

    Usage:
        with ImuEkfFeed(port) as feed:
            feed.wait_for_raw(timeout=3.0)
            ...
            for f_flu, w_flu, t in feed.drain():
                est.predict(f_flu, w_flu, dt, contacts)
            att = feed.attitude()          # AHRS roll/pitch cross-check
    """

    def __init__(self, port: str = DEFAULT_PORT):
        self.imu = ImuDog(port)                    # reuse lifecycle + AHRS ref
        self._buf = collections.deque(maxlen=4096)
        self._last = None
        self._lock = threading.Lock()
        # register the raw-IMU callback on the underlying DETA10 (coexists with
        # ImuDog's own on_ahrs handler).
        self.imu._imu.on_imu(self._on_imu)

    # -- callback --------------------------------------------------------
    def _on_imu(self, d) -> None:
        f_flu = ned_to_flu(d.accel)                # specific force, m/s^2
        w_flu = ned_to_flu(d.gyro)                 # angular rate, rad/s
        t = time.monotonic()
        with self._lock:
            self._buf.append((f_flu, w_flu, t))
            self._last = (f_flu, w_flu, t)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "ImuEkfFeed":
        self.imu.start()
        return self

    def stop(self) -> None:
        self.imu.stop()

    def __enter__(self) -> "ImuEkfFeed":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def wait_for_raw(self, timeout: float = 3.0) -> bool:
        """Block until a raw IMU (0x40) packet has arrived.

        `ImuDog.wait_for_data` only waits for *any* frame (often AHRS); this
        specifically confirms the raw inertial stream is enabled on the sensor.
        """
        deadline = time.monotonic() + timeout
        self.imu.wait_for_data(timeout=timeout)
        while time.monotonic() < deadline:
            if self._last is not None:
                return True
            time.sleep(0.005)
        return self._last is not None

    # -- data ------------------------------------------------------------
    def drain(self):
        """Return and clear all buffered (f_flu, w_flu, t_monotonic) samples."""
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
        return items

    def latest(self):
        with self._lock:
            return self._last

    def attitude(self):
        """AHRS-derived dog-frame attitude (roll/pitch trusted) for cross-check."""
        return self.imu.sample()

    @property
    def rate_hz(self) -> float:
        return self.imu.rate_hz

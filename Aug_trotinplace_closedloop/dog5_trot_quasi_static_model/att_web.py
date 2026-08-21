"""Live roll/pitch/yaw dashboard for trot_hw --web (stdlib only, no installs).

WHAT STREAMS, AND WHERE EACH NUMBER COMES FROM
    roll / pitch   the DETA10 AHRS in the dog FLU frame, mounting offsets
                   applied (imu_dog.sample()) -- the same signal the control
                   loop trusts, read by this module's own sampler thread.
    loop r / p     the runner's view: est.roll/est.pitch with the SETPOINT_*
                   pair subtracted, published by the CAN thread.  The gap
                   between the two traces IS the setpoint (plus imu_calib).
    yaw            the DETA10's magnetometer heading.  imu_dog.py's warning
                   holds: the mag sits next to 12 motors and a steel frame,
                   so this is DISPLAY ONLY -- watch it jump when torque comes
                   on, and do not close a loop on it.  yaw RATE (gyro wz) is
                   inertial and trustworthy.
    rates          gyro, dog frame, deg/s.

REUSE, NOT A FORK
    Telemetry, the HTTP handler, the dual-stack server and the URL printer
    are state_estimator/ekf_web's, imported -- the pattern the gate-C8 stand
    and the D1 walk streamed on.  Only the sampler (AHRS instead of EKF
    mailbox) and the page's field map / cards are this file's own.

ISOLATION (the control loop is real-time, this is not)
    * the sampler READS imu.sample() -- a rebound-dataclass peek, no lock,
      never the estimator, never the bus;
    * the runner publishes `shared.status` by REBINDING a small dict, never
      mutating one in place, so a reader cannot see a half-write;
    * daemon threads + its own socket: a stalled browser cannot block the
      trot, and a dead port 8080 must not kill a run (the caller catches
      OSError and trots without the page).

RUN
    trot_hw.py --web        then open the printed URL from any machine.
"""
from __future__ import annotations

import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
_SE = os.path.join(os.path.dirname(_AUG), "state_estimator")
if _SE not in sys.path:
    sys.path.insert(0, _SE)

from ekf_web import (Telemetry, _DualStackServer, _make_handler,  # noqa: E402
                     _urls, DEFAULT_PORT, DEFAULT_SAMPLE_HZ)

# sample tuple layout, shared with the page's JS (keep in sync)
# The per-joint torque triple (tau_des / tau_cmd / tau_meas) deliberately
# does NOT ride in these rows: it is 36 numbers a sweep and the page shows
# only the LATEST value of each, so it travels in `status` -- rebound whole
# by the runner, no history kept -- and the torque table reads it there.
FIELDS = ["t", "roll", "pitch", "yaw", "est_r", "est_p",
          "wr", "wp", "wy", "c0", "c1", "c2", "c3", "stale"]


class AttShared:
    """Mailbox between the trot's CAN thread and the sampler thread.

    The CAN thread REBINDS .status (a small plain dict) once per sweep; the
    sampler reads it and the ImuDog without locks.  imu is read-only here:
    sample()/rate_hz/age_s are rebound-attribute peeks on the serial thread's
    writes, the same access pattern the control loop itself uses.
    """

    def __init__(self, imu):
        self.imu = imu
        self.status = {}


def _sample(shared, t0):
    """One telemetry row (read-only; None-safe before the AHRS speaks)."""
    row = [round(time.monotonic() - t0, 3)]
    st = shared.status or {}
    s = shared.imu.sample()
    if s is None:
        row += [None] * 3
    else:
        row += [round(s.roll_deg, 3), round(s.pitch_deg, 3),
                round(s.yaw_deg, 2)]
    row += [st.get("est_roll"), st.get("est_pitch")]
    if s is None:
        row += [None] * 3
    else:
        row += [round(s.roll_rate_dps, 2), round(s.pitch_rate_dps, 2),
                round(s.yaw_rate_dps, 2)]
    planted = st.get("planted")
    row += ([int(bool(c)) for c in planted] if planted is not None
            else [None] * 4)
    row += [int(shared.imu.is_stale())]
    return row


def _sampler_loop(shared, tel, stop_evt, hz):
    t0 = time.monotonic()
    period = 1.0 / hz
    while not stop_evt.is_set():
        start = time.perf_counter()
        try:
            tel.append(_sample(shared, t0))
            st = dict(shared.status or {})
            st["imu_hz"] = round(float(shared.imu.rate_hz), 1)
            st["crc"] = int(shared.imu.crc_error_count)
            tel.status = st
        except Exception:
            pass          # telemetry must never take the trot down
        rem = period - (time.perf_counter() - start)
        if rem > 0:
            stop_evt.wait(rem)


def start(shared, port=DEFAULT_PORT, hz=DEFAULT_SAMPLE_HZ):
    """Start sampler + HTTP server as daemon threads.  Returns (stop, urls).

    Raises OSError if the port cannot be bound -- the caller decides whether
    a run without the page is still a run (it is)."""
    tel = Telemetry()
    stop_evt = threading.Event()
    httpd = _DualStackServer(("::", port), _make_handler(tel, page=PAGE))
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                     daemon=True).start()
    threading.Thread(target=_sampler_loop, args=(shared, tel, stop_evt, hz),
                     daemon=True).start()

    def stop():
        stop_evt.set()
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass
    return stop, _urls(port)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>DOG5 attitude live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1216;--panel:#171b21;--line:#262c35;--fg:#e6edf3;--dim:#8b98a5;
      --ok:#3fb950;--bad:#f85149;--a:#58a6ff;--b:#f0883e;--c:#a371f7;--d:#3fb950}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;
       padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line);
       position:sticky;top:0;z-index:2}
h1{font-size:14px;margin:0 12px 0 0;font-weight:600;letter-spacing:.02em}
.kv{color:var(--dim)} .kv b{color:var(--fg);font-weight:600}
.pill{padding:2px 8px;border-radius:999px;font-weight:600}
.ok{background:rgba(63,185,80,.15);color:var(--ok)}
.bad{background:rgba(248,81,73,.15);color:var(--bad)}
main{padding:12px;display:grid;gap:12px;
     grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
      padding:8px 10px 4px}
.card h2{font-size:12px;margin:0 0 4px;color:var(--dim);font-weight:600;
         display:flex;justify-content:space-between}
.card h2 span{color:var(--fg)}
canvas{width:100%;height:120px;display:block}
.legend{display:flex;gap:12px;font-size:11px;color:var(--dim);padding:2px 0 4px;
        flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:12px;margin:2px 0 6px}
th,td{padding:2px 8px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;color:var(--dim)}
thead th{color:var(--dim);font-weight:600}
tr.clip td{background:rgba(248,81,73,.12)}
tr.atcap td{background:rgba(240,136,62,.18)}
.sw{display:inline-block;width:10px;height:2px;vertical-align:middle;margin-right:4px}
#err{color:var(--bad);padding:0 14px}
</style></head><body>
<header>
  <h1>DOG5 attitude</h1>
  <span class="kv">stage <b id="stage">--</b></span>
  <span class="kv">swing <b id="swing">--</b></span>
  <span class="kv">contacts <b id="cbits">----</b></span>
  <span class="kv">yaw lock <b id="ylock">--</b></span>
  <span class="kv">yaw err <b id="yerr">--</b></span>
  <span class="kv">clip <b id="clipn">--</b>/12</span>
  <span class="kv">cap <b id="capv">--</b></span>
  <span class="kv">imu <b id="imuhz">--</b></span>
  <span class="kv">crc <b id="crc">--</b></span>
  <span id="health" class="pill bad">WAIT</span>
  <span class="kv" id="rate"></span>
</header>
<div id="err"></div>
<main>
  <div class="card"><h2>roll <span id="v_r"></span></h2>
    <div class="legend"><span><i class="sw" style="background:var(--a)"></i>AHRS</span>
      <span><i class="sw" style="background:var(--dim)"></i>loop (setpoint-subtracted)</span></div>
    <canvas id="c_r"></canvas></div>
  <div class="card"><h2>pitch <span id="v_p"></span></h2>
    <div class="legend"><span><i class="sw" style="background:var(--b)"></i>AHRS</span>
      <span><i class="sw" style="background:var(--dim)"></i>loop (setpoint-subtracted)</span></div>
    <canvas id="c_p"></canvas></div>
  <div class="card"><h2>yaw (magnetometer -- display only) <span id="v_y"></span></h2>
    <canvas id="c_y"></canvas></div>
  <div class="card"><h2>body rates deg/s <span id="v_w"></span></h2>
    <div class="legend"><span><i class="sw" style="background:var(--a)"></i>roll</span>
      <span><i class="sw" style="background:var(--b)"></i>pitch</span>
      <span><i class="sw" style="background:var(--c)"></i>yaw (gyro, trustworthy)</span></div>
    <canvas id="c_w"></canvas></div>
  <div class="card"><h2>contacts FL FR RL RR</h2><canvas id="c_c"></canvas></div>
  <div class="card" style="grid-column:1/-1">
    <h2>torque per joint, Nm <span id="t_note"></span></h2>
    <div class="legend">
      <span>des = what the control law asked for</span>
      <span>cmd = what the safety gate let through (ramp / cap / slew)</span>
      <span>meas = what the driver reports doing</span>
      <span style="color:var(--bad)">red row: gate clipped this joint</span>
      <span style="color:var(--b)">orange row: pinned at the cap</span></div>
    <table id="tautab"><thead><tr><th>joint</th><th>des</th><th>cmd</th>
      <th>meas</th><th>des&#8722;cmd</th><th>cmd&#8722;meas</th></tr></thead>
      <tbody></tbody></table></div>
</main>
<script>
const F={t:0,roll:1,pitch:2,yaw:3,er:4,ep:5,wr:6,wp:7,wy:8,
         c0:9,c1:10,c2:11,c3:12,stale:13};
const WIN=30;                       // seconds shown
let rows=[],seq=0,frozen=false,lastN=0,lastT=performance.now();

function css(n){return getComputedStyle(document.documentElement)
  .getPropertyValue(n).trim();}

function prep(cv){
  const d=window.devicePixelRatio||1, r=cv.getBoundingClientRect();
  if(cv.width!==Math.round(r.width*d)||cv.height!==Math.round(r.height*d)){
    cv.width=Math.round(r.width*d); cv.height=Math.round(r.height*d);}
  const x=cv.getContext('2d'); x.setTransform(d,0,0,d,0,0);
  x.clearRect(0,0,r.width,r.height); return [x,r.width,r.height];
}

function chart(id,series,fmt){
  const cv=document.getElementById(id); const [x,W,H]=prep(cv);
  const view=frozen?rows:rows.filter(r=>r[0]>=(rows.length?rows[rows.length-1][0]-WIN:0));
  if(!view.length) return;
  const t1=view[view.length-1][0], t0=frozen?view[0][0]:t1-WIN;
  let lo=Infinity,hi=-Infinity;
  for(const s of series) for(const r of view){
    const v=r[s.i]; if(v===null||v===undefined) continue;
    if(v<lo)lo=v; if(v>hi)hi=v;}
  if(!isFinite(lo)){lo=-1;hi=1;}
  if(hi-lo<s_min(series)){const m=(hi+lo)/2,h=s_min(series)/2;lo=m-h;hi=m+h;}
  const pad=(hi-lo)*0.15; lo-=pad; hi+=pad;
  const X=t=>(t-t0)/(t1-t0||1)*(W-38)+34, Y=v=>H-14-(v-lo)/(hi-lo)*(H-24);
  // grid + axis labels
  x.strokeStyle=css('--line'); x.fillStyle=css('--dim');
  x.font='10px ui-monospace,monospace'; x.lineWidth=1;
  for(let k=0;k<=2;k++){const v=lo+(hi-lo)*k/2, y=Math.round(Y(v))+.5;
    x.beginPath();x.moveTo(34,y);x.lineTo(W-4,y);x.stroke();
    x.fillText(fmt(v),2,y+3);}
  for(const s of series){
    x.strokeStyle=css(s.c); x.lineWidth=s.w||1.5;
    if(s.dash)x.setLineDash([3,3]); else x.setLineDash([]);
    x.beginPath(); let pen=false;
    for(const r of view){const v=r[s.i];
      if(v===null||v===undefined){pen=false;continue;}
      const px=X(r[0]),py=Y(v);
      if(pen)x.lineTo(px,py); else {x.moveTo(px,py);pen=true;}}
    x.stroke();}
  x.setLineDash([]);
}
function s_min(series){return series.min||1e-6;}

const JN=['FL_abd','FL_pitch','FL_knee','FR_abd','FR_pitch','FR_knee',
          'RL_abd','RL_pitch','RL_knee','RR_abd','RR_pitch','RR_knee'];
function renderTau(s){
  // All 12 joints, straight from the latest sweep's status -- no worst-joint
  // reduction.  Rows are built once and cells updated in place.
  const g=(id,v)=>document.getElementById(id).textContent=v;
  const des=s.tau_des,cmd=s.tau_cmd,meas=s.tau_meas,cap=s.tau_cap;
  const tb=document.querySelector('#tautab tbody');
  g('capv',cap!=null?cap.toFixed(2)+' Nm':'--');
  if(!des||!cmd||!meas){
    tb.innerHTML=''; g('clipn','--');
    document.getElementById('t_note').textContent='no torque yet';
    return;}
  if(tb.rows.length!==12){
    tb.innerHTML='';
    for(let j=0;j<12;j++){const tr=tb.insertRow();
      for(let k=0;k<6;k++)tr.insertCell();
      tr.cells[0].textContent=JN[j];}}
  let nclip=0;
  for(let j=0;j<12;j++){
    const d=des[j],c=cmd[j],m=meas[j],tr=tb.rows[j];
    const clip=Math.abs(d-c)>0.001;
    if(clip)nclip++;
    tr.cells[1].textContent=d.toFixed(2);
    tr.cells[2].textContent=c.toFixed(2);
    tr.cells[3].textContent=m.toFixed(2);
    tr.cells[4].textContent=(d-c).toFixed(2);
    tr.cells[5].textContent=(c-m).toFixed(2);
    tr.className=clip?(cap!=null&&Math.abs(c)>=0.99*cap?'atcap':'clip'):'';}
  g('clipn',nclip);
  document.getElementById('t_note').textContent='';
}

function contacts(){
  const cv=document.getElementById('c_c'); const [x,W,H]=prep(cv);
  const view=frozen?rows:rows.filter(r=>r[0]>=(rows.length?rows[rows.length-1][0]-WIN:0));
  if(!view.length)return;
  const t1=view[view.length-1][0], t0=frozen?view[0][0]:t1-WIN;
  const names=['FL','FR','RL','RR'], lane=(H-8)/4;
  x.font='10px ui-monospace,monospace';
  for(let L=0;L<4;L++){
    const y=4+L*lane;
    x.fillStyle=css('--line'); x.fillRect(24,y+2,W-28,lane-6);
    x.fillStyle=css('--d');
    for(let k=0;k<view.length;k++){
      if(!view[k][F.c0+L])continue;
      const a=(view[k][0]-t0)/(t1-t0||1)*(W-28)+24;
      const nx=k+1<view.length?view[k+1][0]:view[k][0]+0.05;
      const b=(nx-t0)/(t1-t0||1)*(W-28)+24;
      x.fillRect(a,y+2,Math.max(1,b-a),lane-6);}
    x.fillStyle=css('--dim'); x.fillText(names[L],2,y+lane/2+3);}
}

function draw(){
  chart('c_r',[{i:F.roll,c:'--a'},{i:F.er,c:'--dim',dash:1,w:1}],v=>v.toFixed(1));
  chart('c_p',[{i:F.pitch,c:'--b'},{i:F.ep,c:'--dim',dash:1,w:1}],v=>v.toFixed(1));
  chart('c_y',[{i:F.yaw,c:'--c'}],v=>v.toFixed(1));
  chart('c_w',[{i:F.wr,c:'--a'},{i:F.wp,c:'--b'},{i:F.wy,c:'--c'}],v=>v.toFixed(0));
  contacts();
  const last=rows[rows.length-1];
  if(last){
    const set=(id,v)=>document.getElementById(id).textContent=v;
    set('v_r',last[F.roll]===null?'--':last[F.roll].toFixed(2)+'°');
    set('v_p',last[F.pitch]===null?'--':last[F.pitch].toFixed(2)+'°');
    set('v_y',last[F.yaw]===null?'--':last[F.yaw].toFixed(1)+'°');
    set('v_w',last[F.wr]===null?'--':`${last[F.wr].toFixed(0)}, ${last[F.wp].toFixed(0)}, ${last[F.wy].toFixed(0)}`);
    set('cbits',[0,1,2,3].map(i=>{const v=last[F.c0+i];
      return v===null||v===undefined?'-':(v?'1':'0');}).join(''));
  }
}

async function poll(){
  try{
    const r=await fetch('/data?since='+seq,{cache:'no-store'});
    const j=await r.json();
    document.getElementById('err').textContent='';
    if(j.seq<seq)rows=[];                    // runner restarted
    seq=j.seq;
    if(j.rows.length){rows=rows.concat(j.rows);
      const cut=rows[rows.length-1][0]-240;
      while(rows.length&&rows[0][0]<cut)rows.shift();}
    const s=j.status||{};
    frozen=!!s.frozen;
    const g=(id,v)=>document.getElementById(id).textContent=v;
    g('stage',s.stage||'--'); g('swing',s.swing!=null?s.swing:'--');
    g('imuhz',s.imu_hz!=null?s.imu_hz.toFixed(0)+' Hz':'--');
    g('crc',s.crc!=null?s.crc:'--');
    // "--" until the runner latches, which is NOT the same as 0.0 deg of
    // error: before the lock exists the spring is not acting at all, and a
    // zero would read as "held perfectly".
    g('ylock',s.yaw_lock!=null?s.yaw_lock.toFixed(1)+String.fromCharCode(176):'--');
    g('yerr',s.yaw_err!=null?(s.yaw_err>=0?'+':'')+s.yaw_err.toFixed(1)+String.fromCharCode(176):'--');
    renderTau(s);
    const last=rows[rows.length-1];
    const hp=document.getElementById('health');
    if(!last||last[F.roll]===null){hp.textContent='NO AHRS';hp.className='pill bad';}
    else if(last[F.stale]){hp.textContent='STALE';hp.className='pill bad';}
    else{hp.textContent='LIVE';hp.className='pill ok';}
    lastN+=j.rows.length;
    const now=performance.now();
    if(now-lastT>1000){
      document.getElementById('rate').textContent=
        `${(lastN*1000/(now-lastT)).toFixed(0)} Hz`;
      lastN=0;lastT=now;}
    draw();
  }catch(e){
    document.getElementById('err').textContent='connection lost -- retrying';
  }
  setTimeout(poll,100);
}
addEventListener('resize',draw); poll();
</script></body></html>
"""

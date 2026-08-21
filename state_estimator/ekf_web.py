"""Live EKF telemetry dashboard served over HTTP (stdlib only, no installs).

The hardware runners already publish the estimator state into an `EkfShared`
mailbox.  This module samples that mailbox from its own daemon thread and
serves a self-contained HTML page with scrolling strip charts, so the full
state can be watched from a browser on another machine -- no X forwarding, no
matplotlib, and nothing added to the CAN control thread beyond the plain
attribute writes it already does.

Isolation rules (the control loop is real-time, this is not):
  * the sampler only READS `shared`; it never touches the estimator or the bus;
  * the runner publishes `shared.status` by REBINDING a small dict (never by
    mutating one in place), so a reader can never observe a half-written dict;
  * the HTTP server runs with daemon threads and its own socket -- a stalled
    browser cannot block the walk.

Live use (wired into walk1_hw.py --web):
    walk1_hw.py --raw-log walk.npz --web            # then open the printed URL

Replay use (any recorded log, after the run):
    python3 ekf_web.py walk.npz                     # serves the replayed run
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

DEFAULT_PORT = 8080
DEFAULT_SAMPLE_HZ = 20.0
HISTORY_S = 120.0

# sample tuple layout, shared with the page's JS (keep in sync)
FIELDS = ["t", "z", "vx", "vy", "vz", "roll", "pitch", "yaw",
          "ahrs_r", "ahrs_p", "c0", "c1", "c2", "c3", "healthy", "bf", "bw"]


class Telemetry:
    """Sequence-numbered ring buffer of EKF samples."""

    def __init__(self, capacity=int(HISTORY_S * DEFAULT_SAMPLE_HZ)):
        self.capacity = int(capacity)
        self._rows = []
        self._first_seq = 0          # seq of self._rows[0]
        self._lock = threading.Lock()
        self.status = {}

    def append(self, row):
        with self._lock:
            self._rows.append(row)
            if len(self._rows) > self.capacity:
                drop = len(self._rows) - self.capacity
                del self._rows[:drop]
                self._first_seq += drop

    def since(self, seq):
        """Rows with sequence >= seq, plus the next sequence number."""
        with self._lock:
            start = max(0, int(seq) - self._first_seq)
            rows = self._rows[start:]
            return rows, self._first_seq + len(self._rows), dict(self.status)

    def extend(self, rows):
        with self._lock:
            self._rows.extend(rows)
            self._first_seq = 0
            self.capacity = max(self.capacity, len(self._rows))


def _sample(shared, t0):
    """One telemetry row from the shared mailbox (read-only)."""
    out = shared.out
    contacts = shared.contacts
    row = [round(time.monotonic() - t0, 3)]
    if out is None:
        row += [None] * 9
    else:
        from ekf_runtime import _rpy
        r, p, y = _rpy(out["C"])
        v = out["v_body"]
        st = shared.status or {}
        row += [round(float(out["r"][2]) * 1e3, 1),
                round(float(v[0]) * 1e3, 1), round(float(v[1]) * 1e3, 1),
                round(float(v[2]) * 1e3, 1),
                round(np.degrees(r), 3), round(np.degrees(p), 3),
                round(np.degrees(y), 3),
                st.get("ahrs_roll"), st.get("ahrs_pitch")]
    row += [int(bool(c)) for c in contacts]
    if out is None:
        row += [0, None, None]
    else:
        row += [int(bool(out["healthy"])),
                round(float(np.linalg.norm(out["bf"])), 4),
                round(float(np.linalg.norm(out["bw"])), 6)]
    return row


def sampler_loop(shared, tel, stop_evt, hz=DEFAULT_SAMPLE_HZ):
    t0 = time.monotonic()
    period = 1.0 / hz
    while not stop_evt.is_set():
        start = time.perf_counter()
        try:
            tel.append(_sample(shared, t0))
            st = dict(shared.status or {})
            st["est_ready"] = bool(shared.est_ready)
            st["bias_str"] = shared.bias_str
            tel.status = st
        except Exception:
            pass          # telemetry must never take the walk down
        rem = period - (time.perf_counter() - start)
        if rem > 0:
            stop_evt.wait(rem)


def _make_handler(tel, page=None):
    page_html = PAGE if page is None else page

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):      # keep the runner's stdout clean
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path.startswith("/data"):
                seq = 0
                if "?" in self.path:
                    qs = self.path.split("?", 1)[1]
                    for part in qs.split("&"):
                        if part.startswith("since="):
                            try:
                                seq = int(part[6:])
                            except ValueError:
                                seq = 0
                rows, next_seq, status = tel.since(seq)
                body = json.dumps({"seq": next_seq, "rows": rows,
                                   "status": status}).encode()
                self._send(body, "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(page_html.encode(), "text/html; charset=utf-8")
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
    return Handler


class _DualStackServer(ThreadingHTTPServer):
    """Bind :: with V6ONLY off so IPv4 and IPv6 clients both reach it."""
    daemon_threads = True
    allow_reuse_address = True
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        super().server_bind()


def _urls(port):
    """Reachable URLs: every non-loopback IPv4 on the box, plus the mDNS name.

    Link-local IPv6 is deliberately omitted -- it needs a scope id
    (http://[fe80::1%eth0]:8080) which browsers handle badly.  `ip -br -4 addr`
    is used because getaddrinfo(hostname) misses secondary addresses, and it is
    read once at startup, never in the control path.
    """
    out = []
    try:
        import subprocess
        txt = subprocess.run(["ip", "-br", "-4", "addr", "show"],
                             capture_output=True, text=True, timeout=2).stdout
        for ln in txt.splitlines():
            parts = ln.split()
            if not parts or parts[0].startswith("lo"):
                continue
            for tok in parts[2:]:
                ip = tok.split("/")[0]
                if ip and not ip.startswith("127."):
                    out.append(f"http://{ip}:{port}/")
    except Exception:
        pass
    out.append(f"http://{socket.gethostname()}.local:{port}/")
    return out


def start(shared, port=DEFAULT_PORT, hz=DEFAULT_SAMPLE_HZ):
    """Start sampler + HTTP server as daemon threads.  Returns a stop()."""
    tel = Telemetry()
    stop_evt = threading.Event()
    httpd = _DualStackServer(("::", port), _make_handler(tel))
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                     daemon=True).start()
    threading.Thread(target=sampler_loop, args=(shared, tel, stop_evt, hz),
                     daemon=True).start()

    def stop():
        stop_evt.set()
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass
    return stop, _urls(port)


def serve_replay(npz_path, port=DEFAULT_PORT):
    """Replay a recorded log and serve the result as the same page."""
    import hw_replay
    print(f"[web] replaying {npz_path} (this takes a while for a long log; "
          "the server binds when it finishes)...", flush=True)
    data = hw_replay.load(npz_path)
    res = hw_replay.replay(data)
    enc_t, enc_c = data["enc_t"], data["enc_contacts"]
    idx = np.searchsorted(enc_t, res["t"])
    idx = np.clip(idx, 0, len(enc_c) - 1)
    t0 = res["t"][0]
    rows = []
    for k in range(len(res["t"])):
        c = enc_c[idx[k]]
        rows.append([
            round(float(res["t"][k] - t0), 3),
            round(float(res["z"][k]) * 1e3, 1),
            round(float(res["v"][k][0]) * 1e3, 1),
            round(float(res["v"][k][1]) * 1e3, 1),
            round(float(res["v"][k][2]) * 1e3, 1),
            round(float(np.degrees(res["roll_e"][k])), 3),
            round(float(np.degrees(res["pitch_e"][k])), 3),
            None,
            round(float(np.degrees(res["roll_a"][k])), 3),
            round(float(np.degrees(res["pitch_a"][k])), 3),
            int(c[0]), int(c[1]), int(c[2]), int(c[3]),
            int(bool(res["healthy"][k])),
            round(float(np.linalg.norm(res["bf"][k])), 4),
            round(float(np.linalg.norm(res["bw"][k])), 6),
        ])
    tel = Telemetry(capacity=len(rows))
    tel.extend(rows)
    tel.status = {"stage": "REPLAY", "source": str(npz_path),
                  "est_ready": True,
                  "touchdowns": len(res.get("touchdowns", [])),
                  "frozen": True}
    httpd = _DualStackServer(("::", port), _make_handler(tel))
    for u in _urls(port):
        print(f"[web] replay of {npz_path} at {u}")
    print("[web] Ctrl-C to stop")
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>DOG5 EKF live</title>
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
.legend{display:flex;gap:12px;font-size:11px;color:var(--dim);padding:2px 0 4px}
.sw{display:inline-block;width:10px;height:2px;vertical-align:middle;margin-right:4px}
#err{color:var(--bad);padding:0 14px}
</style></head><body>
<header>
  <h1>DOG5 EKF</h1>
  <span class="kv">stage <b id="stage">--</b></span>
  <span class="kv">step <b id="step">--</b></span>
  <span class="kv">swing <b id="swing">--</b></span>
  <span class="kv">contacts <b id="cbits">----</b></span>
  <span class="kv">blk <b id="blk">--</b></span>
  <span class="kv">|bf| <b id="bf">--</b></span>
  <span class="kv">|bw| <b id="bw">--</b></span>
  <span id="health" class="pill bad">INIT</span>
  <span class="kv" id="rate"></span>
</header>
<div id="err"></div>
<main>
  <div class="card"><h2>base height z <span id="v_z"></span></h2>
    <canvas id="c_z"></canvas></div>
  <div class="card"><h2>body velocity <span id="v_v"></span></h2>
    <div class="legend"><span><i class="sw" style="background:var(--a)"></i>vx</span>
      <span><i class="sw" style="background:var(--b)"></i>vy</span>
      <span><i class="sw" style="background:var(--c)"></i>vz</span></div>
    <canvas id="c_v"></canvas></div>
  <div class="card"><h2>roll <span id="v_r"></span></h2>
    <div class="legend"><span><i class="sw" style="background:var(--a)"></i>EKF</span>
      <span><i class="sw" style="background:var(--dim)"></i>AHRS</span></div>
    <canvas id="c_r"></canvas></div>
  <div class="card"><h2>pitch <span id="v_p"></span></h2>
    <div class="legend"><span><i class="sw" style="background:var(--b)"></i>EKF</span>
      <span><i class="sw" style="background:var(--dim)"></i>AHRS</span></div>
    <canvas id="c_p"></canvas></div>
  <div class="card"><h2>yaw (unobservable, drifts) <span id="v_y"></span></h2>
    <canvas id="c_y"></canvas></div>
  <div class="card"><h2>contacts FL FR RL RR</h2><canvas id="c_c"></canvas></div>
</main>
<script>
const F={t:0,z:1,vx:2,vy:3,vz:4,roll:5,pitch:6,yaw:7,ar:8,ap:9,
         c0:10,c1:11,c2:12,c3:13,h:14,bf:15,bw:16};
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
  chart('c_z',[{i:F.z,c:'--a'}],v=>v.toFixed(0));
  chart('c_v',[{i:F.vx,c:'--a'},{i:F.vy,c:'--b'},{i:F.vz,c:'--c'}],v=>v.toFixed(0));
  chart('c_r',[{i:F.roll,c:'--a'},{i:F.ar,c:'--dim',dash:1,w:1}],v=>v.toFixed(1));
  chart('c_p',[{i:F.pitch,c:'--b'},{i:F.ap,c:'--dim',dash:1,w:1}],v=>v.toFixed(1));
  chart('c_y',[{i:F.yaw,c:'--c'}],v=>v.toFixed(1));
  contacts();
  const last=rows[rows.length-1];
  if(last){
    const set=(id,v)=>document.getElementById(id).textContent=v;
    set('v_z',last[F.z]===null?'--':last[F.z].toFixed(0)+' mm');
    set('v_v',last[F.vx]===null?'--':`${last[F.vx].toFixed(0)}, ${last[F.vy].toFixed(0)}, ${last[F.vz].toFixed(0)} mm/s`);
    set('v_r',last[F.roll]===null?'--':last[F.roll].toFixed(2)+'°');
    set('v_p',last[F.pitch]===null?'--':last[F.pitch].toFixed(2)+'°');
    set('v_y',last[F.yaw]===null?'--':last[F.yaw].toFixed(1)+'°');
    set('cbits',[0,1,2,3].map(i=>last[F.c0+i]?'1':'0').join(''));
    set('bf',last[F.bf]===null?'--':last[F.bf].toFixed(3));
    set('bw',last[F.bw]===null?'--':last[F.bw].toExponential(1));
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
    g('stage',s.stage||'--'); g('step',s.step||'--');
    g('swing',s.swing||'--'); g('blk',s.blk_ms!=null?s.blk_ms.toFixed(1)+' ms':'--');
    const last=rows[rows.length-1];
    const hp=document.getElementById('health');
    if(!s.est_ready){hp.textContent='INITIALISING';hp.className='pill bad';}
    else if(last&&last[F.h]){hp.textContent='HEALTHY';hp.className='pill ok';}
    else{hp.textContent='UNHEALTHY';hp.className='pill bad';}
    lastN+=j.rows.length;
    const now=performance.now();
    if(now-lastT>1000){
      document.getElementById('rate').textContent=
        frozen?`${rows.length} samples (replay)`:`${(lastN*1000/(now-lastT)).toFixed(0)} Hz`;
      lastN=0;lastT=now;}
    draw();
  }catch(e){
    document.getElementById('err').textContent='connection lost -- retrying';
  }
  setTimeout(poll,frozen?2000:100);
}
addEventListener('resize',draw); poll();
</script></body></html>
"""


def main():
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="recorded log to replay and serve")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    serve_replay(args.npz, args.port)


if __name__ == "__main__":
    main()

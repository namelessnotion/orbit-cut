"""Stage 4 — the review UI, and the only place this system learns anything.

A local server on the stdlib, deliberately. The whole app is one page and three
endpoints, and adding a framework would mean another dependency in a project
where three undeclared ones have already caused a crash.

Two things here are less obvious than they look.

**HTTP range requests are not optional.** `SimpleHTTPRequestHandler` answers
every GET with 200 and the whole file, and a browser given a 200 for a video
cannot seek — Safari will not even start playing. Every clip in this UI is a
seek into the middle of a ten-minute proxy, so `_serve_range` implements 206
properly, including the zero-length probe (`bytes=0-1`) browsers open with.

**The decision is the product.** Approvals and rejections are the one thing in
this pipeline that cannot be recomputed: proxies, scores and candidates can all
be regenerated from the originals, but a judgement is a human minute spent. So
every decision is written through to SQLite immediately rather than batched, and
in and out adjustments are stored *alongside* the machine's suggestion rather
than replacing it. The difference between the two is the training signal — it
says which way the selector is biased, which a bare approve/reject cannot.
"""
from __future__ import annotations

import json
import re
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import db

REASONS = ["too shaky", "bad light", "boring", "already have one like it",
           "wrong in/out"]
CHUNK = 512 * 1024


def _rows(conn, ride: str | None) -> list[dict]:
    assets = {a["content_hash"]: a for a in db.assets(conn)}
    out = []
    for s in db.segments(conn):
        a = assets.get(s["content_hash"])
        if not a or not a["proxy_path"] or not Path(a["proxy_path"]).exists():
            continue
        if ride and ride not in (a["ride_id"] or "") and ride not in (a["filename"] or ""):
            continue
        out.append({
            "id": s["id"], "hash": s["content_hash"],
            "ride": a["ride_id"] or a["content_hash"][:8],
            "file": a["filename"], "lighting": a["lighting"] or "-",
            "t_in": s["t_in"], "t_out": s["t_out"],
            "t_in_user": s["t_in_user"], "t_out_user": s["t_out_user"],
            "rank": s["rank"], "score": s["score"], "dominant": s["dominant"],
            "status": s["status"], "reason": s["reason"],
            "duration": a["duration_s"] or 0.0,
        })
    out.sort(key=lambda r: (r["ride"], r["rank"] or 0))
    return out


PAGE = """<!doctype html><meta charset="utf-8"><title>OrbitCut review</title>
<style>
:root{--ink:#11100E;--paper:#F4EFE3;--muted:#8A928B;--accent:#E2673A;
      --ok:#96BB6F;--no:#C0844A}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--paper);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;height:100vh}
#list{width:340px;overflow-y:auto;border-right:1px solid #2a2926;flex:none}
.row{padding:8px 12px;border-bottom:1px solid #1d1c19;cursor:pointer}
.row.on{background:#1f1e1a;border-left:3px solid var(--accent);padding-left:9px}
.row .t{color:var(--muted);font-size:12px}
.approved{color:var(--ok)} .rejected{color:var(--no)}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
video{flex:1;min-height:0;background:#000;width:100%;object-fit:contain}
#bar{padding:10px 16px;border-top:1px solid #2a2926;display:flex;gap:20px;
     align-items:center;flex-wrap:wrap}
kbd{background:#2a2926;border-radius:3px;padding:1px 5px;color:var(--paper)}
#help{color:var(--muted);font-size:12px}
#count{margin-left:auto;color:var(--muted)}
.chip{border:1px solid #3a3935;border-radius:12px;padding:2px 9px;margin-right:5px;
      color:var(--muted);font-size:12px;cursor:pointer}
.chip:hover{border-color:var(--accent);color:var(--paper)}
</style>
<div id=list></div>
<div id=main>
  <video id=v autoplay muted playsinline></video>
  <div id=err style="display:none;padding:10px 16px;background:#3a2119;color:#E2673A"></div>
  <div id=bar>
    <div id=now></div>
    <div id=chips></div>
    <div id=count></div>
    <div id=help><kbd>j</kbd>/<kbd>k</kbd> move &nbsp;<kbd>a</kbd> approve
      &nbsp;<kbd>x</kbd> reject &nbsp;<kbd>u</kbd> undo
      &nbsp;<kbd>[</kbd><kbd>]</kbd> in &nbsp;<kbd>-</kbd><kbd>=</kbd> out
      &nbsp;<kbd>r</kbd> replay</div>
  </div>
</div>
<script>
const DATA = __DATA__, REASONS = __REASONS__;
let i = 0, v = document.getElementById('v'), loaded = null;
const tin  = r => r.t_in_user  ?? r.t_in;
const tout = r => r.t_out_user ?? r.t_out;

function draw(){
  document.getElementById('list').innerHTML = DATA.map((r,n)=>
    `<div class="row ${n===i?'on':''} ${r.status}" onclick="go(${n})">
       <div>${r.ride} · #${r.rank??'-'} · ${r.dominant??''}
         ${r.status==='approved'?'✓':r.status==='rejected'?'✗':''}</div>
       <div class=t>${tin(r).toFixed(1)}–${tout(r).toFixed(1)}s ·
         ${(tout(r)-tin(r)).toFixed(1)}s · score ${(r.score??0).toFixed(2)}
         ${r.reason?'· '+r.reason:''}</div>
     </div>`).join('');
  const r = DATA[i]; if(!r) return;
  document.getElementById('now').textContent =
    `${r.file}  ${tin(r).toFixed(1)}–${tout(r).toFixed(1)}s  ${r.dominant??''}`;
  document.getElementById('chips').innerHTML = REASONS.map((x,n)=>
    `<span class=chip onclick="decide('rejected','${x}')">${n+1} ${x}</span>`).join('');
  const done = DATA.filter(d=>d.status!=='candidate').length;
  document.getElementById('count').textContent = `${done}/${DATA.length} decided`;
  document.querySelector('.row.on')?.scrollIntoView({block:'nearest'});
}
function play(){
  const r = DATA[i]; if(!r) return;
  if(loaded !== r.hash){ v.src = '/proxy/'+r.hash; loaded = r.hash;
    v.onloadedmetadata = ()=>{ v.currentTime = tin(r); v.play().catch(()=>{}); }; }
  else { v.currentTime = tin(r); v.play().catch(()=>{}); }
}
v.addEventListener('error', ()=>{
  const r = DATA[i], e = document.getElementById('err');
  e.style.display = 'block';
  e.textContent = `cannot play the proxy for ${r?.file ?? '?'} — it may be missing `
    + `or truncated. Rebuild it:  orbitcut ingest <the original> --force`;
});
v.addEventListener('loadeddata', ()=>{ document.getElementById('err').style.display='none'; });
v.addEventListener('timeupdate', ()=>{ const r = DATA[i]; if(!r) return;
  if(v.currentTime >= tout(r) || v.currentTime < tin(r) - 0.5) v.currentTime = tin(r); });
function go(n){ i = Math.max(0, Math.min(DATA.length-1, n)); draw(); play(); }
async function post(body){
  const res = await fetch('/api/decide', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  return res.json();
}
async function decide(status, reason){
  const r = DATA[i]; if(!r) return;
  Object.assign(r, await post({id:r.id, status, reason: reason||null,
                               t_in_user:r.t_in_user, t_out_user:r.t_out_user}));
  draw(); if(i < DATA.length-1) go(i+1); else draw();
}
async function nudge(field, delta){
  const r = DATA[i]; if(!r) return;
  const cur = field==='t_in_user' ? tin(r) : tout(r);
  r[field] = Math.max(0, Math.min(r.duration, cur + delta));
  if(tout(r) - tin(r) < 2){ r[field] = cur; return; }
  Object.assign(r, await post({id:r.id, status:r.status, reason:r.reason,
                               t_in_user:r.t_in_user, t_out_user:r.t_out_user}));
  draw(); play();
}
addEventListener('keydown', e=>{
  const k = e.key;
  if(k==='j'||k==='ArrowDown') go(i+1);
  else if(k==='k'||k==='ArrowUp') go(i-1);
  else if(k==='a') decide('approved', null);
  else if(k==='x') decide('rejected', null);
  else if(k==='u') decide('candidate', null);
  else if(k==='r') play();
  else if(k==='[') nudge('t_in_user', -0.5);
  else if(k===']') nudge('t_in_user',  0.5);
  else if(k==='-') nudge('t_out_user', -0.5);
  else if(k==='=') nudge('t_out_user',  0.5);
  else if(/^[1-5]$/.test(k)) decide('rejected', REASONS[+k-1]);
  else return;
  e.preventDefault();
});
draw(); play();
</script>"""


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []
    proxies: dict[str, str] = {}

    def log_message(self, *_a):        # a request log per video chunk is noise
        pass

    def do_GET(self):
        if self.path == "/":
            body = (PAGE.replace("__DATA__", json.dumps(self.rows))
                        .replace("__REASONS__", json.dumps(REASONS))).encode()
            self._send(200, "text/html; charset=utf-8", body)
        elif self.path == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        elif self.path.startswith("/proxy/"):
            path = self.proxies.get(self.path.rsplit("/", 1)[-1])
            if not path or not Path(path).exists():
                self._send(404, "text/plain", b"no proxy")
            else:
                self._serve_range(Path(path))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/api/decide":
            return self._send(404, "text/plain", b"not found")
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")

        conn = db.connect()
        conn.execute(
            """UPDATE segment SET status = ?, reason = ?, t_in_user = ?,
                                  t_out_user = ?, decided_at = ?
               WHERE id = ?""",
            (req.get("status") or "candidate", req.get("reason"),
             req.get("t_in_user"), req.get("t_out_user"),
             db.now() if req.get("status") != "candidate" else None, req["id"]))
        conn.commit()
        row = conn.execute("SELECT * FROM segment WHERE id = ?", (req["id"],)).fetchone()
        conn.close()

        out = {"status": row["status"], "reason": row["reason"],
               "t_in_user": row["t_in_user"], "t_out_user": row["t_out_user"]}
        for r in self.rows:
            if r["id"] == req["id"]:
                r.update(out)
        self._send(200, "application/json", json.dumps(out).encode())

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_range(self, path: Path):
        """206 Partial Content, properly.

        Without this a browser cannot seek, and seeking is the entire job here:
        every clip is a jump into the middle of a ten-minute file. Safari will
        not play a video served as a plain 200 at all, and it opens with a
        `bytes=0-1` probe purely to find out whether ranges are supported — so
        that degenerate two-byte request has to be answered correctly.
        """
        size = path.stat().st_size
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if not m or not rng:
            start, end = 0, size - 1
            code = 200
        else:
            s, e = m.group(1), m.group(2)
            if s == "":                       # suffix form: last N bytes
                length = min(int(e or 0), size)
                start, end = size - length, size - 1
            else:
                start = int(s)
                end = int(e) if e else min(start + CHUNK - 1, size - 1)
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))
            code = 206

        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return          # the browser seeked away; not an error
                left -= len(chunk)


def serve(conn, ride: str | None, port: int = 0, open_browser: bool = True):
    """Start the review server. Returns (server, url, count)."""
    rows = _rows(conn, ride)
    if not rows:
        return None, None, 0
    Handler.rows = rows
    Handler.proxies = {a["content_hash"]: a["proxy_path"]
                       for a in db.assets(conn) if a["proxy_path"]}

    # Bind to loopback only: this serves your footage and takes decisions, and
    # neither belongs on the network.
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return server, url, len(rows)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

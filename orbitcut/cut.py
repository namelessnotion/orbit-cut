"""Stage 5 by hand — assembling one post. Approved clips in, one Reel out.

`render` renders everything approved, grouped by ride, in the order it happened.
That is the right default and the wrong tool for making one post: a Reel is
three or four clips you chose, in an order you chose, from whichever rides they
came from. Choosing that at the command line means holding thirty in-points in
your head. So this is the same job with the clips on screen.

Two things it does that `render` does not.

**It joins across rides.** Which is where a trap lives: this library is 52 rides
at 29.97 fps and 37 at 59.94, and stream-copying those together does not fail,
does not come out short, and does not look wrong in any check made afterwards.
It comes out variable-rate — measured, 180 frames over 4.02 s declaring 60 fps
and averaging 45 — and what happens to that is decided by Instagram's re-encode
on somebody's phone. So every part of a joined cut is rendered at one frame
rate, chosen by `render.plan_fps` before anything is encoded.

**It queues.** A 30-second clip cut from a 5.3K HEVC original is most of a
minute of decode, so a four-clip Reel is a coffee. The queue is a table rather
than a list in memory: reloading the tab, or closing it, must not lose twenty
minutes of work, and afterwards it is still answerable which segments went into
which file.

What it deliberately does not do is touch `segment`. A rendered clip stays
`approved`, because `approved` is the label `fit` trains on and moving it to
`rendered` would quietly shrink the training set every time you posted
something. What was rendered lives in `render_job`, which is disposable; the
decision log is not.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from . import archive as arch_mod, config, db, render as rn, webui

CUTS = "cuts"               # under $ORBITCUT_ROOT/renders
POLL_S = 0.4                # how often the worker looks for a new job
WRITE_S = 0.5               # how often progress reaches SQLite while rendering


# --------------------------------------------------------------------- reading
def clips(conn, ride: str | None = None) -> list[dict]:
    """Every approved clip, with what the UI needs to decide about it.

    `playable` and `renderable` are separate questions with separate answers:
    review and preview read the 540p proxy, rendering reads the original, and
    the originals are the half that gets archived off a laptop first.
    """
    assets = {a["content_hash"]: a for a in db.assets(conn)}
    out = []
    for s in db.segments(conn, status="approved"):
        a = assets.get(s["content_hash"])
        if not a:
            continue
        if ride and ride not in (a["ride_id"] or "") and ride not in (a["filename"] or ""):
            continue
        proxy = a["proxy_path"]
        t_in = s["t_in_user"] if s["t_in_user"] is not None else s["t_in"]
        t_out = s["t_out_user"] if s["t_out_user"] is not None else s["t_out"]
        out.append({
            "id": s["id"], "hash": s["content_hash"],
            "ride": a["ride_id"] or a["content_hash"][:8],
            "chapter": a["chapter"] or 0,
            "file": a["filename"], "dominant": s["dominant"] or "",
            "score": s["score"], "rank": s["rank"],
            "t_in": t_in, "t_out": t_out, "duration": t_out - t_in,
            "fps": a["fps"] or 0.0,
            "shape": _shape(a["width"], a["height"]),
            "lighting": a["lighting"] or "",
            "playable": bool(proxy and Path(proxy).exists()),
            "renderable": bool(a["source_path"] and Path(a["source_path"]).exists()),
        })
    out.sort(key=lambda r: (r["ride"], r["chapter"], r["t_in"]))
    return out


def _shape(w, h) -> str:
    """The label that matters when framing: how much pan is left after a 9:16 crop.

    An 8:7 frame keeps 2010 px of it, a 16:9 frame 2626 px of a much shorter
    box — the crop clears 1080 wide either way, so nothing fails, but a wide
    source shows far less of the trail ahead. Worth seeing before you pick.
    """
    if not w or not h:
        return ""
    r = w / h
    return "16:9" if r > 1.5 else "8:7"


def _jobs(conn) -> list[dict]:
    rows = []
    for j in db.jobs(conn):
        rows.append({
            "id": j["id"], "name": j["name"], "status": j["status"],
            "step": j["step"] or "", "error": j["error"] or "",
            "done_s": j["done_s"] or 0.0, "total_s": j["total_s"] or 0.0,
            "clips": len(json.loads(j["segment_ids"])),
            "level": j["level"] or "none", "frame": j["frame"] or "centre",
            "fps": j["fps"],
            "out": j["out_path"] or "",
            "ready": bool(j["out_path"] and j["status"] == "done"
                          and Path(j["out_path"]).exists()),
        })
    return rows


# ---------------------------------------------------------------------- naming
def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return s[:60] or "cut"


def _discard(files: list[Path], folder: Path | None) -> None:
    """Remove what a cancelled or failed job wrote. Nothing else.

    `folder` is only ever a directory this job created for itself — `_free`
    guarantees the name was unused — so it can go whole. A single-clip job has
    no folder of its own and only its one file is removed.
    """
    for f in files:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)


def _free(path: Path) -> Path:
    """A name nothing is using yet. Never overwrite a render you have posted."""
    if not path.exists():
        return path
    for n in range(2, 999):
        cand = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"too many renders named {path.stem}")


# ---------------------------------------------------------------------- worker
class Worker(threading.Thread):
    """One job at a time, oldest first.

    Serial on purpose. The encoder is the bottleneck and there is one of it —
    two ffmpegs competing for VideoToolbox finish two clips no faster than one
    after the other, and they turn one readable progress figure into two
    half-finished ones.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.cancelled: set[int] = set()
        self.current: int | None = None

    def run(self):
        conn = db.connect()          # this thread's own; sqlite objects are not shared
        try:
            while not self.stop.is_set():
                job = db.next_job(conn)
                if job is None:
                    self.wake.wait(POLL_S)
                    self.wake.clear()
                    continue
                if job["id"] in self.cancelled:
                    db.update_job(conn, job["id"], status="cancelled",
                                  finished_at=db.now())
                    continue
                self._run_job(conn, job)
        finally:
            conn.close()

    def cancel(self, job_id: int) -> None:
        self.cancelled.add(job_id)
        self.wake.set()

    def shutdown(self, timeout: float = 15.0) -> None:
        """Stop, and take the running ffmpeg with us.

        A subprocess is not a child that dies when its parent does. Leaving
        Ctrl-C to sort this out means an orphaned encoder still writing to a
        file nothing is tracking any more, so the running job is cancelled the
        same way the UI cancels one — `render._run`'s watchdog sees it within a
        quarter second and kills ffmpeg — and we wait for that before returning.
        Measured at 0.7 s from shutdown to no ffmpeg running.
        """
        self.stop.set()
        if self.current is not None:
            self.cancelled.add(self.current)
        self.wake.set()
        self.join(timeout)

    def _run_job(self, conn, job) -> None:
        job_id = self.current = job["id"]
        db.update_job(conn, job_id, status="running", started_at=db.now(),
                      step="preparing", error=None)
        try:
            items = self._items(conn, json.loads(job["segment_ids"]))
            out = self._render(conn, job, items)
        except rn.Cancelled:
            db.update_job(conn, job_id, status="cancelled", step="",
                          finished_at=db.now())
        except Exception as exc:
            db.update_job(conn, job_id, status="error", step="",
                          error=f"{type(exc).__name__}: {exc}"[:400],
                          finished_at=db.now())
        else:
            db.update_job(conn, job_id, status="done", step="",
                          out_path=str(out), done_s=job["total_s"],
                          finished_at=db.now())
        finally:
            self.current = None

    def _items(self, conn, ids: list[int]) -> list[dict]:
        """Resolve segment ids to (original, in, out), in the order given.

        The order is the edit, so it is preserved exactly as it arrived rather
        than sorted back into ride order the way `render` does.
        """
        items = []
        for sid in ids:
            row = conn.execute(
                """SELECT s.*, a.source_path, a.archived_path, a.telemetry_path,
                          a.proxy_path, a.track_path, a.filename, a.ride_id, a.fps
                     FROM segment s JOIN asset a ON a.content_hash = s.content_hash
                    WHERE s.id = ?""", (sid,)).fetchone()
            if row is None:
                raise RuntimeError(f"segment {sid} is gone")
            seg = dict(row)
            seg["source_path"] = str(arch_mod.ensure_original(conn, seg))
            t_in = row["t_in_user"] if row["t_in_user"] is not None else row["t_in"]
            t_out = row["t_out_user"] if row["t_out_user"] is not None else row["t_out"]
            items.append({"seg": seg, "t_in": t_in, "t_out": t_out,
                          "dur": t_out - t_in})
        if not items:
            raise RuntimeError("nothing selected")
        return items

    def _render(self, conn, job, items: list[dict]) -> Path:
        job_id, name = job["id"], slug(job["name"])
        level = None if (job["level"] or "none") == "none" else job["level"]
        frame = job["frame"] or "centre"
        root = config.RENDERS / CUTS
        root.mkdir(parents=True, exist_ok=True)

        one = len(items) == 1
        d = root if one else _free(root / name)
        if not one:
            d.mkdir(parents=True)

        fps = job["fps"]
        total = sum(i["dur"] for i in items)
        done = 0.0
        last = [0.0]

        def stop() -> bool:
            return job_id in self.cancelled or self.stop.is_set()

        def tick(secs: float) -> None:
            # Throttled: ffmpeg reports several times a second and every report
            # would otherwise be a commit on the same file the UI is polling.
            now = time.monotonic()
            if now - last[0] < WRITE_S:
                return
            last[0] = now
            db.update_job(conn, job_id, done_s=min(done + secs, total))

        # Every path this job will write, tracked from before it is written:
        # a cancelled ffmpeg leaves a truncated mp4 behind, and a half-rendered
        # clip sitting in the renders folder next to finished ones — same name
        # shape, same place, playable for two seconds — is the kind of thing
        # that gets posted.
        parts: list[Path] = []
        try:
            for n, it in enumerate(items, 1):
                seg = it["seg"]
                stem = (name if one else
                        f"{n:02d}_{seg['ride_id'] or 'clip'}_{seg['dominant'] or 'clip'}")
                parts.append(_free(d / f"{stem}.mp4"))
                db.update_job(conn, job_id, step=(
                    f"clip {n} of {len(items)} — {seg['filename']} "
                    f"@ {it['t_in']:.0f}s"), done_s=done)
                rn.clip(seg["source_path"], it["t_in"], it["t_out"], parts[-1],
                        level=level, telemetry=seg["telemetry_path"],
                        preview=seg["proxy_path"], fps=fps, frame=frame,
                        track=seg["track_path"], progress=tick, stop=stop)
                if stop():
                    raise rn.Cancelled(f"job {job_id}")
                done += it["dur"]
                db.update_job(conn, job_id, done_s=done)

            if one:
                return parts[0]

            db.update_job(conn, job_id, step=f"joining {len(parts)} clips")
            return rn.compile_reel(parts, d / f"{name}.mp4", progress=tick, stop=stop)
        except BaseException:
            _discard(parts, None if one else d)
            raise


# ---------------------------------------------------------------------- server
PAGE = r"""<!doctype html><meta charset="utf-8"><title>OrbitCut cut</title>
<style>
:root{--ink:#11100E;--paper:#F4EFE3;--muted:#8A928B;--accent:#E2673A;
      --ok:#96BB6F;--no:#C0844A;--line:#2a2926}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--paper);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;height:100vh}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
   margin:0;padding:10px 14px 6px;font-weight:400}
#list{width:330px;overflow-y:auto;border-right:1px solid var(--line);flex:none}
#side{width:360px;overflow-y:auto;border-left:1px solid var(--line);flex:none;
      display:flex;flex-direction:column}
.rid{padding:10px 14px 2px;color:var(--muted);font-size:12px;
     border-top:1px solid #1d1c19}
.row{padding:7px 14px;border-bottom:1px solid #1d1c19;cursor:pointer;display:flex;
     gap:9px;align-items:baseline}
.row.on{background:#1f1e1a;border-left:3px solid var(--accent);padding-left:11px}
.row.dead{opacity:.45}
.n{flex:none;width:20px;height:20px;border:1px solid #3a3935;border-radius:4px;
   text-align:center;font-size:11px;line-height:19px;color:var(--muted)}
.row.sel .n{background:var(--accent);border-color:var(--accent);color:#11100E}
.row .t{color:var(--muted);font-size:12px}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
video{flex:1;min-height:0;background:#000;width:100%;object-fit:contain}
#bar{padding:9px 16px;border-top:1px solid var(--line);display:flex;gap:18px;
     align-items:center;flex-wrap:wrap;font-size:13px}
#help{color:var(--muted);font-size:12px;margin-left:auto}
kbd{background:var(--line);border-radius:3px;padding:1px 5px;color:var(--paper)}
#err{display:none;padding:9px 16px;background:#3a2119;color:var(--accent)}
#pick{border-bottom:1px solid var(--line)}
.p{display:flex;gap:8px;align-items:baseline;padding:4px 14px;font-size:13px}
.p:hover{background:#1a1915}
.p .g{color:var(--muted);font-size:12px;flex:1;min-width:0;overflow:hidden;
      text-overflow:ellipsis;white-space:nowrap}
.x{cursor:pointer;color:var(--muted);flex:none}
.x:hover{color:var(--accent)}
.note{padding:4px 14px;color:var(--muted);font-size:12px}
.warn{color:var(--no)}
#form{padding:10px 14px;border-top:1px solid var(--line);display:grid;gap:8px}
input,select,button{font:inherit;background:#1a1915;color:var(--paper);
  border:1px solid #3a3935;border-radius:4px;padding:5px 8px}
button{cursor:pointer;background:#232019}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.4;cursor:default}
button.go{background:var(--accent);border-color:var(--accent);color:#11100E}
.job{padding:8px 14px;border-bottom:1px solid #1d1c19;font-size:12px}
.job .h{display:flex;gap:8px;align-items:baseline}
.job .h b{font-weight:400;flex:1;min-width:0;overflow:hidden;
          text-overflow:ellipsis;white-space:nowrap}
.st{color:var(--muted)} .st.done{color:var(--ok)} .st.error{color:var(--accent)}
.st.running{color:var(--paper)}
.bar{height:3px;background:#232019;border-radius:2px;margin-top:5px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
.path{color:var(--muted);word-break:break-all}
a{color:var(--muted)}
</style>
<div id=list></div>
<div id=main>
  <video id=v autoplay muted playsinline></video>
  <div id=err></div>
  <div id=bar>
    <div id=now></div>
    <div id=help>
      <kbd>j</kbd>/<kbd>k</kbd> move &nbsp;<kbd>space</kbd> pick
      &nbsp;<kbd>p</kbd> play the cut &nbsp;<kbd>r</kbd> replay
      &nbsp;<kbd>c</kbd> clear &nbsp;<kbd>enter</kbd> queue</div>
  </div>
</div>
<div id=side>
  <h2>the cut <span id=tot></span></h2>
  <div id=pick></div>
  <div id=form>
    <input id=name placeholder="name this cut">
    <select id=level>
      <option value=none>horizon: leave it</option>
      <option value=constant>horizon: remove the mount offset</option>
      <option value=dynamic>horizon: lock per frame</option>
    </select>
    <select id=frame>
      <option value=centre>framing: centred crop</option>
      <option value=subject>framing: follow Orbit</option>
    </select>
    <select id=fps>
      <option value=fastest>mixed rates: match the fastest clip</option>
      <option value=slowest>mixed rates: conform down to the slowest</option>
    </select>
    <button class=go id=one onclick="queue(true)">render as one file</button>
    <button id=each onclick="queue(false)">render each separately</button>
  </div>
  <h2>queue</h2>
  <div id=jobs></div>
</div>
<script>
const DATA = __DATA__, MAX_REEL_S = __MAX_REEL_S__;
let i = 0, sel = [], jobs = [], loaded = null, chain = null;
const v = document.getElementById('v');
const el = id => document.getElementById(id);
const fmt = s => s >= 60 ? `${Math.floor(s/60)}m${String(Math.round(s%60)).padStart(2,'0')}`
                         : `${s.toFixed(1)}s`;

function drawList(){
  let html = '', ride = null;
  DATA.forEach((r,n) => {
    if(r.ride !== ride){ ride = r.ride;
      html += `<div class=rid>${ride} · ${r.shape} · ${r.fps.toFixed(2)} fps`
            + `${r.lighting ? ' · ' + r.lighting : ''}</div>`; }
    const k = sel.indexOf(r.id);
    html += `<div class="row ${n===i?'on':''} ${k>=0?'sel':''} ${r.renderable?'':'dead'}"
                  onclick="go(${n})">
      <span class=n onclick="event.stopPropagation();pick(${n})">${k>=0?k+1:''}</span>
      <span style=flex:1>
        <div>${r.dominant||'clip'} · ${r.duration.toFixed(1)}s
             ${r.renderable?'':'<span class=warn>original missing</span>'}</div>
        <div class=t>${r.t_in.toFixed(0)}–${r.t_out.toFixed(0)}s ·
             score ${(r.score??0).toFixed(2)}</div>
      </span></div>`;
  });
  el('list').innerHTML = html;
  document.querySelector('.row.on')?.scrollIntoView({block:'nearest'});
}

function drawPick(){
  const rows = sel.map(id => DATA.find(d => d.id === id)).filter(Boolean);
  const total = rows.reduce((a,r) => a + r.duration, 0);
  el('tot').textContent = rows.length ? `— ${rows.length} clips, ${fmt(total)}` : '';
  el('pick').innerHTML = rows.map((r,n) => `<div class=p>
      <span class=x onclick="move(${n},-1)">↑</span>
      <span class=x onclick="move(${n},1)">↓</span>
      <span class=g>${n+1}. ${r.ride} ${r.dominant||'clip'} ${r.duration.toFixed(1)}s</span>
      <span class=x onclick="pickId(${r.id})">✕</span></div>`).join('')
    || '<div class=note>nothing picked yet — <kbd>space</kbd> on a clip, or click its box</div>';

  // Two things about a selection are worth saying before it costs ten minutes
  // of encoding: whether it will be over Instagram's limit, and whether its
  // parts disagree about frame rate — which is the one mismatch that would
  // otherwise reach the finished file.
  const notes = [];
  const rates = [...new Set(rows.map(r => r.fps.toFixed(2)))];
  if(rates.length > 1){
    const all = rows.map(r => r.fps);
    const target = el('fps').value === 'slowest' ? Math.min(...all) : Math.max(...all);
    notes.push(`<div class=note>mixed frame rates (${rates.join(', ')}) — every
      part renders at ${target.toFixed(2)}, because copying them together as
      they are would come out variable-rate</div>`);
  }
  if(total > MAX_REEL_S)
    notes.push(`<div class="note warn">${fmt(total)} is over Instagram's three
      minutes — it will still render, but it will not post as one Reel</div>`);
  if(rows.some(r => !r.renderable))
    notes.push(`<div class="note warn">some picks have no original on disk and
      will fail — run <kbd>orbitcut relink</kbd></div>`);
  if(rows.length) el('pick').innerHTML += notes.join('');
  el('one').disabled = el('each').disabled = !rows.length;
  el('one').textContent = rows.length > 1
    ? `render ${rows.length} clips as one file` : 'render this clip';
  el('each').style.display = rows.length > 1 ? '' : 'none';
}

function drawJobs(){
  el('jobs').innerHTML = jobs.map(j => {
    const pct = j.total_s ? Math.min(100, 100 * j.done_s / j.total_s) : 0;
    const can = j.status === 'queued' || j.status === 'running';
    return `<div class=job>
      <div class=h><b>${j.name}</b>
        <span class="st ${j.status}">${j.status}</span>
        ${can ? `<span class=x onclick="cancel(${j.id})">✕</span>` : ''}
        ${j.ready ? `<span class=x onclick="playOut(${j.id})">▶</span>` : ''}</div>
      <div class=st>${j.clips} clip${j.clips>1?'s':''} · ${fmt(j.total_s)}
        ${j.fps ? '· ' + j.fps.toFixed(2) + ' fps' : ''}
        ${j.level !== 'none' ? '· ' + j.level : ''}</div>
      ${j.step ? `<div class=st>${j.step}</div>` : ''}
      ${j.error ? `<div class="st error">${j.error}</div>` : ''}
      ${j.status === 'running' ? `<div class=bar><i style=width:${pct}%></i></div>` : ''}
      ${j.out ? `<div class=path>${j.out}</div>` : ''}
    </div>`;
  }).join('') || '<div class=note>nothing queued yet</div>';
}

function draw(){ drawList(); drawPick(); drawJobs();
  const r = DATA[i];
  el('now').textContent = r
    ? `${r.file}  ${r.t_in.toFixed(1)}–${r.t_out.toFixed(1)}s  ${r.dominant||''}`
    : '';
}

// ---- playing. Previewing the cut end to end is the point of picking in a
// browser at all: it is the only way to find out that two clips are the same
// corner before spending ten minutes rendering them next to each other.
function playClip(r, onEnd){
  if(!r.playable){ fail(`no proxy for ${r.file} — orbitcut ingest it again`); return; }
  chain = onEnd || null;
  const start = () => { v.currentTime = r.t_in; v.play().catch(()=>{}); };
  v.dataset.tin = r.t_in; v.dataset.tout = r.t_out;
  if(loaded !== r.hash){ loaded = r.hash; v.src = '/proxy/' + r.hash;
                         v.onloadedmetadata = start; }
  else start();
}
function go(n){ i = Math.max(0, Math.min(DATA.length-1, n)); chain = null; draw();
                playClip(DATA[i]); }
function playOut(id){
  chain = null; loaded = 'job' + id; v.src = '/out/' + id;
  v.dataset.tin = 0; v.dataset.tout = 1e9;
  v.onloadedmetadata = () => v.play().catch(()=>{});
}
function playCut(){
  const rows = sel.map(id => DATA.find(d => d.id === id)).filter(r => r && r.playable);
  if(!rows.length) return;
  const step = n => () => { if(n < rows.length) playClip(rows[n], step(n+1)); };
  step(0)();
}
v.addEventListener('timeupdate', () => {
  const a = +v.dataset.tin, b = +v.dataset.tout;
  if(v.currentTime >= b){ if(chain){ const c = chain; chain = null; c(); }
                          else v.currentTime = a; }
  else if(v.currentTime < a - 0.5) v.currentTime = a;
});
v.addEventListener('loadeddata', () => el('err').style.display = 'none');
v.addEventListener('error', () => fail(`cannot play that — the proxy may be missing`));
function fail(msg){ const e = el('err'); e.style.display = 'block'; e.textContent = msg; }

// ---- picking
function pick(n){ pickId(DATA[n].id); }
function pickId(id){
  const k = sel.indexOf(id);
  if(k >= 0) sel.splice(k,1); else sel.push(id);
  draw();
}
function move(n, d){
  const m = n + d;
  if(m < 0 || m >= sel.length) return;
  [sel[n], sel[m]] = [sel[m], sel[n]];
  draw();
}
function defaultName(){
  const rows = sel.map(id => DATA.find(d => d.id === id)).filter(Boolean);
  if(!rows.length) return 'cut';
  const rides = [...new Set(rows.map(r => r.ride))];
  if(rows.length === 1) return `${rows[0].ride}_${Math.round(rows[0].t_in)}s`;
  return rides.length === 1 ? `${rides[0]}_cut` : `mix_${rows.length}clips`;
}

// ---- queueing
async function queue(joined){
  if(!sel.length) return;
  const name = el('name').value.trim() || defaultName();
  const body = {name, level: el('level').value, frame: el('frame').value,
                fps: el('fps').value,
                jobs: joined ? [sel.slice()] : sel.map(id => [id])};
  const res = await fetch('/api/queue', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const got = await res.json();
  if(got.error) return fail(got.error);
  jobs = got.jobs; sel = []; el('name').value = ''; draw();
}
async function cancel(id){
  const res = await fetch('/api/cancel', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({id})});
  jobs = (await res.json()).jobs; drawJobs();
}
async function poll(){
  try{ const res = await fetch('/api/jobs');
       jobs = (await res.json()).jobs; drawJobs(); }
  catch(e){ /* the server is gone; the page is still readable */ }
}
setInterval(poll, 1000);
el('fps').addEventListener('change', drawPick);

addEventListener('keydown', e => {
  if(e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key;
  if(k === 'j' || k === 'ArrowDown') go(i+1);
  else if(k === 'k' || k === 'ArrowUp') go(i-1);
  else if(k === ' ' || k === 's') pick(i);
  else if(k === 'p') playCut();
  else if(k === 'r') playClip(DATA[i]);
  else if(k === 'c'){ sel = []; draw(); }
  else if(k === 'Enter') queue(!e.shiftKey);
  else return;
  e.preventDefault();
});
draw(); if(DATA.length) playClip(DATA[0]);
</script>"""


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []
    proxies: dict[str, str] = {}
    worker: Worker | None = None

    def log_message(self, *_a):        # a request log per video chunk is noise
        pass

    def do_GET(self):
        if self.path == "/":
            body = (PAGE.replace("__DATA__", json.dumps(self.rows))
                        .replace("__MAX_REEL_S__", repr(rn.MAX_REEL_S))).encode()
            return webui.send(self, 200, "text/html; charset=utf-8", body)
        if self.path == "/favicon.ico":
            return webui.send(self, 204, "image/x-icon", b"")
        if self.path == "/api/jobs":
            return self._json({"jobs": self._read(_jobs)})
        if self.path.startswith("/proxy/"):
            path = self.proxies.get(self.path.rsplit("/", 1)[-1])
            if not path or not Path(path).exists():
                return webui.send(self, 404, "text/plain", b"no proxy")
            return webui.serve_range(self, Path(path))
        if self.path.startswith("/out/"):
            return self._serve_output(self.path.rsplit("/", 1)[-1])
        webui.send(self, 404, "text/plain", b"not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/queue":
            return self._queue(req)
        if self.path == "/api/cancel":
            return self._cancel(req)
        webui.send(self, 404, "text/plain", b"not found")

    # -- helpers
    def _read(self, fn):
        conn = db.connect()
        try:
            return fn(conn)
        finally:
            conn.close()

    def _json(self, obj, code: int = 200):
        webui.send(self, code, "application/json", json.dumps(obj).encode())

    def _serve_output(self, job_id: str):
        """Play a finished render back, so it can be checked before posting."""
        def look(conn):
            row = conn.execute("SELECT out_path FROM render_job WHERE id = ?",
                               (int(job_id),)).fetchone()
            return row["out_path"] if row else None
        path = self._read(look) if job_id.isdigit() else None
        if not path or not Path(path).exists():
            return webui.send(self, 404, "text/plain", b"no render there")
        webui.serve_range(self, Path(path))

    def _queue(self, req):
        """One request, one or many jobs — 'as one file' versus 'each separately'.

        The order the ids arrive in is the order they play, so it is stored as
        given. Anything the browser sent that is not currently an approved clip
        is dropped rather than trusted: the page's copy of the library is a
        snapshot from when it loaded.
        """
        known = {r["id"]: r for r in self.rows}
        name = slug(req.get("name") or "cut")
        level = req.get("level") if req.get("level") in ("constant", "dynamic") else None
        frame = "subject" if req.get("frame") == "subject" else "centre"
        to = "slowest" if req.get("fps") == "slowest" else "fastest"
        groups = [[int(i) for i in g if int(i) in known]
                  for g in (req.get("jobs") or [])]
        groups = [g for g in groups if g]
        if not groups:
            return self._json({"error": "nothing to render"}, 400)

        conn = db.connect()
        try:
            for n, ids in enumerate(groups, 1):
                rows = [known[i] for i in ids]
                fps = rn.plan_fps([r["fps"] for r in rows], to)
                label = name if len(groups) == 1 else f"{name}_{n:02d}"
                db.enqueue_job(conn, label, ids, level, fps,
                               sum(r["duration"] for r in rows), frame=frame)
            out = _jobs(conn)
        finally:
            conn.close()
        if self.worker:
            self.worker.wake.set()
        self._json({"jobs": out})

    def _cancel(self, req):
        job_id = int(req.get("id") or 0)
        if self.worker:
            self.worker.cancel(job_id)
        conn = db.connect()
        try:
            # A queued job can be withdrawn here and now; a running one belongs
            # to the worker, which notices at its next progress report and kills
            # ffmpeg. Either way the row stops being a promise.
            conn.execute("UPDATE render_job SET status = 'cancelled', "
                         "finished_at = ? WHERE id = ? AND status = 'queued'",
                         (db.now(), job_id))
            conn.commit()
            out = _jobs(conn)
        finally:
            conn.close()
        self._json({"jobs": out})


def serve(conn, ride: str | None, port: int = 0, open_browser: bool = True):
    """Start the cut server and its worker. Returns (server, worker, url, count)."""
    rows = clips(conn, ride)
    if not rows:
        return None, None, None, 0
    Handler.rows = rows
    Handler.proxies = {a["content_hash"]: a["proxy_path"]
                       for a in db.assets(conn) if a["proxy_path"]}
    worker = Worker()
    Handler.worker = worker
    worker.start()
    server, url = webui.start(Handler, port, open_browser)
    return server, worker, url, len(rows)

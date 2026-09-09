"""The finished shelf: everything rendered, and how it reaches the phone.

Instagram is posted from a phone, and the renders are on a laptop. The gap is
this: a page listing every finished file, and beside each one a QR code that a
phone camera turns into a download over the house wifi. No cable, no cloud, no
account.

**This is the one thing in OrbitCut that listens on the network, and it does so
on purpose.** `review` and `cut` bind to loopback because they serve the whole
library and take decisions; a QR pointing at 127.0.0.1 is a QR pointing at the
phone itself, so serving a file to another device means being reachable from it.
The exposure is kept as narrow as the job allows:

  * one interface — the LAN address, not 0.0.0.0, so nothing else is offered a
    socket. `--local-only` refuses even that and prints no QR codes.
  * every path sits under a per-run random token. Knowing the address and port
    is not enough, the token is never written to disk, and it dies with the
    process.
  * only finished renders. Paths are resolved and checked to be inside
    `$ORBITCUT_ROOT/renders` before anything is opened, so a crafted URL cannot
    walk out of it into the originals.
  * GET only. There is no endpoint here that writes anything — not to the
    database, not to disk.

It is still your footage on your wifi. On a home network that is the point; on
a café network, use `--local-only` and a cable.
"""
from __future__ import annotations

import json
import secrets
import socket
import subprocess
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from . import config, db, qr, render as rn, webui

POSTER_H = 320          # tall enough to see which clip it is, small enough to fly


# ------------------------------------------------------------------ discovery
def lan_ip() -> str | None:
    """This machine's address on the local network, or None if it has none.

    Opening a UDP socket toward a public address and asking what the kernel
    chose sends no packet — it just makes the routing table answer the question
    "which of my interfaces would reach the world", which is the one the phone
    is also on. `gethostbyname` is not a substitute: on macOS it frequently
    answers 127.0.0.1.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))      # TEST-NET-1, routed nowhere
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


def renders(conn=None) -> list[dict]:
    """Every finished render on the shelf, newest first.

    Reads the filesystem rather than the database on purpose: `render` writes
    per-ride folders and `cut` writes its own, older files predate `render_job`
    entirely, and a file you dropped in by hand is still a file you might want
    to post. The database is consulted only to name what a file came from.
    """
    root = config.RENDERS
    if not root.exists():
        return []
    made = {}
    if conn is not None:
        for j in conn.execute("SELECT name, out_path, segment_ids FROM render_job "
                              "WHERE out_path IS NOT NULL"):
            made[j["out_path"]] = (j["name"], len(json.loads(j["segment_ids"])))

    out = []
    for path in sorted(root.rglob("*.mp4")):
        try:
            stat = path.stat()
        except OSError:
            continue
        meta = _probe(path)
        job = made.get(str(path))
        out.append({
            "path": str(path),
            "name": path.stem,
            "folder": str(path.parent.relative_to(root)) or ".",
            "bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "duration": meta["duration"],
            "width": meta["width"], "height": meta["height"],
            "reel": meta["width"] == rn.TARGET_W and meta["height"] == rn.TARGET_H,
            "clips": job[1] if job else None,
        })
    out.sort(key=lambda r: -r["mtime"])
    return out


_PROBED: dict[tuple, dict] = {}


def _probe(path: Path) -> dict:
    key = (str(path), path.stat().st_mtime, path.stat().st_size)
    if key in _PROBED:
        return _PROBED[key]
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True)
    w = h = 0
    dur = 0.0
    try:
        data = json.loads(r.stdout or "{}")
        st = (data.get("streams") or [{}])[0]
        w, h = int(st.get("width") or 0), int(st.get("height") or 0)
        dur = float((data.get("format") or {}).get("duration") or 0.0)
    except (ValueError, KeyError, json.JSONDecodeError):
        pass
    _PROBED[key] = {"width": w, "height": h, "duration": dur}
    return _PROBED[key]


_POSTERS: dict[str, bytes] = {}


def poster(path: Path, at: float) -> bytes:
    """One frame, as JPEG. Cached, because the gallery asks for all of them.

    Taken a little way in rather than at zero: the first frame of a clip that
    starts on a jump is often motion-blurred to grey, and the point of the
    picture is to tell one clip from another at a glance.
    """
    key = str(path)
    if key in _POSTERS:
        return _POSTERS[key]
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale=-2:{POSTER_H}", "-f", "mjpeg", "-"],
        capture_output=True)
    _POSTERS[key] = r.stdout if r.returncode == 0 else b""
    return _POSTERS[key]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit != "GB" else f"{n:.1f} GB"
        n /= 1024
    return f"{n:.0f} B"


# ----------------------------------------------------------------------- page
PAGE = r"""<!doctype html><meta charset="utf-8"><title>OrbitCut shelf</title>
<style>
:root{--ink:#11100E;--paper:#F4EFE3;--muted:#8A928B;--accent:#E2673A;
      --ok:#96BB6F;--no:#C0844A;--line:#2a2926}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--paper);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
       gap:16px;align-items:baseline;flex-wrap:wrap;position:sticky;top:0;
       background:var(--ink);z-index:2}
h1{font-size:14px;font-weight:400;margin:0;letter-spacing:.08em;
   text-transform:uppercase;color:var(--muted)}
.addr{color:var(--accent)}
.hint{color:var(--muted);font-size:12px;margin-left:auto}
main{padding:16px 20px;display:grid;gap:14px;
     grid-template-columns:repeat(auto-fill,minmax(370px,1fr))}
.card{border:1px solid var(--line);border-radius:6px;overflow:hidden;
      display:flex;background:#151410}
.shot{width:150px;flex:none;background:#000;position:relative;cursor:pointer}
.shot img{width:100%;height:100%;object-fit:cover;display:block}
.shot .play{position:absolute;inset:0;display:grid;place-items:center;
            color:#fff;text-shadow:0 1px 6px #000;font-size:24px;opacity:.85}
.meta{flex:1;min-width:0;padding:10px 12px}
.meta b{font-weight:400;display:block;overflow:hidden;text-overflow:ellipsis;
        white-space:nowrap}
.meta .t{color:var(--muted);font-size:12px}
.warn{color:var(--no)}
.qr{flex:none;padding:8px;background:#fff;display:grid;place-items:center;
    cursor:pointer}
.qr svg{display:block}
video{width:100%;background:#000;display:block}
dialog{border:1px solid var(--line);background:var(--ink);color:var(--paper);
       max-width:min(92vw,420px);border-radius:8px;padding:0}
dialog::backdrop{background:#000c}
.dlg{padding:14px 16px}
.dlg .big{background:#fff;padding:10px;display:grid;place-items:center;
          border-radius:4px}
a{color:var(--accent)}
.none{color:var(--muted);padding:30px 20px}
</style>
<header>
  <h1>shelf</h1>
  <span id=count class=t></span>
  <span>phone → <span class="addr" id=addr></span></span>
  <span class=hint>scan a code with the phone camera · tap a frame to play here</span>
</header>
<main id=grid></main>
<dialog id=dlg><div class=dlg>
  <div class=big id=dlgqr></div>
  <p id=dlgname></p>
  <p class=t>Point the phone camera at this. Same wifi only.</p>
</div></dialog>
<script>
const DATA = __DATA__, BASE = __BASE__, LAN = __LAN__;
const el = id => document.getElementById(id);
const dur = s => s >= 60 ? `${Math.floor(s/60)}m${String(Math.round(s%60)).padStart(2,'0')}s`
                         : `${s.toFixed(1)}s`;
el('addr').textContent = LAN || 'no network — QR codes unavailable';
el('count').textContent = `${DATA.length} finished`;
el('grid').innerHTML = DATA.map((r,n) => `
  <div class=card>
    <div class=shot onclick="play(${n})">
      <img src="${BASE}/p/${n}" loading=lazy alt="">
      <span class=play>▶</span>
    </div>
    <div class=meta>
      <b>${r.name}</b>
      <div class=t>${r.folder === '.' ? '' : r.folder + ' · '}${dur(r.duration)} ·
        ${r.human}${r.clips ? ' · ' + r.clips + ' clips' : ''}</div>
      <div class=t>${r.reel ? `${r.width}×${r.height}`
        : `<span class=warn>${r.width}×${r.height} — not 1080×1920</span>`}</div>
      <div class=t>${r.longer ? '<span class=warn>over 3 minutes for one Reel</span>' : ''}</div>
      <div class=t><a href="${BASE}/dl/${n}">download here</a></div>
    </div>
    ${LAN ? `<div class=qr onclick="big(${n})">${r.qr}</div>` : ''}
  </div>`).join('') || '<p class=none>nothing rendered yet — try <b>orbitcut cut</b></p>';

function play(n){
  const card = el('grid').children[n];
  if(card.querySelector('video')) return;
  const shot = card.querySelector('.shot');
  shot.outerHTML = `<video controls autoplay playsinline style="width:150px;flex:none"
                      src="${BASE}/dl/${n}"></video>`;
}
function big(n){
  el('dlgqr').innerHTML = DATA[n].qrbig;
  el('dlgname').textContent = DATA[n].name;
  el('dlg').showModal();
}
el('dlg').addEventListener('click', e => { if(e.target.id === 'dlg') el('dlg').close(); });
</script>"""

PHONE = r"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__NAME__</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#11100E;color:#F4EFE3;
     font:16px/1.5 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;
     padding:16px;max-width:560px;margin:0 auto}
h1{font-size:18px;margin:0 0 4px;word-break:break-word}
.t{color:#8A928B;font-size:14px}
video{width:100%;border-radius:8px;background:#000;margin:14px 0}
a.dl{display:block;text-align:center;background:#E2673A;color:#11100E;
     text-decoration:none;font-weight:600;padding:15px;border-radius:8px;
     margin:14px 0}
ol{color:#8A928B;font-size:14px;padding-left:20px}
</style>
<h1>__NAME__</h1>
<p class=t>__META__</p>
<video controls playsinline preload=metadata src="__DL__"></video>
<a class=dl href="__DL__" download>Download to this phone</a>
<ol>
  <li>Tap download, then open it from Files (or the download badge).</li>
  <li>Share → Save Video, to put it in Photos.</li>
  <li>Instagram → new Reel → pick it from the camera roll.</li>
</ol>
<p class=t>Served from your laptop over wifi. The link dies when you stop
<code>orbitcut shelf</code>.</p>"""


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []
    token: str = ""
    base: str = ""
    lan: str | None = None

    def log_message(self, *_a):
        pass

    def do_GET(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        # Everything is behind the token, including the gallery. A request that
        # does not carry it is answered 404 rather than 403: a wrong guess
        # should not confirm that there is something here to guess at.
        if len(parts) < 2 or parts[0] != "s" or not secrets.compare_digest(
                parts[1], self.token):
            return webui.send(self, 404, "text/plain", b"not found")
        rest = parts[2:]
        if not rest:
            return self._gallery()
        kind, arg = rest[0], (rest[1] if len(rest) > 1 else "")
        row = self._row(arg)
        if row is None:
            return webui.send(self, 404, "text/plain", b"not found")
        if kind == "p":
            return self._poster(row)
        if kind == "g":
            return self._phone(row)
        if kind == "dl":
            return self._download(row)
        webui.send(self, 404, "text/plain", b"not found")

    def _row(self, arg: str) -> dict | None:
        if not arg.isdigit():
            return None
        n = int(arg)
        return self.rows[n] if 0 <= n < len(self.rows) else None

    def _safe(self, row: dict) -> Path | None:
        """The file, only if it is really inside the renders directory.

        The index came from our own list, so this cannot currently fail — which
        is exactly why it is checked here rather than trusted: the day someone
        adds a by-name endpoint, this is the line that has to already exist.
        """
        p = Path(row["path"]).resolve()
        root = config.RENDERS.resolve()
        return p if p.is_file() and root in p.parents else None

    def _gallery(self):
        data = []
        for n, r in enumerate(self.rows):
            url = f"http://{self.lan}:{self.server.server_address[1]}{self.base}/g/{n}"
            data.append(dict(
                r, human=human(r["bytes"]),
                longer=r["duration"] > 180,
                qr=qr.svg(qr.encode(url), px=2) if self.lan else "",
                qrbig=qr.svg(qr.encode(url), px=8) if self.lan else "",
            ))
        body = (PAGE.replace("__DATA__", json.dumps(data))
                    .replace("__BASE__", json.dumps(self.base))
                    .replace("__LAN__", json.dumps(
                        f"{self.lan}:{self.server.server_address[1]}"
                        if self.lan else ""))).encode()
        webui.send(self, 200, "text/html; charset=utf-8", body)

    def _phone(self, row):
        meta = (f"{row['duration']:.0f}s · {human(row['bytes'])} · "
                f"{row['width']}×{row['height']}")
        n = self.rows.index(row)
        body = (PHONE.replace("__NAME__", row["name"])
                     .replace("__META__", meta)
                     .replace("__DL__", f"{self.base}/dl/{n}")).encode()
        webui.send(self, 200, "text/html; charset=utf-8", body)

    def _poster(self, row):
        path = self._safe(row)
        if path is None:
            return webui.send(self, 404, "text/plain", b"gone")
        img = poster(path, min(0.7, max(0.0, row["duration"] / 4)))
        if not img:
            return webui.send(self, 404, "text/plain", b"no frame")
        webui.send(self, 200, "image/jpeg", img)

    def _download(self, row):
        path = self._safe(row)
        if path is None:
            return webui.send(self, 404, "text/plain", b"gone")
        # An attachment rather than an inline video: tapping it on a phone
        # should put a file in Files, which can then be saved to Photos.
        # Inline, Safari plays it fullscreen and there is nothing to save.
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Disposition",
                         f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as f:
            while chunk := f.read(webui.CHUNK):
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return


def serve(conn, port: int = 0, local_only: bool = False,
          open_browser: bool = True):
    """Start the shelf. Returns (server, url, count, lan)."""
    rows = renders(conn)
    lan = None if local_only else lan_ip()
    token = secrets.token_urlsafe(12)
    Handler.rows = rows
    Handler.token = token
    Handler.base = f"/s/{token}"
    Handler.lan = lan

    host = lan or "127.0.0.1"
    server, _ = webui.start(Handler, port, open_browser=False, host=host)
    url = f"http://{host}:{server.server_address[1]}/s/{token}/"
    if open_browser:
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return server, url, len(rows), lan

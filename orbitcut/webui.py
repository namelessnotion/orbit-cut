"""What the two local browser tools — `review` and `cut` — both need.

A local server on the stdlib, deliberately. Each tool is one page and a handful
of endpoints, and adding a framework would mean another dependency in a project
where three undeclared ones have already caused a crash.

The fiddly part is video, and it is the same fiddly part in both tools.
`SimpleHTTPRequestHandler` answers every GET with 200 and the whole file, and a
browser given a 200 for a video cannot seek — Safari will not even start
playing. Every clip in either UI is a seek into the middle of a ten-minute
proxy, so `serve_range` implements 206 properly, including the zero-length probe
(`bytes=0-1`) browsers open with purely to find out whether ranges are
supported. A second copy of that would be a second place for it to regress.
"""
from __future__ import annotations

import re
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK = 512 * 1024


def send(h: BaseHTTPRequestHandler, code: int, ctype: str, body: bytes) -> None:
    h.send_response(code)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def serve_range(h: BaseHTTPRequestHandler, path: Path,
                ctype: str = "video/mp4") -> None:
    """206 Partial Content, properly.

    Without this a browser cannot seek, and seeking is the entire job here:
    every clip is a jump into the middle of a ten-minute file. Safari will not
    play a video served as a plain 200 at all, and it opens with a `bytes=0-1`
    probe purely to find out whether ranges are supported — so that degenerate
    two-byte request has to be answered correctly.
    """
    size = path.stat().st_size
    rng = h.headers.get("Range", "")
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
    h.send_response(code)
    h.send_header("Content-Type", ctype)
    h.send_header("Accept-Ranges", "bytes")
    h.send_header("Content-Length", str(length))
    if code == 206:
        h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    h.end_headers()
    with path.open("rb") as f:
        f.seek(start)
        left = length
        while left > 0:
            chunk = f.read(min(CHUNK, left))
            if not chunk:
                break
            try:
                h.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return          # the browser seeked away; not an error
            left -= len(chunk)


def start(handler_cls, port: int = 0, open_browser: bool = True,
          host: str = "127.0.0.1"):
    """Bind, serve in a thread, return (server, url).

    Loopback by default, because `review` and `cut` serve your whole library
    and take decisions, and neither belongs on the network. `shelf` overrides
    `host` with this machine's LAN address — deliberately, since handing a file
    to a phone means being reachable from it — and narrows what is exposed in
    the ways its module docstring sets out. Nothing else should pass `host`.
    """
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return server, url


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

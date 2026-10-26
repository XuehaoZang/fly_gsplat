"""Serve the viewer over a directory of viz bundles (written by eval.viz.export).

    python -m eval.viz.view --data outputs/viz [--port 8765]

GET /            -> eval/viz/index.html
GET /index.json  -> {dataset: [method, ...]} scanned from --data
GET /<dataset>/<method>/<file> -> bundle files
GET /img/<dataset>/<method>/<path> -> raw training image, read from frames_root in that bundle's cameras.json
"""
import argparse
import functools
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(HTML, "rb") as f:
                return self._send(f.read(), "text/html; charset=utf-8")
        if path == "/index.json":
            root = self.directory
            index = {d: sorted(m for m in os.listdir(os.path.join(root, d))
                               if os.path.exists(os.path.join(root, d, m, "meta.json")))
                     for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))}
            return self._send(json.dumps({d: m for d, m in index.items() if m}).encode(), "application/json")
        if path.startswith("/img/"):
            return self._image(path[5:])
        return super().do_GET()

    def _image(self, rel):
        parts = rel.split("/", 2)
        ok = len(parts) == 3 and all(p not in ("", ".", "..") for p in parts[:2])
        cams = os.path.join(self.directory, *parts[:2], "cameras.json") if ok else ""
        if not os.path.exists(cams):
            return self.send_error(404)
        with open(cams) as f:
            root = os.path.realpath(json.load(f)["frames_root"])
        target = os.path.realpath(os.path.join(root, parts[2]))
        if not target.startswith(root + os.sep) or not os.path.isfile(target):
            return self.send_error(404)
        with open(target, "rb") as f:
            return self._send(f.read(), "image/png")

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/viz")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    handler = functools.partial(Handler, directory=os.path.abspath(args.data))
    print(f"viewer: http://localhost:{args.port}/   (data: {os.path.abspath(args.data)})")
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()

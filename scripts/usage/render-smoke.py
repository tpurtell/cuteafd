#!/usr/bin/env python3
"""Render /usage in real Chrome using stdlib-only DevTools/WebSocket transport."""
import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

PANELS = ["Flow over time", "Latency", "Prefix cache", "What the cache buys",
          "Speculation", "Clients / protocols / models", "Sessions", "Full log",
          "Errors and stops", "Requests", "Settings"]


class CDP:
    def __init__(self, url):
        parsed = urlparse(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (f"GET {parsed.path} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
                   f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(request.encode())
        self.buffer = b""
        while b"\r\n\r\n" not in self.buffer:
            self.buffer += self.sock.recv(4096)
        headers, self.buffer = self.buffer.split(b"\r\n\r\n", 1)
        if b" 101 " not in headers:
            raise RuntimeError("DevTools WebSocket handshake failed: " + headers.decode())
        self.id = 0
        self.errors = []

    def read(self, n):
        while len(self.buffer) < n:
            chunk = self.sock.recv(max(4096, n - len(self.buffer)))
            if not chunk:
                raise RuntimeError("DevTools closed unexpectedly")
            self.buffer += chunk
        result, self.buffer = self.buffer[:n], self.buffer[n:]
        return result

    def send_frame(self, payload, opcode=1):
        mask = os.urandom(4)
        size = len(payload)
        length = bytes([size | 128]) if size < 126 else bytes([126 | 128]) + struct.pack("!H", size) if size < 65536 else bytes([127 | 128]) + struct.pack("!Q", size)
        self.sock.sendall(bytes([128 | opcode]) + length + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))

    def receive(self):
        parts = []
        while True:
            first, second = self.read(2)
            size = second & 127
            if size == 126:
                size = struct.unpack("!H", self.read(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self.read(8))[0]
            mask = self.read(4) if second & 128 else None
            payload = self.read(size)
            if mask:
                payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
            opcode = first & 15
            if opcode == 9:
                self.send_frame(payload, 10)
                continue
            if opcode == 8:
                raise RuntimeError("DevTools socket closed")
            if opcode in (0, 1):
                parts.append(payload)
                if first & 128:
                    return json.loads(b"".join(parts))

    def event(self, message):
        method, params = message.get("method"), message.get("params", {})
        if method == "Runtime.exceptionThrown":
            details = params["exceptionDetails"]
            self.errors.append(details.get("exception", {}).get("description", details.get("text", "JS exception")))
        if method == "Runtime.consoleAPICalled" and params.get("type") in ["error", "assert"]:
            self.errors.append(" ".join(str(arg.get("value", arg.get("description", ""))) for arg in params.get("args", [])))

    def call(self, method, params=None):
        self.id += 1
        request_id = self.id
        self.send_frame(json.dumps(dict(id=request_id, method=method, params=params or {})).encode())
        while True:
            message = self.receive()
            self.event(message)
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(str(message["error"]))
                return message.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", dict(expression=expression, awaitPromise=True, returnByValue=True))
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"].get("text", "Evaluation failed") + ": " + result.get("result", {}).get("description", ""))
        return result.get("result", {}).get("value")

    def settle(self, seconds=.5):
        # Round trips drain Runtime events, including asynchronous exceptions.
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.evaluate("document.readyState")
            time.sleep(.05)

    def close(self):
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", "--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cookie", action="append", default=[])
    parser.add_argument("--click", action="append", default=[], help="CSS selector; repeated in order")
    parser.add_argument("--eval", action="append", default=[], help="JS expression, may return a promise; repeated in order")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--path", default="/usage")
    parser.add_argument("--expect", action="append", help="Override default usage panel headings")
    parser.add_argument("--temp-dir", type=Path, default=Path.home() / ".cache/cuteafd/builds/full-log/browser")
    args = parser.parse_args()
    # Keep disposable Chrome profiles off the checkout.
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    process = None
    client = None
    with tempfile.TemporaryDirectory(prefix="render-", dir=args.temp_dir, ignore_cleanup_errors=True) as profile:
        try:
            process = subprocess.Popen(["/usr/bin/google-chrome", "--headless=new", "--disable-gpu",
                "--disable-dev-shm-usage", "--disable-extensions", "--disable-background-networking", "--no-first-run", "--no-default-browser-check",
                "--enable-automation", "--password-store=basic", "--no-proxy-server", "--remote-debugging-port=0", "--remote-allow-origins=http://localhost", "--user-data-dir=" + profile,
                "--window-size=" + str(args.width) + "," + str(args.height), args.base_url.rstrip('/') + args.path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + args.timeout
            active = Path(profile) / "DevToolsActivePort"
            while not active.exists():
                if process.poll() is not None:
                    raise RuntimeError("Chrome exited before DevTools was ready")
                if time.monotonic() > deadline:
                    raise TimeoutError("Chrome DevTools startup timed out")
                time.sleep(.05)
            port = active.read_text().splitlines()[0]
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
                targets = json.load(response)
            target = next(t for t in targets if t["type"] == "page")
            client = CDP(target["webSocketDebuggerUrl"])
            for domain in ["Page", "Runtime"]:
                client.call(domain + ".enable")
            client.call("Emulation.setDeviceMetricsOverride", dict(width=args.width, height=args.height, deviceScaleFactor=1, mobile=args.width < 600))
            for cookie in args.cookie:
                name, value = cookie.split("=", 1)
                result = client.call("Network.setCookie", dict(name=name, value=value, url=args.base_url, path="/"))
                if not result.get("success"):
                    raise RuntimeError("Chrome rejected cookie " + name)
            client.call("Page.reload", dict(ignoreCache=True))
            while True:
                ready = client.evaluate("document.body?.dataset.usageReady === 'true'" if args.path == "/usage" else "document.readyState === 'complete'")
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Page did not settle before deadline")
                client.settle(.1)
            client.settle(.4)
            for selector in args.click:
                found = client.evaluate("(() => { const n = document.querySelector(" + json.dumps(selector) + "); if (!n) return false; n.click(); return true; })()")
                if not found:
                    raise RuntimeError("Missing click target " + selector)
                client.settle(.5)
            for expression in args.eval:
                print("eval: " + json.dumps(client.evaluate(expression)), flush=True)
                client.settle(.3)
            headings = client.evaluate("Array.from(document.querySelectorAll('h2')).map(n => n.textContent.trim())")
            expected = args.expect if args.expect is not None else PANELS if args.path == "/usage" else []
            missing = [h for h in expected if h not in headings]
            if missing:
                raise RuntimeError("Missing panel headings: " + ", ".join(missing))
            if args.path == "/usage":
                overflow = client.evaluate("document.documentElement.scrollWidth > innerWidth + 1")
                if overflow:
                    raise RuntimeError("Usage page overflows the viewport horizontally")
            metrics = client.call("Page.getLayoutMetrics")
            size = metrics.get("cssContentSize", metrics["contentSize"])
            shot = client.call("Page.captureScreenshot", dict(format="png", captureBeyondViewport=True,
                              clip=dict(x=0, y=0, width=size["width"], height=size["height"], scale=1)))
            args.out.write_bytes(base64.b64decode(shot["data"]))
            client.settle(.2)
            if client.errors:
                raise RuntimeError("JavaScript errors:\n" + "\n".join(client.errors))
            print(f"PASS {args.out}: {int(size['width'])} x {int(size['height'])}; {len(headings)} panel headings; no JavaScript errors", flush=True)
        finally:
            if client:
                client.close()
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()

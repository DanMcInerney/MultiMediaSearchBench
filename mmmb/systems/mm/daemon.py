"""The daemon of an mm home, as the benchmark starts and calls it (moved from the fixtures of tests/conftest.py of Multimem at fe0ece5,
which the old bench.py imported): `Daemon` runs `mm start` in a process of its own over a home, `Api` calls it, `write_config` writes
the home's config and approves the sources in it as `mm approve` does, and `wait_for` polls. A daemon is always over the home it is
given: its process gets MULTIMEM_HOME set to it, and its API is on a port the operating system picks and writes to the home's
`daemon.json`. Nothing here reads or writes ~/.multimem, and no code of this repository stops or calls a daemon it did not start."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from multimem import config as configuration
from multimem import consent


def env(home: Path) -> dict:
    out = dict(os.environ, MULTIMEM_HOME=str(home), PYTHONUTF8="1", HF_HUB_OFFLINE="1")  # the daemon never downloads
    out.pop("PYTHONPATH", None)
    return out


def mm(home: Path, *args: str, check: bool = True, timeout: float = 120, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run the CLI as the agent would, in its own process, in `cwd` (the home, which is in no project, unless given)."""
    done = subprocess.run([sys.executable, "-m", "multimem.cli", *args], env=env(home), capture_output=True,
                          text=True, encoding="utf-8", timeout=timeout, cwd=cwd or home)
    if check and done.returncode != 0:
        raise AssertionError(f"mm {' '.join(args)} exited {done.returncode}\n{done.stdout}\n{done.stderr}")
    return done


class ApiFailure(Exception):
    """An error the API gave: its status and its `error` object."""

    def __init__(self, status: int, error: dict):
        super().__init__(f"{status} {error}")
        self.status, self.error = status, error
        self.code, self.field = error["code"], error.get("field")


class Api:
    """The daemon's API as a test calls it: `daemon.json` is read at each call, so a restarted daemon is followed."""

    def __init__(self, home: Path):
        self.home = home

    def info(self) -> dict:
        return json.loads((self.home / "daemon.json").read_text(encoding="utf-8"))

    def raw(self, op: str, body=None, token: str | None = "", method: str = "POST") -> tuple[int, dict]:
        info = self.info()
        connection = http.client.HTTPConnection("127.0.0.1", info["port"], timeout=60)
        headers = {} if token is None else {"Authorization": f"Bearer {info['token'] if token == '' else token}"}
        try:
            connection.request(method, f"/v1/{op}", json.dumps(body if body is not None else {}).encode(), headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    def call(self, op: str, **body) -> dict:
        status, data = self.raw(op, body)
        if status != 200:
            raise ApiFailure(status, data["error"])
        return data

    def failure(self, op: str, **body) -> ApiFailure:
        """The error a call must give."""
        try:
            self.call(op, **body)
        except ApiFailure as error:
            return error
        raise AssertionError(f"{op} {body} did not fail")


class Daemon:
    """`mm start` in its own process; `stop()` asks it to stop with `mm stop` and waits."""

    def __init__(self, home: Path):
        self.home = home
        self.log = home / "daemon.test.log"
        self.process = None
        self.api = Api(home)

    def start(self, ready: bool = True) -> "Daemon":
        self._out = open(self.log, "ab")
        self.process = subprocess.Popen([sys.executable, "-m", "multimem.cli", "start"], env=env(self.home),
                                        stdout=self._out, stderr=subprocess.STDOUT)
        if ready:
            self.wait_ready()
        return self

    def wait_ready(self) -> None:
        def answers():
            if self.process.poll() is not None:
                raise AssertionError(f"the daemon exited {self.process.returncode}\n{self.text()}")
            try:
                return self.api.call("status")
            except (OSError, ValueError, ApiFailure):
                return None
        wait_for(answers, timeout=60, what="the daemon's API", detail=self.text)

    def stop(self, timeout: float = 30) -> int:
        if self.process is None or self.process.poll() is not None:
            return self.process.returncode if self.process else 0
        mm(self.home, "stop", check=False)
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._out.close()
        return self.process.returncode

    def kill(self) -> None:
        self.process.kill()
        self.process.wait()
        self._out.close()

    def text(self) -> str:
        return self.log.read_text(encoding="utf-8", errors="replace") if self.log.exists() else ""


def write_config(home: Path, text: str, approve: bool = True) -> None:
    """Write `multimem.toml`, and approve every source in it that needs it (as `mm approve` does once the owner has
    typed the name), unless told not to."""
    (home / "multimem.toml").write_text(text, encoding="utf-8")
    if approve:
        for name, source in configuration.load(home).sources.items():
            if source.what() is not None:
                consent.record(home, name, source.what())


def wait_for(check, timeout: float = 20, every: float = 0.1, what: str = "the condition", detail=None):
    """Poll `check` until it returns something truthy, and return that. On a timeout the message adds `detail()`,
    such as a daemon's log."""
    end = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > end:
            raise AssertionError(f"timed out after {timeout:g} s waiting for {what}"
                                 + (f"\n{detail()}" if detail else ""))
        time.sleep(every)

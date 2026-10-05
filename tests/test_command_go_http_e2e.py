"""Genuine local HTTP E2E for Command API + OpenCode Go routes.

Real TestClient POSTs to /v1/chat/completions hit REAL provider call
functions; upstream is stdlib ThreadingHTTPServer on 127.0.0.1:0.
No provider-function mocks, no external network, dummy keys only.
"""
import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proxy_internal.providers.command_code as cmd_mod  # noqa: E402

CMD_MODEL = "command_api:deepseek/deepseek-v4-flash"
GO_MODEL = "opencode-go:glm-5.3"


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []  # dicts: method, path, headers, body


def _make_server(handler_cls, state):
    handler_cls.state = state
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    srv.timeout = 5
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    return srv, thread


def _read_body(handler):
    length = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(length) if length else b""


def _send_json(handler, code, obj):
    raw = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _isolate_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    for var in ("OPENCODE_API_KEY", "COMMANDCODE_API_URL"):
        monkeypatch.delenv(var, raising=False)


class TestCommandApiHttpE2E:
    def test_command_api_nonstream(self, test_client, proxy_module, monkeypatch, tmp_path):
        _isolate_home(monkeypatch, tmp_path)
        monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
        monkeypatch.setattr(proxy_module, "command_code_api_key", "dummy-cmd-key", raising=False)
        monkeypatch.setattr(proxy_module, "_resolve_command_code_api_key", lambda: "dummy-cmd-key", raising=False)

        state = _State()

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = _read_body(self)
                with self.state.lock:
                    self.state.requests.append({
                        "method": "POST", "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    })
                _send_json(self, 200, {
                    "choices": [{"message": {"content": "cmd-e2e-hello"}}],
                })

            def log_message(self, *a):
                pass

        srv, thread = _make_server(H, state)
        try:
            monkeypatch.setattr(
                cmd_mod, "COMMAND_CODE_API_CHAT_URL",
                f"http://127.0.0.1:{srv.server_port}/provider/v1/chat/completions",
            )
            resp = test_client.post("/v1/chat/completions", json={
                "model": CMD_MODEL,
                "messages": [
                    {"role": "system", "content": "sys-ping"},
                    {"role": "user", "content": "hello-e2e"},
                ],
                "stream": False,
            })
        finally:
            srv.shutdown()
            thread.join(timeout=5)
            srv.server_close()

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "cmd-e2e-hello"
        assert data["model"] == CMD_MODEL
        assert "dummy-cmd-key" not in resp.text
        assert len(state.requests) == 1
        seen = state.requests[0]
        assert seen["headers"].get("authorization") == "Bearer dummy-cmd-key"
        payload = json.loads(seen["body"].decode())
        assert payload["model"] == "deepseek/deepseek-v4-flash"
        roles = [m["role"] for m in payload["messages"]]
        assert roles[0] == "system" and payload["messages"][0]["content"] == "sys-ping"
        assert any(m["role"] == "user" and "hello-e2e" in str(m["content"]) for m in payload["messages"])


class TestOpenCodeGoHttpE2E:
    def _go_server(self, monkeypatch, proxy_module, tmp_path, mode):
        _isolate_home(monkeypatch, tmp_path)
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        monkeypatch.setattr(proxy_module, "opencode_go_api_key", "dummy-go-key", raising=False)
        monkeypatch.setattr(proxy_module, "OPENCODE_GO_URL",
                            f"http://127.0.0.1:0-PLACEHOLDER", raising=False)
        state = _State()

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = _read_body(self)
                with self.state.lock:
                    self.state.requests.append({
                        "method": "POST", "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    })
                try:
                    payload = json.loads(body.decode() or "{}")
                except ValueError:
                    payload = {}
                if payload.get("stream"):
                    chunks = [
                        b'data: {"id":"c","object":"chat.completion.chunk","choices":[{"delta":{"content":"go-"}}]}\n\n',
                        b'data: {"id":"c","object":"chat.completion.chunk","choices":[{"delta":{"content":"stream"}}]}\n\n',
                        b"data: [DONE]\n\n",
                    ]
                    raw = b"".join(chunks)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                else:
                    _send_json(self, 200, {
                        "choices": [{"message": {"content": "go-e2e-hello"}}],
                    })

            def log_message(self, *a):
                pass

        srv, thread = _make_server(H, state)
        monkeypatch.setattr(proxy_module, "OPENCODE_GO_URL",
                            f"http://127.0.0.1:{srv.server_port}", raising=False)
        return srv, thread, state

    def test_go_nonstream(self, test_client, proxy_module, monkeypatch, tmp_path):
        srv, thread, state = self._go_server(monkeypatch, proxy_module, tmp_path, "direct")
        try:
            resp = test_client.post("/v1/chat/completions", json={
                "model": GO_MODEL,
                "messages": [{"role": "user", "content": "hi-go"}],
                "stream": False,
            }, headers={"x-opencode-session": "sess-stable-1"})
        finally:
            srv.shutdown()
            thread.join(timeout=5)
            srv.server_close()
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "go-e2e-hello"
        assert "dummy-go-key" not in resp.text
        assert len(state.requests) == 1
        seen = state.requests[0]
        assert seen["headers"].get("authorization") == "Bearer dummy-go-key"
        assert seen["headers"].get("user-agent") == "skyrimnet-codex-proxy/1.0"
        assert seen["headers"].get("x-opencode-session") == "sess-stable-1"
        payload = json.loads(seen["body"].decode())
        assert payload["model"] == "glm-5.3"
        assert "x-opencode-session" not in json.dumps(payload)

    def test_go_stream(self, test_client, proxy_module, monkeypatch, tmp_path):
        srv, thread, state = self._go_server(monkeypatch, proxy_module, tmp_path, "stream")
        try:
            resp = test_client.post("/v1/chat/completions", json={
                "model": GO_MODEL,
                "messages": [{"role": "user", "content": "hi-go-stream"}],
                "stream": True,
            }, headers={"x-opencode-session": "sess-stable-1"})
        finally:
            srv.shutdown()
            thread.join(timeout=5)
            srv.server_close()
        assert resp.status_code == 200
        assert "go-" in resp.text and "stream" in resp.text
        assert "[DONE]" in resp.text
        assert len(state.requests) == 1
        seen = state.requests[0]
        assert seen["headers"].get("x-opencode-session") == "sess-stable-1"
        assert seen["headers"].get("user-agent") == "skyrimnet-codex-proxy/1.0"


class TestCommandApi403CliFallbackHttpE2E:
    def test_upgrade_required_falls_back_through_real_cli(self, test_client, proxy_module, monkeypatch, tmp_path):
        _isolate_home(monkeypatch, tmp_path)
        monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
        monkeypatch.setattr(proxy_module, "command_code_api_key", "dummy-cmd-key", raising=False)
        monkeypatch.setattr(proxy_module, "_resolve_command_code_api_key", lambda: "dummy-cmd-key", raising=False)

        api_state = _State()
        cli_state = _State()

        class ApiH(BaseHTTPRequestHandler):
            def do_POST(self):
                body = _read_body(self)
                with self.state.lock:
                    self.state.requests.append({"path": self.path, "body": body})
                _send_json(self, 403, {"code": "upgrade_required"})

            def log_message(self, *a):
                pass

        class FinalH(BaseHTTPRequestHandler):
            def do_POST(self):
                body = _read_body(self)
                with self.state.lock:
                    self.state.requests.append({
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    })
                _send_json(self, 200, {"ok": True})

            def log_message(self, *a):
                pass

        api_srv, api_thread = _make_server(ApiH, api_state)
        final_srv, final_thread = _make_server(FinalH, cli_state)
        try:
            monkeypatch.setattr(
                cmd_mod, "COMMAND_CODE_API_CHAT_URL",
                f"http://127.0.0.1:{api_srv.server_port}/provider/v1/chat/completions",
            )
            monkeypatch.setattr(cmd_mod, "_CMD_API_BASE", f"http://127.0.0.1:{final_srv.server_port}")

            argv_log = tmp_path / "argv.json"
            stdin_log = tmp_path / "stdin.txt"
            fake_cli = tmp_path / "cmd"
            fake_cli.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys, urllib.request\n"
                "data = sys.stdin.buffer.read()\n"
                "open(os.environ['FAKE_ARGV_LOG'], 'w').write(json.dumps(sys.argv))\n"
                "open(os.environ['FAKE_STDIN_LOG'], 'wb').write(data)\n"
                "prompt = data.decode('utf-8', errors='replace')\n"
                "base = os.environ['COMMANDCODE_API_URL']\n"
                "payload = {\n"
                "  'skills': ['cli-skill'], 'taste': 'x', 'system_prompt_breakdown': 'y',\n"
                "  'params': {\n"
                "    'messages': [{'role': 'user', 'content': prompt}],\n"
                "    'tools': [{'name': 'evil-tool'}], 'system': 'cli-injected-system',\n"
                "  },\n"
                "}\n"
                "req = urllib.request.Request(base + '/alpha/generate',\n"
                "  data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})\n"
                "urllib.request.urlopen(req, timeout=10).read()\n"
                "sys.stdout.write('fallback-hello')\n"
            )
            fake_cli.chmod(fake_cli.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            monkeypatch.setenv("FAKE_ARGV_LOG", str(argv_log))
            monkeypatch.setenv("FAKE_STDIN_LOG", str(stdin_log))
            monkeypatch.setattr(proxy_module, "COMMAND_CODE_PATH", str(fake_cli), raising=False)

            resp = test_client.post("/v1/chat/completions", json={
                "model": CMD_MODEL,
                "messages": [{"role": "user", "content": "need-fallback"}],
                "stream": False,
            })
        finally:
            api_srv.shutdown()
            final_srv.shutdown()
            api_thread.join(timeout=5)
            final_thread.join(timeout=5)
            api_srv.server_close()
            final_srv.server_close()

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "fallback-hello"
        assert len(api_state.requests) == 1
        assert len(cli_state.requests) == 1

        argv = json.loads(argv_log.read_text())
        argv_text = " ".join(argv)
        assert argv[0] == str(fake_cli)
        assert "-m" in argv and "deepseek/deepseek-v4-flash" in argv_text
        stdin_text = stdin_log.read_text()
        assert "need-fallback" in stdin_text
        assert stdin_text not in argv_text  # full prompt via stdin, not argv

        forwarded = json.loads(cli_state.requests[0]["body"].decode())
        assert "skills" not in forwarded
        assert "taste" not in forwarded
        assert "system_prompt_breakdown" not in forwarded
        params = forwarded.get("params", {})
        assert "tools" not in params
        assert "system" not in params
        msgs = params.get("messages") or forwarded.get("messages")
        assert any(m.get("role") == "user" and "need-fallback" in str(m.get("content")) for m in msgs)

from __future__ import annotations

import json
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

from ergon_studio.debug_log import configure_debug_logging, disable_debug_logging
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.server import ProxyServerHandle, start_proxy_server_in_thread
from ergon_studio.registry import load_registry
from ergon_studio.upstream import UpstreamSettings, probe_upstream_models


@dataclass(frozen=True)
class RealE2EConfig:
    upstream_base_url: str
    model: str
    upstream_api_key: str | None = None


class RealProxyE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._repo_root = Path(__file__).resolve().parents[1]
        cls._config = _load_real_e2e_config(cls._repo_root / ".env.e2e-tests")
        _ensure_upstream_available(cls._config)
        registry = load_registry(
            cls._repo_root / "ergon_studio" / "default_definitions",
            upstream=UpstreamSettings(
                base_url=cls._config.upstream_base_url,
                api_key=cls._config.upstream_api_key,
            ),
        )
        cls._handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=ProxyOrchestrationCore(registry),
        )
        cls._client = OpenAI(
            api_key=cls._config.upstream_api_key or "test",
            base_url=f"http://127.0.0.1:{cls._handle.port}/v1",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        handle = getattr(cls, "_handle", None)
        if handle is not None:
            handle.close()
        super().tearDownClass()

    _handle: ProxyServerHandle
    _client: OpenAI
    _config: RealE2EConfig
    _repo_root: Path

    def test_chat_completion_direct_answer_uses_real_model(self) -> None:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly PONG and nothing else.",
                }
            ],
        )

        content = _normalize_scalar_reply(response.choices[0].message.content)
        self.assertEqual(content, "PONG")
        self.assertEqual(response.model, self._config.model)

    def test_chat_completion_tool_loop_runs_against_real_upstream(self) -> None:
        user_prompt = (
            "Use the read_file tool on README.md. "
            "After reading it, answer with exactly YES if README.md says "
            "'OpenAI-compatible orchestration proxy for local coding models', "
            "otherwise answer with exactly NO."
        )
        tool_schema = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read.",
                        }
                    },
                    "required": ["path"],
                },
            },
        }

        first_response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool_schema],
            tool_choice="required",
        )

        assistant_message = first_response.choices[0].message
        self.assertEqual(first_response.choices[0].finish_reason, "tool_calls")
        self.assertIsNotNone(assistant_message.tool_calls)
        tool_call = assistant_message.tool_calls[0]

        tool_args = _parse_tool_arguments(tool_call.function.arguments)
        normalized_arguments = json.dumps(tool_args, separators=(",", ":"))
        tool_result = _read_repo_file(self._repo_root, str(tool_args["path"]))
        second_response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": normalized_arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                },
            ],
            tool_choice="none",
        )

        content = _normalize_scalar_reply(second_response.choices[0].message.content)
        self.assertEqual(content, "YES")

    def test_responses_direct_answer_uses_real_model(self) -> None:
        response = self._client.responses.create(
            model=self._config.model,
            input="Reply with exactly GAMMA and nothing else.",
        )

        content = _normalize_scalar_reply(response.output_text)
        self.assertEqual(content, "GAMMA")

    def test_orchestrator_builds_project_from_brief(self) -> None:
        """Real end-to-end orchestration run.

        Sends a product brief to the orchestrator with no instructions on *how*
        to build it.  The orchestrator decides what channels to open, which
        agents to involve, and what to write to disk.  Nothing is scripted —
        this is a live run against the configured upstream model.

        After the test completes, inspect:
          ~/.ergon-workspace/<session_id>/run.log   — every orchestration event
          ~/.ergon-workspace/<session_id>-parallel-*/  — files the agents wrote
        """
        session_id = f"tictactoe-e2e-{int(time.time())}"
        workspace_root = Path.home() / ".ergon-workspace"
        session_dir = workspace_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        log_path = configure_debug_logging(path=session_dir / "run.log")

        print(f"\n{'='*60}")
        print(f"Session : {session_id}")
        print(f"Log     : {log_path}")
        print(f"Workspace: {workspace_root}/{session_id}-parallel-*")
        print(f"{'='*60}\n")

        brief = (
            "Build a tic-tac-toe game in JavaScript for two players. "
            "It should support alternating turns between X and O, "
            "display the 3×3 board after each move, detect wins and draws, "
            "and run in the terminal via Node.js."
        )

        try:
            stream = self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": brief}],
                stream=True,
                extra_headers={"X-Ergon-Session": session_id},
                timeout=1200,
            )

            finish_reason: str | None = None
            final_content: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                reasoning = getattr(delta, "reasoning", None) or getattr(
                    delta, "reasoning_content", None
                )
                if reasoning:
                    print(reasoning, end="", flush=True)
                if delta.content:
                    final_content.append(delta.content)
                    print(delta.content, end="", flush=True)

        finally:
            disable_debug_logging()

        print(f"\n\n{'='*60}")
        print(f"Finish reason: {finish_reason}")

        # List every file written to the workspace by sub-sessions.
        written: list[Path] = sorted(
            p
            for p in workspace_root.glob(f"{session_id}-parallel-*/**/*")
            if p.is_file()
        )
        print(f"Files written ({len(written)}):")
        for p in written:
            rel = p.relative_to(workspace_root)
            size = p.stat().st_size
            print(f"  {rel}  ({size} bytes)")

        print(f"\nLog: {log_path}")
        print(f"{'='*60}\n")

        self.assertNotEqual(
            finish_reason,
            "error",
            "orchestration turn ended with finish_reason='error' — check log",
        )


def _load_real_e2e_config(path: Path) -> RealE2EConfig:
    if not path.exists():
        raise unittest.SkipTest(f"missing real E2E config file: {path}")
    raw_values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"invalid line in {path}: {raw_line!r}")
        raw_values[key.strip()] = value.strip()
    upstream_base_url = raw_values.get("UPSTREAM_BASE_URL")
    model = raw_values.get("MODEL")
    if not upstream_base_url:
        raise ValueError(f"{path} must define UPSTREAM_BASE_URL")
    if not model:
        raise ValueError(f"{path} must define MODEL")
    return RealE2EConfig(
        upstream_base_url=upstream_base_url,
        model=model,
        upstream_api_key=raw_values.get("UPSTREAM_API_KEY") or None,
    )


def _ensure_upstream_available(config: RealE2EConfig) -> None:
    parsed = urlparse(config.upstream_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise unittest.SkipTest(
            f"invalid upstream URL in .env.e2e-tests: {config.upstream_base_url}"
        )
    try:
        models = probe_upstream_models(
            UpstreamSettings(
                base_url=config.upstream_base_url,
                api_key=config.upstream_api_key,
            ),
            timeout=3,
        )
    except Exception as exc:
        raise unittest.SkipTest(
            f"upstream unavailable for real E2E tests: {exc}"
        ) from exc
    if models and config.model not in {str(entry.get('id', '')) for entry in models}:
        raise unittest.SkipTest(
            f"configured real E2E model not advertised by upstream: {config.model}"
        )


def _read_repo_file(repo_root: Path, relative_path: str) -> str:
    path = (repo_root / relative_path).resolve()
    if repo_root.resolve() not in path.parents and path != repo_root.resolve():
        raise AssertionError(f"tool requested path outside repo: {relative_path}")
    return path.read_text(encoding="utf-8")


def _normalize_scalar_reply(value: str | None) -> str:
    text = (value or "").strip()
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _parse_tool_arguments(raw: str) -> dict[str, object]:
    payload, _end = json.JSONDecoder().raw_decode(raw.strip())
    if not isinstance(payload, dict):
        raise AssertionError(f"tool arguments were not a JSON object: {raw!r}")
    return payload

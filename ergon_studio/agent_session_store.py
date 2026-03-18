from __future__ import annotations

import json
from collections.abc import Callable

from agent_framework import AgentSession

from ergon_studio.paths import StudioPaths


class AgentSessionStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths

    def session_path(self, *, thread_id: str, agent_id: str) -> str:
        return str(self.paths.sessions_dir / thread_id / f"{agent_id}.json")

    def load_session(self, *, thread_id: str, agent_id: str) -> AgentSession | None:
        path = self.paths.sessions_dir / thread_id / f"{agent_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentSession.from_dict(data)

    def load_or_create_session(
        self,
        *,
        thread_id: str,
        agent_id: str,
        session_factory: Callable[[str], AgentSession],
    ) -> AgentSession:
        existing = self.load_session(thread_id=thread_id, agent_id=agent_id)
        if existing is not None:
            return existing
        return session_factory(self._session_id(thread_id=thread_id, agent_id=agent_id))

    def save_session(self, *, thread_id: str, agent_id: str, session: AgentSession) -> None:
        path = self.paths.sessions_dir / thread_id / f"{agent_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(session.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _session_id(*, thread_id: str, agent_id: str) -> str:
        return f"{thread_id}:{agent_id}"

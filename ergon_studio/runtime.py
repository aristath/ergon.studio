from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.agent_factory import build_agent
from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.paths import StudioPaths
from ergon_studio.registry import RuntimeRegistry, load_registry
from ergon_studio.storage.models import MessageRecord, ThreadRecord
from ergon_studio.tool_registry import build_workspace_tool_registry


MAIN_SESSION_ID = "session-main"
MAIN_THREAD_ID = "thread-main"


@dataclass(frozen=True)
class RuntimeContext:
    paths: StudioPaths
    registry: RuntimeRegistry
    tool_registry: dict[str, object]
    conversation_store: ConversationStore
    main_session_id: str = MAIN_SESSION_ID
    main_thread_id: str = MAIN_THREAD_ID

    def build_agent(self, agent_id: str):
        return build_agent(self.registry, agent_id, tool_registry=self.tool_registry)

    def list_threads(self) -> list[ThreadRecord]:
        return self.conversation_store.list_threads(self.main_session_id)

    def list_main_messages(self) -> list[MessageRecord]:
        return self.conversation_store.list_messages(self.main_thread_id)

    def append_message_to_main_thread(
        self,
        *,
        message_id: str,
        sender: str,
        kind: str,
        body: str,
        created_at: int,
    ) -> MessageRecord:
        self.ensure_main_conversation()
        return self.conversation_store.append_message(
            thread_id=self.main_thread_id,
            message_id=message_id,
            sender=sender,
            kind=kind,
            body=body,
            created_at=created_at,
        )

    def ensure_main_conversation(self) -> None:
        self.conversation_store.ensure_session(self.main_session_id)
        self.conversation_store.ensure_thread(
            session_id=self.main_session_id,
            thread_id=self.main_thread_id,
            kind="main",
        )


def load_runtime(project_root: Path, home_dir: Path) -> RuntimeContext:
    paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
    registry = load_registry(paths)
    tool_registry = build_workspace_tool_registry(paths.project_root)
    conversation_store = ConversationStore(paths)
    runtime = RuntimeContext(
        paths=paths,
        registry=registry,
        tool_registry=tool_registry,
        conversation_store=conversation_store,
    )
    runtime.ensure_main_conversation()
    return runtime

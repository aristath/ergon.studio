from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StaffedParticipant:
    agent_id: str
    label: str
    instance_index: int
    total_instances: int


def expand_staffed_participants(
    base_participants: tuple[str, ...],
    *,
    participants: tuple[str, ...] = (),
) -> tuple[StaffedParticipant, ...]:
    count_map = _participant_counts(participants)
    allowed = set(count_map) if count_map else None
    staffed_participants: list[StaffedParticipant] = []
    seen_agents: set[str] = set()
    for agent_id in base_participants:
        if agent_id in seen_agents:
            continue
        seen_agents.add(agent_id)
        if allowed is not None and agent_id not in allowed:
            continue
        total_instances = count_map.get(agent_id, 1) if count_map else 1
        for instance_index in range(1, total_instances + 1):
            staffed_participants.append(
                StaffedParticipant(
                    agent_id=agent_id,
                    label=_participant_label(
                        agent_id=agent_id,
                        instance_index=instance_index,
                        total_instances=total_instances,
                    ),
                    instance_index=instance_index,
                    total_instances=total_instances,
                )
            )
    return tuple(staffed_participants)


def expand_staffed_sequence(
    base_sequence: tuple[str, ...],
    *,
    participants: tuple[StaffedParticipant, ...],
) -> tuple[str, ...]:
    if not base_sequence or not participants:
        return ()
    labels_by_agent: dict[str, list[str]] = {}
    for participant in participants:
        labels_by_agent.setdefault(participant.agent_id, []).append(participant.label)
    expanded: list[str] = []
    for agent_id in base_sequence:
        expanded.extend(labels_by_agent.get(agent_id, ()))
    return tuple(expanded)


def participant_by_label(
    participants: tuple[StaffedParticipant, ...],
    label: str | None,
) -> StaffedParticipant | None:
    if label is None:
        return None
    for participant in participants:
        if participant.label == label:
            return participant
    return None


def participant_for_agent(
    participants: tuple[StaffedParticipant, ...],
    agent_id: str | None,
) -> StaffedParticipant | None:
    if agent_id is None:
        return None
    for participant in participants:
        if participant.agent_id == agent_id:
            return participant
    return None


def participant_context(participant: StaffedParticipant) -> str | None:
    if participant.total_instances <= 1:
        return None
    return (
        f"You are instance {participant.instance_index} of "
        f"{participant.total_instances} staffed {participant.agent_id}s for this "
        "round. Produce an independently useful contribution."
    )


def participant_labels_for_agents(
    participants: tuple[StaffedParticipant, ...],
    agent_ids: tuple[str, ...],
) -> tuple[str, ...]:
    allowed = set(agent_ids)
    return tuple(
        participant.label
        for participant in participants
        if participant.agent_id in allowed
    )


def _participant_label(
    *,
    agent_id: str,
    instance_index: int,
    total_instances: int,
) -> str:
    if total_instances <= 1:
        return agent_id
    return f"{agent_id}[{instance_index}]"


def _participant_counts(
    participants: tuple[str, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent_id in participants:
        counts[agent_id] = counts.get(agent_id, 0) + 1
    return counts

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
    base_counts = _participant_counts(base_participants)
    staffed_participants: list[StaffedParticipant] = []
    emitted_counts: dict[str, int] = {}
    for agent_id in base_participants:
        if allowed is not None and agent_id not in allowed:
            continue
        total_instances = (
            count_map.get(agent_id, 0) if count_map else base_counts.get(agent_id, 0)
        )
        if total_instances <= 0:
            continue
        instance_index = emitted_counts.get(agent_id, 0) + 1
        emitted_counts[agent_id] = instance_index
        if instance_index > total_instances:
            continue
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
    if count_map:
        for agent_id, total_instances in count_map.items():
            if allowed is None or agent_id not in allowed:
                continue
            already_emitted = emitted_counts.get(agent_id, 0)
            for instance_index in range(already_emitted + 1, total_instances + 1):
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


def participant_context(participant: StaffedParticipant) -> str | None:
    if participant.total_instances <= 1:
        return None
    return (
        f"You are instance {participant.instance_index} of "
        f"{participant.total_instances} staffed {participant.agent_id}s in this "
        "channel. Produce a distinct, independently useful contribution."
    )


def resolve_staffed_recipients(
    *,
    staffed_members: tuple[StaffedParticipant, ...],
    recipients: tuple[str, ...],
) -> tuple[StaffedParticipant, ...]:
    if not recipients:
        return ()

    exact_remaining: dict[str, int] = {}
    agent_remaining: dict[str, int] = {}
    for recipient in recipients:
        if "[" in recipient and recipient.endswith("]"):
            exact_remaining[recipient] = exact_remaining.get(recipient, 0) + 1
            continue
        agent_remaining[recipient] = agent_remaining.get(recipient, 0) + 1

    selected: list[StaffedParticipant] = []
    for participant in staffed_members:
        exact_count = exact_remaining.get(participant.label, 0)
        if exact_count > 0:
            selected.append(participant)
            exact_remaining[participant.label] = exact_count - 1
            continue
        agent_count = agent_remaining.get(participant.agent_id, 0)
        if agent_count <= 0:
            continue
        selected.append(participant)
        agent_remaining[participant.agent_id] = agent_count - 1
    return tuple(selected)


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

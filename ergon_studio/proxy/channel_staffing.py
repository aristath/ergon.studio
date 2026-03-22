from __future__ import annotations

from collections import Counter
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
    _validate_duplicate_recipient_selection(
        staffed_members=staffed_members,
        recipients=recipients,
    )
    remaining = list(staffed_members)
    selected: list[StaffedParticipant] = []
    for recipient in recipients:
        match_index = -1
        for index, participant in enumerate(remaining):
            if recipient == participant.label or recipient == participant.agent_id:
                match_index = index
                break
        if match_index < 0:
            continue
        selected.append(remaining.pop(match_index))
    return tuple(selected)


def require_staffed_recipients(
    *,
    staffed_members: tuple[StaffedParticipant, ...],
    recipients: tuple[str, ...],
) -> tuple[StaffedParticipant, ...]:
    selected = resolve_staffed_recipients(
        staffed_members=staffed_members,
        recipients=recipients,
    )
    if len(selected) == len(recipients):
        return selected

    available: set[str] = set()
    for participant in staffed_members:
        available.add(participant.agent_id)
        available.add(participant.label)
    invalid = [recipient for recipient in recipients if recipient not in available]
    if not invalid:
        invalid = list(recipients)
    raise ValueError(
        "channel recipients are not staffed in this channel: "
        + ", ".join(sorted(invalid))
    )


def _validate_duplicate_recipient_selection(
    *,
    staffed_members: tuple[StaffedParticipant, ...],
    recipients: tuple[str, ...],
) -> None:
    repeated_roles = Counter(
        participant.agent_id
        for participant in staffed_members
        if participant.total_instances > 1
    )
    if not repeated_roles:
        return

    labels_to_roles = {
        participant.label: participant.agent_id for participant in staffed_members
    }
    generic_counts: Counter[str] = Counter()
    explicit_counts: Counter[str] = Counter()
    for recipient in recipients:
        role = labels_to_roles.get(recipient)
        if role is not None and recipient != role:
            explicit_counts[role] += 1
            continue
        generic_counts[recipient] += 1

    for role, total_instances in repeated_roles.items():
        generic_count = generic_counts.get(role, 0)
        if not generic_count:
            continue
        if explicit_counts.get(role, 0) or generic_count != total_instances:
            raise ValueError(
                "duplicate staffed recipients must be addressed explicitly for "
                f"{role}: use staffed labels like {role}[1]"
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

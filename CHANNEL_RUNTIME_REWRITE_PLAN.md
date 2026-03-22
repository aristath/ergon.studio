# Channel Runtime Rewrite Plan

## Goal

Make collaboration behave exactly like this:

- The orchestrator opens a channel to one or more teammates.
- They talk naturally.
- Tools are just passthrough for the speaking agent.
- The orchestrator closes the channel when done.

No workflow engine. No turn scheduler. No completion markers. No hidden handoff logic.

## Core Principle

The LLMs decide behavior.

The runtime only carries:

- channels
- messages
- tool routing
- pending tool resumptions
- transport adaptation

If a piece of code decides:

- who should speak next
- whether someone is done
- whether a collaboration pattern is "best-of-n"
- whether control should hand back now

that code should be deleted.

## Message Model

There are 2 separate conversations:

1. PM <-> orchestrator
2. orchestrator <-> teammates inside channels

Inside channels:

- the orchestrator is the `user`
- participants are the agents

When invoking a participant from channel history:

- orchestrator messages serialize as `user`
- the current participant's own prior messages serialize as `assistant`
- other participants' messages are also included in structured history
- use upstream `name` support when available
- if `name` is unavailable, preserve authorship in message content, but still as message history, not prompt summary

The channel transcript itself is the source of truth.

## Runtime Scope

The runtime is allowed to:

- open a channel
- store channel membership
- append messages to the transcript
- invoke the explicitly addressed participant
- pass host tool calls through
- resume the exact participant that asked for a tool
- close a channel

The runtime is not allowed to:

- auto-fan-out to everyone in the channel
- invent speaker order
- force turns or rounds
- require `COMPLETE`, `BLOCKED`, or similar markers
- infer task completion from heuristics
- stuff collaboration policy into prompts

## Channel Rules

### Channels are first-class objects

Channels exist because they were opened.

They must not be recovered from conversation-history hashing.

Required:

- server-side `ChannelStore`
- explicit `channel_id`
- explicit open / message / close lifecycle

Delete:

- `_conversation_sessions`
- `_conversation_key(...)`
- `_parent_conversation_key(...)`
- any channel recovery keyed by transcript hash

### Messages target explicit recipients

A channel may have multiple participants.

That supports conference calls.

But the runtime must never decide that "everyone replies now".

Rules:

- a channel can include many participants
- each message explicitly names the recipient or recipients
- only the explicitly named recipients are invoked
- nobody else is auto-invoked

This keeps conference calls while removing hidden scheduling.

### Best-of-N is separate channels

Best-of-N should not be one shared channel.

If multiple workers can see each other, their attempts are no longer independent.

So best-of-N should be modeled as:

- multiple separate channels
- same assignment
- isolated transcripts
- orchestrator compares results afterward

This should not be a runtime mode.

It should emerge from how the orchestrator chooses to open channels.

## Tool Continuations

Tool-call ids must stop carrying application state.

Replace stateful continuation payloads with opaque pending handles.

Required:

- server-side `PendingStore`
- opaque `pending_id`
- tool-call ids contain only:
  - the opaque pending handle
  - the original tool call id

Pending state stores:

- actor
- channel_id
- tool-call metadata
- minimal resume context

Delete:

- embedded channel snapshots in tool-call ids
- embedded worklog tails in tool-call ids

## Prompt Rules

Prompts should provide only lightweight framing:

- role identity
- minimal task context if needed
- tool-use reminder if needed

Prompts must not:

- restate the transcript as prose
- restate team history as the source of truth
- encode process instructions about how to hand work back

The conversation history should do the real work.

## Tool Surface

Target internal tools:

- `open_channel(participants, recipients, message)`
- `message_channel(channel, recipients, message)`
- `close_channel(channel)`

Notes:

- `participants` defines who is on the call
- `recipients` defines who this specific message is addressed to
- `recipients` may be one or many
- the runtime invokes only the explicitly named recipients

Participants should also be able to use `message_channel(...)`.

That allows:

- teammate-to-teammate messaging in the same channel
- natural escalation
- natural collaboration without workflow markers

## Target Runtime Shape

### Channel store

`Channel`

- `channel_id`
- `name`
- `participants`
- `messages`
- `open`

`ChannelMessage`

- `author`
- `content`

Potentially also:

- `kind`
- `tool_call`
- `tool_result`

if needed for transport fidelity

### Pending store

`PendingCall`

- `pending_id`
- `channel_id`
- `actor`
- `assistant_tool_call`
- minimal resume payload

## Execution Model

### Opening a channel

- orchestrator calls `open_channel(...)`
- runtime creates channel
- runtime appends orchestrator message to channel transcript
- runtime invokes only the addressed recipient(s)

### Messaging a channel

- sender calls `message_channel(...)`
- runtime appends the message to the transcript
- runtime invokes only the addressed recipient(s)

### Tool use

- current speaking participant uses host tools
- runtime creates an opaque pending handle
- tool result resumes only that same participant in that same channel

### Closing a channel

- orchestrator calls `close_channel(...)`
- runtime marks it closed / removes it from open store
- no further messages allowed to that channel

## Deletions Required

Delete all remaining code that does any of the following:

- recipient fan-out by default
- speaker selection
- turn or round execution
- completion or handoff keywords
- conversation-hash-based channel persistence
- worklog/state embedded in continuation ids
- prompt-based transcript reconstruction as a correctness dependency

## Test Plan

Rewrite tests around the actual channel contract.

Must-have tests:

- opening a channel returns a stable channel id
- channels persist until explicitly closed
- messaging a channel invokes only the named recipient(s)
- orchestrator messages appear as `user` to participants
- a participant sees their own prior replies as `assistant`
- other participant messages are included in structured history
- tool resumption resumes only the originating participant
- participants can message another participant in the same channel
- no path depends on conversation-history hashing
- no path depends on application state embedded in tool-call ids
- closed channels reject further messages cleanly

## Execution Order

1. Delete fan-out and hidden speaker-selection behavior.
2. Delete conversation-hash channel persistence.
3. Delete embedded-state continuation tokens.
4. Add `ChannelStore` and `PendingStore`.
5. Rebuild message-role serialization around the orchestrator-as-`user` contract.
6. Make `message_channel(...)` available to participants too.
7. Rebuild channel execution as a thin router.
8. Rewrite tests around explicit channel handles and opaque pending ids.
9. Clean up prompts and naming after behavior is correct.

## Done Criteria

We are done when all of the following are true:

- the runtime never chooses the next speaker
- the orchestrator is truly the `user` inside team channels
- participants are invoked from real channel history, not prompt summaries
- tool-call ids are opaque handles, not serialized state containers
- channels persist until explicitly closed
- participants can talk naturally without engine-owned handoff behavior
- conference calls work without auto-fan-out
- best-of-N comes from multiple separate channels, not a runtime mode

If any collaboration behavior still depends on hidden runtime judgment, we are not done.

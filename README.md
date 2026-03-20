# ergon.studio

`ergon.studio` is an OpenAI-compatible orchestration proxy for local coding models.

It sits in front of an existing host client and turns one model request into an orchestrated multi-agent turn:

- the orchestrator plans the turn
- specialists handle focused sub-work
- host-provided tools are passed through unchanged
- the proxy streams orchestration worklog plus final assistant output

The host keeps the UI, sessions, tools, MCPs, and approvals.

`ergon.studio` keeps the orchestration brain.

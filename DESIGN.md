# ergon.studio — Product Design

## What This Product Is

`ergon` is a transparent orchestration proxy for AI software teams.

To the host client, it looks like a normal OpenAI-compatible model endpoint.
To the user, it should feel like they are talking to a strong lead developer who
can coordinate a full AI team behind the scenes.

The host owns:

- the chat UI
- the session history
- the model picker
- tool execution
- MCP integrations
- approvals and diffs

`ergon` owns:

- orchestration
- delegation
- review
- iteration
- judgment
- final synthesis

The product is not a chatbot wrapper and not a workflow runner. It is a lead
developer with a company behind them.

## Core Mental Model

The user is the product manager.

The orchestrator is the lead developer.

Other agents are employees the lead can bring in, direct, challenge, replace,
or ignore depending on the task.

That metaphor is not decorative. It is the operating model.

If the PM asks for a typo fix, the lead may do it personally.

If the PM wants a feature, the lead may:

- talk through the idea
- sharpen the plan
- ask an architect for structure
- spin up several coders with different angles
- ask a reviewer to compare results
- hand the winner to another coder for polishing
- ask a tester for evidence
- deliver when satisfied

The orchestrator is the person the PM trusts to get the work done. Everything
else is in service of that.

## The Real Product Promise

Weak or middling local models often fail because they do one shallow pass.

`ergon` improves outcomes by adding the behavior a good lead developer adds:

- break the problem down
- choose the right people
- create multiple shots when needed
- inspect results critically
- iterate on weak spots
- decide when the work is good enough to ship

The value is not "multi-agent" by itself.

The value is disciplined, adaptive collaboration.

## The Orchestrator

The orchestrator is the center of the system.

### As the PM's counterpart

Most of the time the orchestrator is simply talking to the PM:

- clarifying goals
- brainstorming
- challenging weak assumptions
- refining acceptance criteria
- deciding whether the request is ready for execution

This should feel like talking to a strong lead developer, not to a router.

### As the staffing authority

When execution is needed, the orchestrator decides:

- whether to work directly
- whether to ask one specialist for help
- whether to involve multiple specialists
- whether to run one attempt or several
- whether to compare, debate, repair, polish, or verify

The orchestrator can effectively "hire" and "fire" on demand for the current
problem. In practice that means instantiating role templates as needed.

The same role may appear multiple times in one effort:

- three coders for best-of-N
- two critics for stress-testing a plan
- one architect early and another later to revisit structure

### As the final judge

The orchestrator reads every meaningful result and decides what happens next.

There is no blind forward-only pipeline.

After each meaningful step, the orchestrator can:

- continue
- ask follow-up questions
- request clarification
- send work back for revision
- change staffing
- abandon the current approach
- deliver

This judgment loop is the product.

## Agents and Roles

Agents are not fixed people. They are reusable role templates.

The shipped defaults should represent a small but capable software company:

- `orchestrator`: lead developer and delivery owner
- `architect`: systems and design thinker
- `coder`: implementation specialist
- `reviewer`: quality and risk reviewer
- `tester`: verification and evidence specialist
- `researcher`: context gathering and option analysis
- `critic`: adversarial challenger for assumptions and weak plans

Users should be able to define their own roles freely. A designer, SEO
specialist, security auditor, data engineer, or documentation writer should all
fit the same model.

## Playbooks, Not Pipelines

What the code currently calls "workflows" should be understood as playbooks.

A playbook is a familiar tactic the orchestrator knows how to use. It is not a
rigid script and not the source of truth.

Good default playbooks include:

- direct execution
- staged build
- research and synthesis
- best-of-N
- debate
- repair loop
- polish pass
- handoff chain

For each playbook, the important thing is:

- when it is useful
- what kinds of roles it usually involves
- what shape the collaboration often takes
- what "done" tends to look like

The orchestrator may:

- use the playbook as-is
- skip steps
- add extra rounds
- repeat a step
- combine multiple playbooks
- ignore playbooks entirely and improvise

Playbooks are defaults, not laws.

## The Orchestrator Loop

The real loop is:

1. Understand the PM's intent
2. Decide the best immediate move
3. Run that move
4. Read and evaluate the result
5. Decide the next move
6. Repeat until the goal is materially satisfied

The "next move" might be:

- answer directly
- ask the PM a clarifying question
- ask the architect for a plan
- ask a coder to implement
- spin up several coders
- ask the reviewer to compare options
- ask the tester for evidence
- ask the critic to attack the current plan
- summarize and deliver

The key is that control returns to the orchestrator after each move.

## Information Flow

The orchestrator is responsible for context shaping.

Agents should receive the context that is relevant to their assignment, not a
raw dump of every message and tool result.

Examples:

- a coder should receive the goal, the current plan, constraints, and relevant
  prior outputs
- a reviewer should receive the goal, the proposed work, and the evidence
- a critic should receive the proposal and be asked to break it

This mirrors a real company: the lead developer synthesizes the situation and
briefs people clearly.

## Iteration and Repair

A review is useless if the system cannot do anything with it.

When a reviewer, tester, or critic finds a problem, the orchestrator should be
able to decide among several responses:

- accept the criticism and send the work back
- ask for clearer evidence
- bring in a different specialist
- narrow the scope
- change the approach entirely
- judge the issue as non-blocking and move forward anyway

That decision belongs to the orchestrator, not to a rigid repair policy.

## What "Done" Means

The orchestrator decides when the work is ready.

That decision should be informed by:

- the PM's goal
- acceptance criteria
- the state of the deliverable
- review feedback
- verification evidence
- whether further iteration is likely to improve outcomes materially

There is no universal hard gate. This should feel like a good lead deciding when
the work is ready to hand back to the PM.

## Host Interaction

The PM should mostly experience one person: the orchestrator.

Internal collaboration should remain visible as a worklog or reasoning stream,
but the host-facing conversation is still primarily PM <-> lead developer.

The host's tools remain the execution layer. `ergon` should not re-own:

- sessions
- tool UX
- approvals
- diffs
- file browsers

The proxy exists to improve judgment and collaboration, not to replace the host.

## What We Must Avoid

The product becomes weaker if we drift into any of these shapes:

- a classifier that picks one fixed lane and disappears
- a rigid workflow runner that treats playbooks as scripts
- a system where reviewers and testers are decorative instead of influential
- a model that always delegates, even when direct action is better
- a model that always pipelines work forward instead of reconsidering after each step

Those behaviors may be easier to implement, but they are not the product.

## Product Direction

The architecture should move toward:

- an orchestrator-first control loop
- roles as staffing templates
- playbooks as optional tactics
- explicit iteration, comparison, and review
- adaptive staffing per task
- a transparent host-facing worklog

The question for every future change should be:

"Does this make the orchestrator behave more like a trusted lead developer who
can coordinate a team and get the PM's vision materialized?"

If the answer is no, it is probably drift.

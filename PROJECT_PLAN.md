# The Orchestrator Project Plan

## 1. Product Definition

The Orchestrator is a local-first AI coding environment built as a Python CLI/TUI on top of Microsoft Agent Framework.

It is not meant to feel like a generic assistant. It should feel like working with a senior engineer who understands the product, manages a team of specialist AI agents, and uses those agents only when doing so improves the result.

The user acts like a product lead or engineering manager. The orchestrator acts like the lead engineer. Specialist agents act like members of the same software firm.

The product goal is to produce higher-quality coding outcomes from local or OpenAI-compatible models by using process, review, orchestration, and task decomposition instead of relying on a single model pass.

Project identity rule:
Each tracked project should contain a committed file at `<project-path>/.ergon.studio/project.json` with only:
- `project_uuid`

Core principles:
- local-first
- colleague UX, not assistant UX
- adaptive orchestration
- transparent internal work
- quality through review and iteration
- project-aware and user-aware behavior
- OpenAI-compatible provider support
- full codebase agency
- minimal heuristic behavior
- no keyword-triggered actions
- test-driven development
- extensive automated testing

## 2. Core Technology Decisions

Locked decisions:
- language: Python
- primary UI: TUI
- TUI library: Textual
- framework/runtime: Microsoft Agent Framework
- model access: OpenAI-compatible endpoints plus supported native providers
- no custom provider abstraction at this stage
- product name: ergon.studio
- Python package name: ergon_studio

Primary stack:
- Python
- Textual
- Microsoft Agent Framework
- SQLite for structured state
- filesystem storage for large artifacts, logs, and checkpoints

Storage roots:
- global config and editable definitions at `~/.ergon.studio/`
- per-project data at `~/.ergon.studio/<project-uuid>/`
- project identity file at `<project-path>/.ergon.studio/project.json`

Target backends:
- llama-server
- Ollama
- vLLM
- LM Studio
- OpenRouter
- Together
- Groq
- OpenAI
- Anthropic where supported by the framework

## 3. Product Mental Model

The system behaves like a software firm.

The user works mainly with the orchestrator.

The orchestrator:
- understands goals
- remembers product and project context
- plans work
- decides whether to work directly or delegate
- hires specialist agents when useful
- collects results
- runs review and repair loops
- judges whether the work fits the goals, principles, and style

Specialist agents are team members, not theatrical personas. They exist to improve outcomes.

Small tasks should stay simple.
Large or ambiguous tasks should trigger orchestration.

## 4. High-Level Architecture

Top-level components:
- TUI app
- session/project state
- agent definitions
- workflow definitions
- tool layer
- provider/model configuration
- memory and knowledge store
- task ledger and transparency views
- approvals and execution safety
- artifact manager

Architecture shape:
- TUI and CLI shell
- product runtime built on Microsoft Agent Framework
- agent/workflow execution through Agent Framework primitives
- local tools and MCP integrations
- persistent storage for sessions, tasks, memory, and artifacts

Important rule:
Microsoft Agent Framework runs the team.
Our code defines how the team behaves.

Framework-first rule:
Before building any subsystem, check whether Microsoft Agent Framework already provides the needed primitive.
We should compose framework capabilities before inventing parallel infrastructure.

Heuristics rule:
We should keep heuristic checks and behavior to a minimum.
We should not trigger actions based on keywords or substring matching under any circumstances.
Routing, delegation, approvals, and workflow selection should be driven by structured state, explicit user intent, typed model outputs, and orchestrator decisions rather than text-pattern rules.

Testing rule:
We follow TDD.
Core functionality should be implemented test-first whenever practical.
The project should maintain extensive automated coverage across unit, integration, and end-to-end layers.

## 5. Conversation and Thread Model

The product should use threaded transparency instead of a single shared public room.

### 5.1 Main Thread

Participants:
- user
- orchestrator

Purpose:
- discuss goals
- clarify requirements
- report progress
- present plans
- ask for approvals
- summarize internal work
- present outcomes
- react to changes in direction

This is the primary relationship surface.

Agent Framework fit:
- use framework conversations and sessions as the base communication primitive
- use group chat and workflow primitives for internal collaboration
- implement the product-specific thread model on top

### 5.2 Agent Direct Threads

Participants:
- orchestrator
- one specialist agent

Purpose:
- delegation
- implementation
- research
- debugging
- refinement
- specialist consultation

### 5.3 Group Workrooms

Participants:
- orchestrator
- multiple agents

Purpose:
- brainstorms
- architecture debates
- best-of-N comparisons
- collaborative review
- conflict resolution

### 5.4 Review Threads

Participants:
- orchestrator
- reviewer
- optionally implementer
- optionally fixer

Purpose:
- focused review
- findings
- repair loop
- recommendation to approve or reject

### 5.5 Approval Threads / Approval Events

Participants:
- orchestrator
- user

Purpose:
- risky commands
- destructive edits
- network access
- dependency installation
- strategic replans

### 5.6 System Threads

Not normal chat.

Purpose:
- tool logs
- command output
- workflow bookkeeping
- trace/debug events

### 5.7 Thread Rules

- main thread stays clean
- all side threads remain inspectable
- threads belong to tasks
- orchestrator creates and closes threads
- summaries flow upward into the main thread

Implementation rule:
The thread model is product-specific.
We should not replace it with a raw framework conversation model, but we should build it on top of framework conversations, sessions, and workflow state.

## 6. Agent Roster

### 6.1 Orchestrator

Role:
- senior engineer
- planner
- final approver
- user-facing colleague
- workflow controller

Responsibilities:
- understand user goals
- maintain project context
- create and update plans
- choose workflow type
- choose staffing
- delegate work
- collect results
- trigger review and repair loops
- judge alignment with goals and style
- communicate with the user

Tools:
- all tools
- full memory access
- thread and task management

### 6.2 Architect

Role:
- system design
- decomposition
- interface design
- tradeoff analysis

Use for:
- new features
- refactors
- unclear structure
- technical decisions

### 6.3 Coder

Role:
- implementation

Use for:
- writing code
- modifying code
- integration work

### 6.4 Reviewer

Role:
- critical code review

Use for:
- any non-trivial implementation
- risky changes
- architecture-sensitive work

Behavior:
- skeptical
- precise
- quality-focused

### 6.5 Fixer

Role:
- resolve review findings and test failures

Use for:
- corrective patches
- polishing
- regression repair

### 6.6 Researcher

Role:
- gather external or internal reference material

Use for:
- framework lookup
- package evaluation
- standards and spec lookup
- API behavior

### 6.7 Tester

Role:
- verify behavior

Use for:
- feature validation
- bug reproduction
- regression checks

### 6.8 Documenter

Role:
- docs and help text

Use for:
- docs updates
- usage guides
- design notes

### 6.9 Brainstormer

Role:
- divergent thinking

Use for:
- ideation
- alternative architectures
- escaping dead ends

High temperature by design.

### 6.10 Designer

Role:
- UX and visual design

Use for:
- TUI design
- layout decisions
- interaction design

### 6.11 Agent Rules

- orchestrator decides who is needed
- agents do not exist for show
- agents may be assigned per task
- role capabilities follow permissions and tool access
- model choice may differ by role

Agent Framework fit:
- each role should be implemented as a framework agent
- instructions, tools, model client, context providers, and middleware should come from framework-native patterns
- workflows may be exposed as agents where useful

## 7. Workflow Catalog

Workflows are selected by the orchestrator based on scope, ambiguity, risk, impact, confidence, and model capability.

### 7.1 Direct Response

Use for:
- questions
- tiny changes
- simple explanations
- low-risk commands

### 7.2 Single-Agent Execution

Use for:
- straightforward isolated work

Flow:
- orchestrator creates task
- delegates to coder
- checks result
- optionally requests review

### 7.3 Architecture First

Use for:
- new features
- significant refactors
- unclear design

Flow:
- architect produces architecture artifact
- orchestrator approves direction
- implementation follows

### 7.4 Research Then Decide

Use for:
- unknown frameworks
- dependency decisions
- standards questions

### 7.5 Standard Build Workflow

Flow:
- orchestrator plans
- architect if needed
- coder implements
- reviewer reviews
- fixer repairs if needed
- reviewer re-checks
- orchestrator decides

### 7.6 Best-of-N Generation

Use for:
- ambiguous implementation paths
- quality-sensitive work
- local model weakness mitigation

Flow:
- N coders or N coding runs work in parallel
- reviewer selects or synthesizes best result
- fixer may polish
- orchestrator decides

### 7.7 Debate Workflow

Use for:
- competing approaches
- high-stakes tradeoffs
- agent disagreement

### 7.8 Review-Driven Repair Loop

Use for:
- non-trivial implementations with findings

### 7.9 Test-Driven Repair

Use for:
- failing tests
- runtime bugs
- regressions

### 7.10 Approval-Gated Workflow

Use for:
- risky commands
- broad refactors
- dependency installs
- network actions

### 7.11 Documentation Workflow

Use for:
- changes that require documentation updates

### 7.12 Replanning Workflow

Use for:
- user changes direction
- agents discover a better path
- blockers invalidate the current plan

### 7.13 Task States

- created
- planned
- assigned
- in_progress
- blocked
- in_review
- needs_fix
- awaiting_approval
- approved
- rejected
- completed
- canceled

### 7.14 Workflow States

- draft
- running
- waiting
- failed
- completed
- aborted

Agent Framework fit:
- prefer built-in orchestrations such as Sequential, Concurrent, Group Chat, Magentic, and Handoff
- custom workflows should be compositions of framework orchestrations, not separate workflow engines
- workflow state, checkpoints, and events should rely on framework mechanisms where possible

Current workflow-to-framework mapping:
- Direct Response -> plain orchestrator agent run
- Single-Agent Execution -> orchestrator plus one worker agent
- Architecture First -> Sequential
- Research Then Decide -> Sequential
- Standard Build Workflow -> Sequential
- Best-of-N Generation -> Concurrent, then review
- Debate Workflow -> Group Chat
- Review-Driven Repair Loop -> Sequential with controlled re-entry
- Test-Driven Repair -> Sequential with controlled re-entry
- Approval-Gated Workflow -> pause/resume with human approval
- Replanning Workflow -> orchestrator-driven cancellation or supersession
- dynamic open-ended delegation -> Magentic, used sparingly

Workflow policy rule:
- do not select workflows with keyword heuristics
- use structured planner outputs, typed state, and explicit orchestrator decisions

## 8. Memory and Context Model

Memory should be built around Microsoft Agent Framework primitives instead of a custom parallel memory engine.

Relevant framework features to use:
- AgentSession
- conversation history providers
- AIContextProvider
- workflow checkpointing and persistence
- agent memory / retrieval patterns
- storage and reducers/compaction where available

### 8.1 Memory Layers

#### History Memory

Use framework session and history support for:
- recent messages
- thread-local continuity
- short-term context

Scope:
- per thread

#### Whiteboard Context Provider

Structured active context for the current task:
- goal
- constraints
- plan
- decisions
- open questions
- acceptance criteria

Scope:
- per task

#### Project Memory Provider

Durable facts about the project:
- architecture decisions
- conventions
- preferences
- recurring patterns
- product vision
- user preferences

Scope:
- per project
- across sessions

#### Retrieval Provider

Semantic lookup over:
- docs
- decision notes
- important artifacts
- indexed knowledge

Scope:
- per project

#### Agent Profile Provider

Inject role-specific guidance:
- role instructions
- tool access
- speaking rules
- review standards
- model profile

Scope:
- per agent type

### 8.2 Memory Policy

Promote to durable memory when:
- a decision is made
- a stable preference is established
- an architecture fact is declared
- a convention becomes binding
- a requirement becomes durable

Do not promote:
- ordinary noise
- full raw transcripts
- temporary failed ideas unless important

### 8.3 Retrieval Policy

Before agent execution:
- retrieve thread-local history
- retrieve task whiteboard
- retrieve relevant project memory
- retrieve role profile
- retrieve relevant artifacts if needed

The orchestrator gets broader context than worker agents.

Agent Framework fit:
- use framework sessions, history, context providers, storage, and RAG primitives directly
- only add product-specific providers for whiteboard memory, project memory policy, and persistent project facts

## 9. Tool System

The tool system should use Agent Framework’s built-in tool patterns.

Relevant framework features to use:
- Python function tools
- MCP tools
- agent-as-tool patterns
- middleware/hooks around actions
- approval support for MCP tools

### 9.1 Native Function Tools

Core local tools:
- read file
- write file
- patch file
- list files
- search files
- run command
- inspect diff
- read logs
- manage tasks
- manage threads
- manage artifacts

These should be first-class and local.

### 9.2 MCP Tools

Use MCP for:
- docs lookup
- browser/search
- GitHub
- issue trackers
- package registries
- future integrations

### 9.3 Agent-as-Tool

Use where useful for specialist invocation inside workflows.

Examples:
- reviewer as callable specialist
- architect as callable specialist
- researcher as callable specialist

### 9.4 Tool Risk Classes

safe:
- read file
- search
- list
- inspect diff
- read docs

moderate:
- write file
- patch file
- create artifact
- run non-destructive commands

high-risk:
- delete files
- move files
- install dependencies
- network-changing actions
- git-changing actions
- destructive shell commands

### 9.5 Tool Access by Agent

- orchestrator: all tools
- architect: read/search/docs and limited write
- coder: read/write/patch/commands
- reviewer: read/search/diff/verification
- fixer: coder tools
- researcher: web/docs/MCP lookup
- tester: tests, commands, logs
- documenter: docs read/write
- brainstormer: mostly read/context tools

Agent Framework fit:
- implement tools as framework function tools or MCP tools
- implement logging, policy, and interception through framework middleware where possible
- use agent-as-tool patterns for specialist invocation instead of hand-rolled delegation wrappers

## 10. Approval Model

Approvals should be first-class.

### 10.1 Approval Targets

- file writes
- destructive file operations
- shell commands
- dependency installs
- git actions
- network actions
- strategic replans

### 10.2 Approval Levels

- auto
- notify
- ask
- block

### 10.3 Recommended Default Policy

- reads/search: auto
- file writes/patches: notify or ask
- normal commands: ask
- destructive commands: block unless approved
- network/dependency install: ask
- major replans: ask

### 10.4 Approval Record Fields

- id
- task id
- thread id
- requesting agent
- action
- risk class
- reason
- affected resources
- timestamp
- user decision

Agent Framework fit:
- use framework approval and human-in-the-loop capabilities for tool and workflow gating
- build the TUI approval experience and persistence layer around those primitives

## 11. Provider and Model Strategy

We should use Agent Framework provider support directly instead of building extra abstraction now.

Framework-aligned approach:
- use OpenAI-style agents for broad compatibility
- use supported native integrations where useful
- configure base URLs and models directly
- allow per-role model settings

### 11.1 Supported Backends

- llama-server
- Ollama
- vLLM
- LM Studio
- OpenRouter
- Together
- Groq
- OpenAI
- Anthropic where supported

### 11.2 Configurable Model Settings

Per provider and per role:
- base_url
- api_key
- model_id
- temperature
- max_tokens
- timeout
- structured output support
- tool calling support

### 11.3 Per-Role Routing

Example:
- orchestrator: best reasoning model
- architect: strong reasoning model
- coder: best coding model
- reviewer: strict low-temperature reasoning model
- fixer: fast lower-cost iterative model
- brainstormer: high-temperature creative model
- tester: deterministic precise model

### 11.4 Capability Registry

Track actual model capabilities:
- tool calling
- structured output
- context length
- streaming
- parallel suitability
- reasoning strength

The orchestrator should use this when choosing workflows.

Agent Framework fit:
- use framework-native model clients directly
- avoid introducing a custom provider abstraction unless a real need appears later

Initial provider/client decision:
- start with OpenAI-compatible chat clients
- target `llama-server` first
- add `Ollama` next
- keep Responses-style support optional for later

## 12. TUI Design

The TUI is the office workspace.

It is not just chat.

Primary panels:
- Main Chat
- Task Tree
- Threads
- Active Thread View
- Artifacts / Diffs
- Commands / Output
- Approvals
- Memory / Context
- Activity / Events
- Settings

### 12.1 View Modes

- Conversation
- Workflows
- Artifacts
- Memory
- Settings

### 12.2 Interaction Rules

- main chat remains central
- task selection filters related data
- thread selection opens conversation
- artifact selection opens diff/content
- approvals are actionable inline
- side panels are collapsible
- keyboard-first navigation is mandatory

### 12.3 UX Goal

The interface should feel like a live software firm control room:
- clear
- inspectable
- transparent
- low-noise
- serious and work-focused

### 12.4 Settings Requirements

The TUI should include a settings area for editing core system configuration without leaving the app.

Settings should include:
- team composition
- agent definitions
- system prompts per agent
- model assignments per role
- workflow definitions
- approval defaults
- provider settings

Edits made in settings should update the on-disk files under `~/.ergon.studio/`.

Reload behavior:
- settings edits write to disk immediately
- definitions are validated on save
- registries reload after save
- active workflow runs keep their current loaded definitions
- only future runs use the updated definitions

Agent Framework fit:
- the TUI is product-specific
- settings should hydrate framework agents, workflows, tools, and model clients from editable on-disk definitions

## 13. Storage Design

Use:
- SQLite for structured metadata and state
- filesystem storage for large artifacts, logs, and checkpoints

Suggested storage layout:

```text
~/.ergon.studio/
  config.json
  agents/
    orchestrator.md
    architect.md
    coder.md
    reviewer.md
    fixer.md
    researcher.md
    tester.md
    documenter.md
    brainstormer.md
    designer.md
  workflows/
    direct-response.md
    single-agent-execution.md
    architecture-first.md
    standard-build.md
    best-of-n.md
    review-repair-loop.md
    test-driven-repair.md
    approval-gated.md
    replanning.md
  <project-path>/.ergon.studio/
    project.json
  <project-uuid>/
    state.db
    sessions/
    threads/
    tasks/
    memory/
    artifacts/
    checkpoints/
    indexes/
    logs/
    diffs/
    exports/
```

Storage rules:
- SQLite stores metadata only
- message bodies should live as markdown files on disk
- SQLite records reference on-disk files
- all datetimes must be stored as Unix time `INT`

### 13.1 Core Records

Session:
- id
- project id
- created_at
- active thread ids
- active task ids
- provider config
- current model routing

Thread:
- id
- type
- participants
- parent_task_id
- parent_thread_id
- summary
- status

Message:
- id
- thread id
- sender
- type
- body_path
- linked artifacts
- linked tool calls
- timestamp

Task:
- id
- title
- state
- workflow type
- owner agent
- child task ids
- acceptance criteria
- artifact ids

Memory Fact:
- id
- scope
- kind
- content
- source
- confidence
- tags
- created_at
- last_used_at

Artifact:
- id
- type
- task id
- thread id
- path or blob ref
- summary

### 13.2 Suggested SQLite Tables

- sessions
- threads
- messages
- tasks
- task_edges
- artifacts
- approvals
- tool_calls
- memory_facts
- workflow_runs
- events
- provider_profiles
- agent_profiles

First-pass schema should remain metadata-focused.
Do not store large message bodies or large artifacts in SQLite.

### 13.3 Filesystem Buckets

- artifacts/
- diffs/
- command_logs/
- checkpoints/
- exports/

Agent Framework fit:
- use framework checkpoint and storage mechanisms where they map cleanly
- keep custom storage focused on product entities not covered by the framework, such as task tree metadata, thread registry, approval records, and artifact indexing

## 14. Config Design

We should keep configuration and editable definitions centralized under the user’s home directory.

Global config:

```text
~/.ergon.studio/config.json
```

This should hold:
- provider definitions
- model assignments per role
- default approval policies
- global UX preferences
- global runtime defaults

Editable markdown definitions should live under:

```text
~/.ergon.studio/agents/
~/.ergon.studio/workflows/
```

These files should define:
- agent identity and role
- system prompts
- tool permissions
- speaking behavior
- workflow behavior
- workflow trigger guidance

Markdown format rule:
- use YAML frontmatter for structured configuration
- use markdown body for instructions and policy text

Agent definition section recommendation:
- `## Identity`
- `## Responsibilities`
- `## Rules`
- `## Tool Usage`
- `## Collaboration`
- `## Output Style`

Workflow definition section recommendation:
- `## Purpose`
- `## When To Use`
- `## Flow`
- `## Decision Rules`
- `## Exit Conditions`

Project-specific state should live under:

```text
~/.ergon.studio/<project-uuid>/
```

Config layers:
- built-in defaults
- global user config
- project state/config where needed
- runtime overrides

Agent Framework fit:
- config should compile into framework-native agent, workflow, tool, and client definitions

## 15. Package Structure

Suggested Python package layout:

```text
theorchestrator/
  __init__.py
  main.py
  app/
  tui/
  agents/
  workflows/
  tools/
  memory/
  storage/
  config/
  providers/
  approvals/
  artifacts/
  tasks/
  threads/
  events/
  evals/
  tests/
```

Implementation rule:
Package structure should reflect product concerns, but core agent and workflow execution should remain framework-native inside those modules.

## 16. Observability and Transparency

We need strong observability because orchestration quality depends on inspectability.

Track:
- workflow state changes
- task lifecycle changes
- agent assignments
- tool calls
- command results
- approvals
- memory extraction events
- retrieval events
- review findings
- final acceptance decisions

The user should be able to inspect:
- why an agent was used
- what it produced
- what was accepted or rejected
- why the orchestrator made a decision

Agent Framework fit:
- use framework event streams, middleware, and workflow events as the primary observability source
- store and present those events in the product transparency layer rather than building an unrelated event system

## 17. Testing and Evaluation

We need both software tests and orchestration evaluations.

### 17.1 Software Tests

- unit tests
- storage tests
- config tests
- tool tests
- workflow tests
- TUI component tests
- end-to-end tests

### 17.2 Agent and Workflow Evals

Measure:
- task success
- review quality
- fix-loop success
- best-of-N improvement
- approval correctness
- memory retrieval quality
- replanning quality

This product cannot rely only on normal software tests because a core part of the product is orchestration behavior.

Testing rule:
- test workflow decisions and transitions through structured state and typed outputs
- do not encode behavior tests around keyword-triggering assumptions
- write tests before or alongside implementation as the default development mode
- maintain strong coverage for both deterministic code and orchestrated flows
- prefer deterministic fixtures and fake providers in unit and integration tests

## 18. Packaging and Distribution

Target form:
- installable Python package
- terminal entry command
- local app data directory
- project-local config support

Possible distribution modes:
- pip
- pipx
- packaged standalone build later if useful

## 19. Build Order

This is the construction order, not a scope reduction.

1. Create project skeleton
2. Set up packaging and Python environment
3. Implement config loading and validation
4. Implement storage layer
5. Implement thread, message, and task models
6. Implement provider setup and model configuration
7. Implement core function tools
8. Implement orchestrator agent
9. Implement specialist agents
10. Implement workflow catalog
11. Implement memory/context providers
12. Implement approval system
13. Implement TUI shell
14. Implement thread/task/artifact panels
15. Implement command/output and tool activity views
16. Implement eval harness
17. Polish packaging, docs, and defaults

## 20. Final Product Statement

The Orchestrator is a local-first AI software firm in a terminal workspace.

The user works with an orchestrator that behaves like a senior engineer.
The orchestrator holds context, manages specialist agents, delegates work when useful, reviews outcomes, and keeps the user in control.

Microsoft Agent Framework provides the core agent, workflow, memory, tool, and runtime foundation.

Our job is to build the product layer on top:
- the colleague UX
- the thread/task model
- the orchestration policy
- the TUI workspace
- the approval system
- the persistent project memory
- the transparency and quality controls

## 21. Agent Framework Mapping Summary

Use Microsoft Agent Framework directly for:
- agents
- workflow orchestration
- group collaboration patterns
- sessions and conversation history
- context providers
- memory and RAG primitives
- tool calling
- MCP integration
- approvals and human-in-the-loop flows
- middleware
- checkpoints
- workflow and runtime events
- model client integrations

Build custom product code only for:
- TUI and terminal UX
- main-thread and side-thread office model
- task tree and staffing model
- markdown-defined team and workflow files
- project storage layout under `~/.ergon.studio/`
- artifact management and inspection UX
- product-specific memory extraction policy
- transparency ledger and views
- project-level configuration editing

## 22. Anti-Heuristic Rule

The system should avoid brittle heuristic behavior.

Hard rules:
- no keyword-triggered actions
- no substring-based workflow selection
- no routing based on raw text pattern checks
- no agent selection based on matching words in user messages
- no approval behavior driven by keyword spotting

Preferred drivers of behavior:
- explicit user requests
- structured planner outputs
- typed state
- workflow state transitions
- tool metadata
- approval metadata
- orchestrator decisions
- model outputs constrained to structured schemas

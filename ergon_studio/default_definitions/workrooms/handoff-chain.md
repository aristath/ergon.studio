---
id: handoff-chain
name: Handoff Chain
orchestration: handoff
steps:
  - researcher
  - architect
  - coder
  - tester
  - critic
  - reviewer
start_agent: researcher
finalizers:
  - reviewer
handoffs:
  researcher:
    - architect
    - coder
  architect:
    - coder
    - critic
  coder:
    - tester
    - reviewer
  tester:
    - coder
    - reviewer
  critic:
    - architect
    - coder
max_rounds: 6
selection_hints:
  - handoff
  - relay
  - staged_handoff
  - long_chain
delivery_candidate: true
acceptance_mode: delivery
---

## Purpose
Move the work through a deliberate chain of specialists where each handoff is an
explicit judgment about who should take over next.

## Use When
- ownership needs to pass from research to design to implementation to
  verification
- the lead developer wants each specialist to decide who should receive the work
  next
- the task benefits from progressive refinement across several roles

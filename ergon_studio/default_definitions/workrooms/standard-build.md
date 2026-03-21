---
id: standard-build
name: Staged Build
shape: sequential
steps:
  - architect
  - coder
  - tester
  - reviewer
selection_hints:
  - build
  - implementation
  - feature
  - staged_delivery
  - ship_it
---

## Purpose
Common delivery playbook for new features or meaningful code changes.

## Use When
- the goal is concrete enough to build
- the lead developer wants a measured pass through planning, implementation,
  verification, and review
- the work is non-trivial but not chaotic enough to require open-ended staffing

## Notes
This is a playbook, not a law. The lead developer may skip steps, revisit them,
or use a different tactic entirely.

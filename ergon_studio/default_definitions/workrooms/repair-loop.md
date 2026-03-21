---
id: repair-loop
name: Repair Loop
shape: sequential
steps:
  - reviewer
  - coder
  - tester
  - reviewer
selection_hints:
  - repair
  - revise
  - fix_review_findings
  - stabilize
---

## Purpose
Take concrete findings, revise the work, verify the fix, and review again.

## Use When
- there is already a candidate result
- quality concerns are specific enough to address
- the lead developer wants focused iteration instead of a full restart

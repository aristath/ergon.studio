---
id: polish-pass
name: Polish Pass
shape: sequential
steps:
  - reviewer
  - coder
  - reviewer
selection_hints:
  - polish
  - refine
  - final_pass
  - cleanup
delivery_candidate: true
acceptance_mode: delivery
---

## Purpose
Tighten an already-good result before delivery.

## Use When
- the core work is done
- the remaining issues are about clarity, rough edges, or final quality
- a focused cleanup pass is cheaper than a bigger repair cycle

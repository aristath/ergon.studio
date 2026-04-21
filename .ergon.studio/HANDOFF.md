# Handoff

## Completed this session

Built a complete quality loop system to ensure code quality before marking tasks complete:

### New Files Created

1. **`.ergon.studio/COMPLETION.md`** — Defines what "done" means with explicit checklist:
   - Code quality (no bugs, no TODOs, edge cases)
   - Testing (tests written, passing, meaningful)
   - Documentation (inline docs, README, config)
   - Integration (no breaking changes)

2. **`agents/design_reviewer.md`** — New subagent for optimality review:
   - Reviews code architecture, structure, performance, trade-offs
   - Returns "APPROVED" or "Needs Improvement" with specific suggestions
   - Separate from reviewer (which checks bugs) to focus on design quality

3. **`agents/quality_controller.md`** — New subagent that runs the full quality loop:
   - Phase 1: Invokes reviewer (bug check)
   - Phase 2: Invokes design_reviewer (optimality check)
   - Phase 3: Verifies COMPLETION.md checklist
   - Returns "APPROVED" only if ALL phases pass
   - Tracks iterations, warns after 3 failures

### Files Modified

4. **`agents/orchestrator.md`** — Added mandatory quality gates section:
   - Must invoke quality_controller after ANY code task
   - Cannot declare completion without "APPROVED"
   - Must fix issues and re-invoke if "REJECTED"
   - After 3 iterations, must ask user for direction

5. **`prompts/steward.md`** — Added judge examples for quality preferences:
   - "Always run quality controller before marking complete"
   - "Write tests for all new functionality"
   - "Include design reviewer in quality loop"
   - "Update README for user-facing changes"
   - "No TODOs in completed code"

6. **`opencode.json`** — Added permissions for new agents:
   - `design_reviewer`: edit deny, bash deny
   - `quality_controller`: edit deny, bash deny

## In progress

Nothing. The quality loop system is complete and ready to test.

## Decisions pending

None.

## Start here next session

Test the quality loop with a real code task:

1. Switch to orchestrator
2. Give it a small but non-trivial task (e.g., "add a function to parse X")
3. Watch whether it:
   - Invokes quality_controller after coding
   - quality_controller properly chains reviewer → design_reviewer → checklist
   - The loop terminates correctly when all phases pass
   - The orchestrator fixes issues and re-invokes when rejected

## Watch out for

1. **Orchestrator might skip quality_controller** — Local models ignore instructions. If it declares completion without invoking quality_controller, strengthen the prompt or add explicit "before you reply to the user" language.

2. **Quality controller might be too lenient** — Watch its first few approvals. If it approves code with obvious issues, tighten its criteria in the prompt.

3. **Infinite loops** — If the code can't pass (fundamental design flaw), the 3-iteration limit should kick in. Verify this works.

4. **Steward memory** — The new judge examples need to be exercised. After a few sessions where you correct the orchestrator about quality, check if the steward is saving those preferences.

5. **COMPLETION.md is static** — The quality_controller reads it, but it's not injected into context like scratchpad. If the controller "forgets" to check it, consider plugin injection.

## Architecture Notes

The quality loop is:
```
orchestrator codes → quality_controller → [reviewer → design_reviewer → checklist] → APPROVED/REJECTED
```

If REJECTED:
```
orchestrator fixes → quality_controller → [repeat] → APPROVED
```

After 3 REJECTED cycles, orchestrator must ask user for direction.

This is enforced by prompts, not code. Local models are unreliable at following multi-step processes, so watch for shortcuts.

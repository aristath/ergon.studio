# Task Completion Checklist

A task is NOT complete until EVERY item below passes. The quality controller enforces this.

## Code Quality
- [ ] Reviewer verdict: Accept (no bugs, no logic errors)
- [ ] Design reviewer verdict: Optimal (no improvements suggested)
- [ ] No TODOs, FIXMEs, or placeholders in the code
- [ ] Edge cases handled (empty input, errors, boundaries, type safety)

## Testing
- [ ] Tests written for all new functionality
- [ ] All tests passing
- [ ] Edge cases covered in tests
- [ ] Tests are meaningful (not just "it runs")

## Documentation
- [ ] Inline docs for all public functions, classes, methods, APIs
- [ ] README updated (if user-facing changes)
- [ ] New config/options documented
- [ ] Breaking changes documented with migration path

## Integration
- [ ] No breaking changes to existing functionality (or migration documented)
- [ ] Existing tests still pass
- [ ] Code compiles/builds successfully

## Final Verification
- [ ] Quality controller returned "APPROVED"
- [ ] All items above are checked

If ANY item is unchecked, the task is NOT complete. The quality controller will reject it.

---

## What "Complete" Means

Complete means:
- The code works correctly (no bugs)
- The design is optimal (not just "it works")
- Tests prove it works (not just "I tested it manually")
- Docs explain how to use it (not just "it's obvious")
- Nothing breaks existing functionality (or migration is provided)

Complete does NOT mean:
- "I wrote code that works"
- "I think it's good enough"
- "Tests can be added later"
- "Docs are someone else's job"

---

## Quality Loop

The quality controller runs this loop:

1. **Reviewer Pass** → Check for bugs and logic errors
   - If issues found: return them for fixing
   - If clean: proceed to step 2

2. **Design Reviewer Pass** → Check for optimality
   - If improvements suggested: implement them, return to step 1
   - If optimal: proceed to step 3

3. **Checklist Verification** → Verify all items above
   - If anything unchecked: complete the missing work, return to step 1
   - If all checked: return "APPROVED"

This loop continues until ALL phases pass. Maximum 3 iterations; after that, ask the user for direction.

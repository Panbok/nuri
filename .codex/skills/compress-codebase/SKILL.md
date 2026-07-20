---
name: compress-codebase
description: Maps every in-scope source file and executes evidence-driven semantic compression passes that remove redundant checks, branches, comments, wrappers, representations, and shallow abstractions while preserving behavior, ownership, and measured performance. Use when auditing a codebase for major LOC reduction, creating a file-by-file compression plan, or implementing staged code-compression and consolidation passes.
---

# Compress Codebase

## Contract

Compress relationships and states, not formatting. Preserve observable behavior,
public contracts, ownership, concurrency, lifetime, and measured hot-path
performance. A smaller file is not success when total maintained code or concept
count grows elsewhere.

Map every file in scope before broad edits. Coverage means every original file has
an explicit disposition, including files deliberately kept unchanged.

## Select the mode

- **Map:** Produce an exhaustive implementation handoff without editing source.
- **Execute:** Build or refresh the map, then implement it as verified vertical
  slices. Never start a broad compression pass from aggregate LOC alone.
- **Resume:** Reconcile the existing ledger against the current tree before more
  edits; do not silently skip or reinterpret mapped rows.

## Establish evidence

1. Read repository instructions, domain context, accepted decisions, build/test
   workflow, and relevant dirty-worktree state.
2. Define scope and exclusions literally. Record commit, file count, physical and
   nonblank LOC, generated code, comments, and branch/check proxies.
3. Define behavioral, interface, ownership, concurrency, lifetime, and performance
   invariants before proposing deletion.
4. For implementation, capture the smallest trustworthy baseline owned by each
   affected invariant. Never approve or update baselines implicitly.

## Map every file

Read [references/map-schema.md](references/map-schema.md). Run:

```text
python <skill>/scripts/inventory.py <scope...> --exclude <glob> --output <ledger.tsv>
```

Then inspect concrete callers and at least two real examples before claiming
repetition. Complete every ledger row and produce a narrative map that groups
cross-file work into ordered vertical slices. Reconcile the inventory count to
100%; ignored, generated, vendor, and retained files still need an explicit scope
decision.

## Choose compression

Read [references/compression-rules.md](references/compression-rules.md). Prefer, in
order: delete dead paths; merge duplicate facts into one owner; replace parallel
fields/switch clusters with typed tables; replace broad state scans with
state-specific worklists; collapse forwarding layers into deep concrete modules;
lower canonical data once.

Do not introduce macros, generic frameworks, flag-driven mega-functions,
compatibility aliases, or speculative seams to reduce local LOC. An abstraction
needs two semantically identical concrete uses and must reduce total relationships.

## Execute vertical slices

1. Record the exact files, current LOC, repeated cases, surviving owner, checks
   removed or retained, lifetime implications, expected net deletion, and evidence
   owner.
2. Migrate one source of truth end to end. Delete the superseded representation and
   adapters in the same slice when safe.
3. Keep external/fallible validation at its boundary. Internal code consumes
   references, spans, indices, or state-specific records proven by that boundary.
4. Run the cheapest complete affected gate, recount the whole scope, and record
   both deleted and added lines. Moving code is not compression.
5. Stop and repair any behavior, lifetime, visual, or measured performance
   regression before starting the next slice.

For `lib/nuri` or renderer-facing work, also use `nuri-renderer-design` and the
applicable Nuri autotest, snapshot, and benchmark skills. Preserve
prepare/submit/abandon, graph-payload ownership, and completion-gated GPU identity.

## Finish

Re-inventory the final tree and reconcile every original and new file. Report file
and LOC deltas, removed codepaths/checks, surviving authorities, validation results,
exceptions, and unresolved risks. A map-only result must clearly state that no
implementation or runtime validation was performed.

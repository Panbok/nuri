# Semantic compression rules

## Success measure

Optimize lifetime maintenance cost: concepts, relationships, states, codepaths,
files, branches, checks, comments, allocations, and total maintained LOC. Prefer
readable procedural locality over terse syntax. Count the full scope after every
slice.

## Compression order

1. Delete dead APIs, compatibility includes, translation-unit anchors, generated
   source payloads, unused fields, and one-value modes.
2. Select one authoritative representation for each fact and delete mirrors.
3. Merge repeated handles, pools, binary cursors, cache I/O, descriptor schemas,
   upload rings, or work records only after two concrete policies match.
4. Replace parallel named fields and switch clusters with typed indexed records
   and descriptor tables.
5. Replace “scan all records and ask every state question” with state-specific
   worklists and explicit transitions.
6. Collapse provider/feature/pass, controller/backend, facade/adapter, or PIMPL
   ladders that only forward into a deep concrete owner.
7. Make invalid combinations unrepresentable with tagged records, references,
   spans, fixed ranges, and state-specific operations; lower once.

Keep unique policy local. Do not unify domains merely because their data shapes
look similar.

## Bare-minimum checks

| Seam | Policy |
| --- | --- |
| Untrusted input, config, file, shader, OS, vendor, network | Validate and normalize once; use one checked reader/table rather than repeated local arithmetic. |
| Public handle/API | Perform one canonical lookup or input validation, then pass references/indices internally. |
| Creation/compile boundary | Reject invalid combinations and emit canonical typed records. |
| Internal helper | Accept caller-proven types; remove defensive null/empty/state ladders. |
| Hot loop/lowering | No recoverable validation; at most a debug assertion for a proven invariant. |
| Cancellation/async completion | Keep stale-generation and state checks when reuse and out-of-order completion are real. |
| Ownership/concurrency/submission | Keep checks that are clocks or proofs: capacity, generation, lock ownership, submit/abandon, last use, completion, and retirement. |

Several guards in one internal function usually mean its input type is too broad or
it owns several state-machine phases. Change the representation before deleting
guards blindly.

## Borrowed-view lifetime

Never remove count, reserve, fixed-capacity, arena, or retention machinery merely
because it resembles boilerplate. A published `span`, `string_view`, pointer,
iterator, device address, descriptor index, or borrowed backend object requires
stable backing storage for the complete consumer lifetime.

Before publishing views, use one proven policy: exact pre-reservation before any
view is formed, fixed-capacity storage, an owning arena with a dominating scope,
or owned compiled storage. Reallocation after publication is a use-after-free even
when every handle value and range was valid when created. Add a deterministic
ownership regression at this seam when it can cover the invariant completely.

## Comments

Remove narrative comments after code structure expresses the fact through domain
names, typed states, and explicit data flow. Move lasting rationale and invariants
to context documents, ADRs, schemas, or contract tests. Keep legal attribution,
generated provenance, public API contracts that cannot be typed, and required tool
directives. Bulk comment deletion without structural repair is not compression.

## Architecture tests

- One fact has one owner and one lowering.
- A reusable abstraction has two semantically identical concrete uses.
- A module is deep: small interface, substantial hidden policy, no pass-through
  peer hierarchy.
- Adding N+1 changes one descriptor, table, tagged record, or real adapter without
  rewriting unrelated callers or adding hot-loop indirection.
- High-level operations retain usable lower-level granularity for exceptional
  work.
- New helpers and modules delete more code and relationships than they add.

## Reject

- minification, compressed formatting, chained expressions, or cryptic names;
- macros/templates that hide ordinary control flow only to reduce physical LOC;
- flag-driven mega-functions combining unrelated policy;
- speculative interfaces with one implementation;
- compatibility aliases that preserve the old and new authorities together;
- moving code between files without net scope reduction;
- deleting error handling at fallible boundaries;
- deleting capacity, transaction, synchronization, or retirement proof;
- performance claims without same-configuration measured evidence.

## Renderer overlay

For renderers, preserve visual output, prepare/build/submit/abandon transactions,
graph structure versus frame payload ownership, exact resource use, recording
retention, logical handle invalidation, and completion-gated physical identity.
Keep pipeline creation, allocation, owning-handle churn, strings, locks, and waits
out of per-draw/per-dispatch work. Use focused contract tests for deterministic
ownership, scenarios for multi-frame behavior, snapshots for pixels, and isolated
profiling-off release benchmarks for performance.

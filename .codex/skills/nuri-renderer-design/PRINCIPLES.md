# Compression and N+1 principles

## Casey Muratori: compression-oriented programming

Primary sources:

- [Semantic Compression](https://caseymuratori.com/blog_0015)
- [Complexity and Granularity](https://caseymuratori.com/blog_0016)
- [Defining a Single Enumerant](https://caseymuratori.com/blog_0017)

Apply these ideas:

- Start with concrete working cases and compress only after repetition is visible.
- Require at least two real uses before extracting reusable machinery.
- Optimize total lifetime human work: writing, debugging, changing, adapting, and
  operating the code—not physical line count alone.
- Give frequent domain operations stable names so source mirrors the renderer's
  language.
- Keep unique code local instead of routing it through an abstraction made for a
  different repetition.
- Build high-level operations by progressively bundling lower-level operations.
  Keep the lower granularity available so exceptional work does not require a
  wholesale rewrite.
- Prefer procedural locality for operations that change together. Do not scatter a
  single operation across a class hierarchy merely to organize by nouns.
- When usage code is missing, write representative usage first instead of guessing
  the interface.

Compression is semantic, not minification. A shorter implementation is worse when
it hides ordering, ownership, synchronization, or failure invariants.

## Ryan Fleury: codepaths, batches, and constraints

Primary sources:

- [The Codepath Combinatoric Explosion](https://www.dgtlgrove.com/p/the-codepath-combinatoric-explosion)
- [The Easiest Way To Handle Errors Is To Not Have Them](https://www.dgtlgrove.com/p/the-easiest-way-to-handle-errors)
- [Multi-Threading & Mutation](https://www.dgtlgrove.com/p/multi-threading-and-mutation)
- [Emergence and Composition](https://www.dgtlgrove.com/p/emergence-and-composition)

Apply these ideas:

- Every independent branch/state combination adds execution possibilities, test
  burden, and failure surface. Collapse cases into common exercised codepaths.
- Errors are data/cases, not a separate metaphysical category. Let recoverable
  cases flow through shared paths; eliminate impossible cases structurally.
- Operate on batches of data for both simpler memory ownership and better hardware
  behavior. Allocate batch lifetime together when possible.
- Treat concurrency, lifetime, and determinism as shaping constraints early enough
  to avoid accidental mutation and later rewrites.
- Preserve multiple useful granularities. Low-level representations and higher-level
  domain interpretations coexist; one does not erase the other.

## Nuri's project N+1 rule

N+1 here is a project design test, not the database-query problem and not attributed
to either author above:

> Given N supported backends/features/states, adding one more should extend a table,
> tagged variant, descriptor, adapter, or registration point without rewriting N
> unrelated callers.

Use it at stable module interfaces and data descriptions. Reject it when it would:

- add indirection or branching inside a measured per-draw/per-dispatch loop;
- create a seam with only one real adapter;
- replace a small explicit switch/table with a large class hierarchy;
- permit invalid combinations that every consumer must validate;
- obscure GPU ownership, ordering, or completion requirements.

Good N+1 examples include typed feature bits, descriptor tables, tagged compiled-pass
records, private backend adapters, and immutable capabilities. Poor examples include
one virtual object per draw, speculative plugin points, and `void*` extension bags.

## Review questions

1. What concrete repetition is being compressed?
2. Which codepath/state combinations disappear?
3. Does the interface expose one source of truth?
4. Can a high-level operation be replaced by a few lower-level operations?
5. What does the N+1 addition edit, and what stays untouched?
6. Are ownership, ordering, and performance characteristics explicit?
7. Is the abstraction outside the hot loop or supported by measurement?

---
name: nuri-task-workflow
description: Routes substantive Nuri work through durable notes, bounded subagent coordination, and appropriate renderer validation. Use for multi-step research, design, diagnosis, performance investigation, or feature implementation with code, shader, build, or manifest changes; do not use for simple questions, brief explanations, status checks, or ordinary conversation.
---

# Nuri Task Workflow

## Classify The Request

- **Simple:** a focused question or explanation that can be answered directly without multi-file investigation. Answer normally and stop.
- **Research:** substantive exploration, diagnosis, design, planning, or performance investigation without implementation edits.
- **Implementation:** feature work that changes code, shaders, build files, or manifests.
- Start simple when uncertain. Escalate to research or implementation when the work becomes multi-step.

## Shared Substantive Workflow

1. Create or reuse `.scratch/<task-slug>.md` before deep work. Record the objective, constraints, decisions, evidence, commands and results, artifact paths, delegated summaries, and unresolved questions. Link to large logs instead of copying them. Never record secrets.
2. Keep the primary agent as coordinator and integrator. Delegate only concrete, independent, bounded work that can run in parallel without overlapping edits.
3. Prefer one or two read-heavy subagents. Keep nesting at depth one and keep one writer for overlapping code. Require concise summaries with file paths and evidence.
4. When selectable and available, prefer Luna with low reasoning for deterministic searches, inventories, extraction, and report summaries; prefer Terra with medium reasoning for code-path analysis, test investigation, and review. Use deeper reasoning only for ambiguity or high-risk decisions.
5. Do not spawn an agent merely to satisfy the pattern. If delegation would cost more than it saves, keep the coordinator pattern without fan-out and note the reason in the scratchpad.
6. Update the scratchpad after material findings, decisions, delegated results, validation runs, and before handoff or likely context compaction.

## Research Workflow

- State the research question and evidence needed in the scratchpad.
- Divide genuinely independent investigation branches between bounded subagents; keep synthesis and decisions in the primary task.
- Do not edit implementation files or run expensive GPU validation unless the research explicitly requires measurements or reproduction.
- Finish with conclusions, supporting evidence, uncertainty, and next steps in both the scratchpad and the user-facing response.

## Implementation Workflow

1. Define observable acceptance criteria and the intended validation evidence before editing.
2. Use the smallest relevant build and affected tests, then run all renderer feature gates below.
3. Follow `../nuri-autotests/SKILL.md` for a deterministic Niagara Bistro stress scenario with rapid translation, sharp turns, reversals, and broad traversal through substantial visibility and occlusion changes.
4. Follow `../nuri-benchmarks/SKILL.md` for authoritative frame-time evidence on the rapid-camera Bistro workload. Autotest timings are not the performance oracle.
5. Follow `../nuri-snapshots/SKILL.md` for the smallest relevant visual-regression comparison after the change.
6. Relevant autotest, benchmark, and snapshot manifests must use GTAO Ultra, Shadows Ultra, and TAA Ultra unless the feature explicitly tests a different quality mode. Record any intentional exception.
7. If a required gate is unavailable or inapplicable, record the concrete reason and remaining risk; never report an unrun gate as passed.

## Baseline And Artifact Safety

- Inspect JSON reports before logs or HTML and preserve stable artifact paths in the scratchpad.
- Never accept, update, copy, or approve benchmark, autotest, or snapshot baselines without explicit user authorization.
- Treat forced runs, dry runs, visible local profiles, and incompatible comparisons as investigative evidence, not authoritative passes.

## Finish

- Reconcile delegated results, implementation state, validation evidence, unresolved risks, and next steps in the scratchpad.
- Give the user a compact outcome-first summary and identify every gate that passed, failed, or was not run.

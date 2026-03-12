# Master Operating Document — Mistral-lens

**Version:** 1.0  
**Date:** <DATE>  
**Owner:** Product + Architecture + Engineering  
**Applies to:** All work under `2.Projects/Mistral-lens/`

## 1. Purpose and Operating Principles

Mistral-lens uses a documentation-first, handoff-driven delivery model with clear role separation and auditable decisions.

### Operating principles

- Single source of truth: Official docs live in Google Drive under `2.Projects/Mistral-lens/`.
- Documentation before implementation: No build work starts without approved functional and technical baselines.
- Explicit ownership: Every artifact has one accountable owner and one review owner.
- Versioned decisions: Major scope or architecture changes require ADR and decision-log updates.
- Handoff discipline: Codex and Claude Code operate through atomic, testable handoffs.
- Traceability: Every sprint item maps to requirement IDs and acceptance criteria.
- No silent changes: Any scope, timeline, quality, or architecture deviation is logged and communicated.

## 2. Roles and Responsibilities (Simple RACI)

| Activity | Product/PO | Architecture (Claude Code) | Engineering (Codex + Claude Code) | Conversational Claude/ChatGPT |
|---|---|---|---|---|
| Product vision, outcomes, priorities | A/R | C | I | C |
| Functional requirements | A/R | C | I | C |
| Technical architecture and ADRs | I | A/R | C | I |
| Task decomposition for execution | I | A/R | C | I |
| Implementation | I | C | A/R | I |
| QA strategy and validation gates | C | A/R | R | I |
| Release readiness decision | A | R | R | I |
| Documentation updates | A (Functional/PM) | A (Technical) | R | C |
| Brainstorming and discovery | A | C | I | R |

Legend: `A` Accountable, `R` Responsible, `C` Consulted, `I` Informed.

## 3. Standard Documentation Structure

### 3.1 Required files

**Functional**
- `functional_requirements_Mistral-lens.md`
- `functional_requirements_Mistral-lens.docx`

**Technical**
- `technical_architecture_Mistral-lens.md`
- `technical_architecture_Mistral-lens.docx`

**Project Management**
- `project_tracker_Mistral-lens.md`
- `project_tracker_Mistral-lens.docx`
- `decision_log_Mistral-lens.md`
- `decision_log_Mistral-lens.docx`
- `handoff_log_Mistral-lens.md`
- `handoff_log_Mistral-lens.docx`

### 3.2 `.md` + `.docx` format policy

- `.md` is the working/editing format for AI collaboration and change diffs.
- `.docx` is the business distribution format for stakeholders.
- Both formats must always match version, date, and section content.
- Any update is incomplete until both file formats are synchronized.

### 3.3 What gets updated at each phase

| Phase | Mandatory updates |
|---|---|
| Discovery | `project_tracker_*`, `functional_requirements_*` draft scope, `decision_log_*` assumptions |
| Functional design | `functional_requirements_*` BR/FR/NFR + acceptance criteria, tracker priorities |
| Technical design | `technical_architecture_*`, ADR entries, risk register in tracker |
| Implementation | `project_tracker_*` sprint/task status, `handoff_log_*` entries, impacted FR references |
| QA/validation | QA evidence in tracker, DoD checklist, defects + resolution notes |
| Handoff/maintenance | release notes, known issues, ownership transitions, backlog re-prioritization |

## 4. Official Naming Conventions

### 4.1 Project, version, folders, files

- Project name must be exactly: `Mistral-lens`
- Root path: `2.Projects/Mistral-lens/`
- Version format: `Major.Minor` (example: `2.1`)
- Date format: `YYYY-MM-DD`
- Requirement IDs: `BR-xx`, `FR-xx`, `NFR-xx`
- Decision IDs: `ADR-xxx`
- Handoff IDs: `HO-xxx`
- Sprint IDs: `S-xx`

### 4.2 Branch conventions

- `main` for protected production-ready code
- `develop` for integration
- `feature/<area>-<short-description>`
- `fix/<area>-<issue>`
- `hotfix/<release>-<issue>`

Example: `feature/recommendation-ranking-refactor`

### 4.3 Commit conventions

Format: `<type>(<scope>): <summary>`  
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

Example: `docs(pm): update project_tracker_Mistral-lens to v1.3`

### 4.4 Tag conventions

- Release: `v<MAJOR>.<MINOR>.<PATCH>`
- Milestone: `ms-<quarter>-<name>`

Examples: `v1.2.0`, `ms-2026Q2-discovery-complete`

## 5. Operational Workflow by Phase

### 5.1 Discovery

Checklist:
- Define problem statement, goals, constraints, KPIs.
- Capture assumptions, dependencies, unknowns.
- Produce initial scope and non-goals.
- Log open questions and decision deadlines.

Exit criteria:
- Approved discovery summary and prioritized opportunity list.

### 5.2 Functional design

Checklist:
- Create BR/FR/NFR with IDs and measurable acceptance criteria.
- Define user journeys and edge cases.
- Define MVP vs later-phase scope boundaries.
- Confirm Product sign-off.

Exit criteria:
- `functional_requirements_Mistral-lens.*` approved and versioned.

### 5.3 Technical design

Checklist:
- Define architecture, modules, interfaces, data boundaries.
- Record ADRs for all non-trivial decisions.
- Produce delivery slices mapped to FR IDs.
- Confirm architecture sign-off.

Exit criteria:
- `technical_architecture_Mistral-lens.*` approved and handoff-ready.

### 5.4 Implementation

Checklist:
- Create atomic tasks with file-level scope and acceptance criteria.
- Execute tasks through Codex/Claude Code handoff workflow.
- Update tracker daily with status, blockers, risks.
- Maintain requirement traceability.

Exit criteria:
- All planned tasks complete with linked QA evidence.

### 5.5 QA/validation

Checklist:
- Execute functional, regression, and non-functional checks.
- Validate against DoD and release checklist.
- Log defects by severity and disposition.
- Confirm stakeholder acceptance for critical flows.

Exit criteria:
- No open critical defects; release readiness approved.

### 5.6 Handoff and maintenance

Checklist:
- Publish release notes and known limitations.
- Transfer operational ownership and support rules.
- Update backlog and technical debt register.
- Schedule post-release review.

Exit criteria:
- Stable ownership and monitored production behavior.

## 6. Communication Protocol

### 6.1 Channels

- Product/PO channel: scope, priorities, decisions.
- Architecture channel: design, ADR, technical risk.
- Engineering channel: implementation status, blockers, QA results.
- Tracker documents in Drive: official record and source of truth.

### 6.2 Cadence

- Daily async update: Engineering to Product + Architecture.
- 2x weekly sync: Product + Architecture + Engineering.
- Weekly planning/review: backlog, risks, release readiness.
- Ad-hoc escalation: within 24h for critical blockers.

### 6.3 Status update template (daily)

- Date: `<DATE>`
- Sprint: `<S-xx>`
- Completed: `<items>`
- In progress: `<items>`
- Blockers: `<item + owner + needed by>`
- Risks: `<risk + impact + mitigation>`
- Next 24h: `<planned work>`

### 6.4 Escalation and decision path

1. Engineer flags issue in `project_tracker_Mistral-lens`.
2. Architecture assesses technical options and impact.
3. Product/PO decides scope/time/quality trade-off.
4. Decision recorded in `decision_log_Mistral-lens` with owner and date.
5. Tracker updated with action and due date.

## 7. Codex ↔ Claude Code Execution Workflow

### 7.1 Responsibility split

- Claude Code: architecture authority, task design, standards, QA strategy, final technical validation.
- Codex: implementation execution, refactoring, tests, bug fixes, documentation updates per task.
- Both: uphold coding standards, traceability, and delivery quality.

### 7.2 Task handoff rules

- One atomic task per handoff.
- Each task must include objective, file scope, constraints, acceptance criteria, test expectations, and out-of-scope notes.
- No ambiguous ownership; each task has one accountable reviewer.
- Handoff IDs logged in `handoff_log_Mistral-lens`.

### 7.3 Review and validation loop

1. Claude Code issues task packet.
2. Codex implements and reports outcomes.
3. Claude Code reviews for architecture fit, regression risk, and quality gates.
4. If rejected, rework instructions are explicit and scoped.
5. If approved, tracker and relevant docs are updated immediately.

## 8. Brainstorming Workflow with Conversational Claude/ChatGPT

### 8.1 How to capture insights

- Use conversational sessions only for ideation, exploration, and framing.
- Capture every session output as: hypotheses, opportunities, assumptions, risks.
- Store raw notes in `Context/` folder with date and owner metadata.

### 8.2 How to convert ideas into actionable requirements

1. Convert ideas into problem statements.
2. Map each statement to BR/FR/NFR candidates.
3. Define measurable acceptance criteria.
4. Prioritize using impact/effort/risk.
5. Submit to Product for scope decision.
6. Update `functional_requirements_Mistral-lens` and tracker.

## 9. Google Drive Documentation Governance

### 9.1 Recommended folder structure

`2.Projects/Mistral-lens/`
- `Functional/`
- `Technical/`
- `Project Management/`
- `Context/`
- `Archive/`

### 9.2 Version control rules

- Every document must have version header and version history table.
- Minor updates: `x.y -> x.(y+1)`.
- Major scope/structure changes: `(x+1).0`.
- No overwrite without changelog entry.

### 9.3 Single source of truth policy

- Drive is the official collaboration source.
- Repository markdown mirrors approved Drive content for engineering traceability.
- If mismatch occurs, latest approved Drive version wins, then sync repo copy.

## 10. QA Framework

### 10.1 Quality criteria

- Requirement coverage: every implemented item maps to FR/NFR.
- Functional correctness: acceptance criteria met.
- Regression safety: no breakage in critical flows.
- Technical quality: maintainable structure, standards compliance.
- Documentation quality: updated, versioned, and synchronized.

### 10.2 Definition of Done (DoD)

- Feature implemented and reviewed.
- Acceptance criteria passed.
- QA evidence attached in tracker.
- Relevant docs updated in `.md` and `.docx`.
- No critical/high unresolved defects.
- Product/PO sign-off captured.

### 10.3 Release readiness checklist

- Scope complete for release target.
- Risk register reviewed and accepted.
- Defect triage completed.
- Rollback/mitigation plan documented.
- Stakeholder communication prepared.
- Tag/version assigned and logged.

## 11. Ready-to-Use Templates

### 11.1 Sprint update template

| Field | Value |
|---|---|
| Sprint | `<S-xx>` |
| Date | `<DATE>` |
| Owner | `<OWNER>` |
| Objectives | `<objective list>` |
| Done | `<completed items>` |
| In Progress | `<items>` |
| Blockers | `<blocker + owner + ETA>` |
| Risks | `<risk + mitigation>` |
| Next Actions | `<next 3 actions>` |

### 11.2 Change request template

| Field | Value |
|---|---|
| CR ID | `CR-<YYYYMMDD>-<NN>` |
| Requester | `<OWNER>` |
| Date | `<DATE>` |
| Description | `<change summary>` |
| Reason | `<business/technical driver>` |
| Impact | `<scope/time/cost/risk>` |
| Affected Docs | `<filenames>` |
| Decision | `<approved/rejected/deferred>` |
| Decision Owner | `<OWNER>` |

### 11.3 Technical ADR template

| Field | Value |
|---|---|
| ADR ID | `ADR-<NNN>` |
| Date | `<DATE>` |
| Status | `<proposed/accepted/superseded>` |
| Context | `<problem and constraints>` |
| Decision | `<chosen approach>` |
| Alternatives | `<considered options>` |
| Consequences | `<trade-offs and impact>` |
| Related FR/NFR | `<IDs>` |

### 11.4 Product decision log template

| Field | Value |
|---|---|
| Decision ID | `PD-<YYYYMMDD>-<NN>` |
| Date | `<DATE>` |
| Decision | `<what was decided>` |
| Options Considered | `<A/B/C>` |
| Rationale | `<why>` |
| Owner | `<PO/Stakeholder>` |
| Impacts | `<scope/timeline/quality>` |
| Follow-up Actions | `<owners + due dates>` |

## 12. Common Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Documentation drift between `.md` and `.docx` | Misalignment and rework | Mandatory sync check at phase gate |
| Ambiguous role boundaries | Delays and duplicated effort | Enforce RACI + handoff templates |
| Untracked scope changes | Timeline and quality erosion | Require CR + decision log entry |
| Weak requirement clarity | Defects and rework | Acceptance criteria mandatory before build |
| Handoff quality inconsistency | Implementation churn | Atomic tasks + strict review loop |
| Decision latency | Blocked execution | 24h escalation SLA with named owner |
| Knowledge trapped in chat | Lost context | Capture all outcomes in Drive context/docs |

## 13. Day 1 to Day 7 Onboarding/Activation Plan

### Day 1

- Create Drive structure under `2.Projects/Mistral-lens/`.
- Create all canonical docs in `.md` and `.docx`.
- Assign owners and reviewers.

### Day 2

- Run discovery workshop.
- Populate goals, constraints, assumptions, open questions.

### Day 3

- Draft `functional_requirements_Mistral-lens.*` with BR/FR/NFR IDs.
- Product review and priority ranking.

### Day 4

- Draft `technical_architecture_Mistral-lens.*` and initial ADRs.
- Architecture review and approval.

### Day 5

- Build initial `project_tracker_Mistral-lens.*`.
- Create first sprint plan and handoff backlog.

### Day 6

- Start implementation through Codex ↔ Claude Code task loop.
- Begin daily status cadence and risk tracking.

### Day 7

- Execute first QA gate on completed items.
- Run readiness review and adjust week-2 priorities.

## 14. Appendix: Sample Folder/File Tree

```text
2.Projects/
└── Mistral-lens/
    ├── Functional/
    │   ├── functional_requirements_Mistral-lens.md
    │   └── functional_requirements_Mistral-lens.docx
    ├── Technical/
    │   ├── technical_architecture_Mistral-lens.md
    │   └── technical_architecture_Mistral-lens.docx
    ├── Project Management/
    │   ├── project_tracker_Mistral-lens.md
    │   ├── project_tracker_Mistral-lens.docx
    │   ├── decision_log_Mistral-lens.md
    │   ├── decision_log_Mistral-lens.docx
    │   ├── handoff_log_Mistral-lens.md
    │   └── handoff_log_Mistral-lens.docx
    ├── Context/
    │   ├── discovery_notes_<DATE>.md
    │   ├── brainstorming_session_<DATE>.md
    │   └── stakeholder_inputs_<DATE>.docx
    └── Archive/
        └── superseded_versions/
```

# Skill Benchmark: cuequivariance

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuequivariance`
- Evaluation date: 2026-10-19
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:64264cc94192671086b1204edef9cf61e0c0710d636da7afc0c1f0c594e2cd55` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 85.5% — baseline ran, but no comparable score was available; uplift unavailable | 91.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 83.3% (±0.0 points) | 83.3% → 100.0% (+16.7 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 84.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 94.2% → 92.9% (-1.3 points) | 88.3% → 99.2% (+10.9 points) |
| Efficiency | 66.3% — baseline ran, but no comparable score was available; uplift unavailable | 72.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,881,399 | 7,298,202 | -1,416,803 | -19.41% | skill 6/6; base 6/6 |
| claude-code | core-001 | 956,726 | 1,188,219 | -231,493 | -19.48% | skill 1/1; base 1/1 |
| claude-code | core-002 | 510,535 | 869,031 | -358,496 | -41.25% | skill 1/1; base 1/1 |
| claude-code | core-003 | 710,993 | 2,235,618 | -1,524,625 | -68.20% | skill 1/1; base 1/1 |
| claude-code | core-004 | 708,256 | 959,125 | -250,869 | -26.16% | skill 1/1; base 1/1 |
| claude-code | core-005 | 2,964,658 | 2,015,952 | +948,706 | +47.06% | skill 1/1; base 1/1 |
| claude-code | core-006 | 30,231 | 30,257 | -26 | -0.09% | skill 1/1; base 1/1 |
| codex | All cases | 2,400,367 | 2,780,675 | -380,308 | -13.68% | skill 6/6; base 6/6 |
| codex | core-001 | 720,238 | 494,550 | +225,688 | +45.64% | skill 1/1; base 1/1 |
| codex | core-002 | 474,477 | 618,515 | -144,038 | -23.29% | skill 1/1; base 1/1 |
| codex | core-003 | 429,875 | 645,457 | -215,582 | -33.40% | skill 1/1; base 1/1 |
| codex | core-004 | 356,473 | 692,463 | -335,990 | -48.52% | skill 1/1; base 1/1 |
| codex | core-005 | 405,363 | 315,871 | +89,492 | +28.33% | skill 1/1; base 1/1 |
| codex | core-006 | 13,941 | 13,819 | +122 | +0.88% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,281,766 | 10,078,877 | -1,797,111 | -17.83% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within SKILL.md:
  "### IrDictPolynomial descriptors" in SKILL.md (lines 247-257)
  vs "### IrDictPolynomial" in SKILL.md (lines 287-301)
  vs "### split_polynomial_by_irreps" in SKILL.md (lines 312-322)
  vs "# Split an operand into per-irrep pieces (for ir_dict interface)" in SKILL.md (lines 336-338) (`SKILL.md:247`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cuequivariance/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (310 chars, recommend 50-150) (`skills/cuequivariance/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/cuequivariance/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/cuequivariance/SKILL.md`)
- 3 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.

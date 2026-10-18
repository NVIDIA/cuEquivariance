# Skill Benchmark: cuequivariance

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuequivariance`
- Evaluation date: 2026-10-18
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
| Overall | 88.7% — baseline ran, but no comparable score was available; uplift unavailable | 82.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 50.0% → 83.3% (+33.3 points) | 83.3% → 66.7% (-16.6 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 86.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 90.0% → 98.8% (+8.8 points) | 86.7% → 92.1% (+5.4 points) |
| Efficiency | 75.6% — baseline ran, but no comparable score was available; uplift unavailable | 66.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,991,254 | 7,417,446 | -2,426,192 | -32.71% | skill 6/6; base 6/6 |
| claude-code | core-001 | 1,552,225 | 1,548,257 | +3,968 | +0.26% | skill 1/1; base 1/1 |
| claude-code | core-002 | 596,064 | 1,089,910 | -493,846 | -45.31% | skill 1/1; base 1/1 |
| claude-code | core-003 | 872,820 | 1,146,419 | -273,599 | -23.87% | skill 1/1; base 1/1 |
| claude-code | core-004 | 663,184 | 865,437 | -202,253 | -23.37% | skill 1/1; base 1/1 |
| claude-code | core-005 | 1,276,221 | 2,736,942 | -1,460,721 | -53.37% | skill 1/1; base 1/1 |
| claude-code | core-006 | 30,740 | 30,481 | +259 | +0.85% | skill 1/1; base 1/1 |
| codex | All cases | 2,155,811 | 4,007,025 | -1,851,214 | -46.20% | skill 6/6; base 6/6 |
| codex | core-001 | 420,443 | 845,088 | -424,645 | -50.25% | skill 1/1; base 1/1 |
| codex | core-002 | 473,578 | 731,568 | -257,990 | -35.27% | skill 1/1; base 1/1 |
| codex | core-003 | 436,256 | 849,609 | -413,353 | -48.65% | skill 1/1; base 1/1 |
| codex | core-004 | 361,551 | 1,414,735 | -1,053,184 | -74.44% | skill 1/1; base 1/1 |
| codex | core-005 | 450,019 | 152,246 | +297,773 | +195.59% | skill 1/1; base 1/1 |
| codex | core-006 | 13,964 | 13,779 | +185 | +1.34% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,147,065 | 11,424,471 | -4,277,406 | -37.44% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cuequivariance/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (310 chars, recommend 50-150) (`skills/cuequivariance/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/cuequivariance/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/cuequivariance/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/cuequivariance/SKILL.md`)
- 2 additional finding(s) are available in the full evaluation artifacts.

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

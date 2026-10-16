# Skill Benchmark: cuequivariance-jax

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuequivariance-jax`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:f653a01cf39e3dffe96fbacf75b1b1ee214a79c4fc5b682f49f31e6963d7bb1a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 81.2% — baseline ran, but no comparable score was available; uplift unavailable | 87.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 33.3% → 50.0% (+16.7 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 84.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 97.9% → 100.0% (+2.1 points) | 90.8% → 87.9% (-2.9 points) |
| Efficiency | 72.1% — baseline ran, but no comparable score was available; uplift unavailable | 63.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,770,880 | 5,344,900 | +1,425,980 | +26.68% | skill 6/6; base 6/6 |
| claude-code | jax-001 | 1,092,116 | 1,209,345 | -117,229 | -9.69% | skill 1/1; base 1/1 |
| claude-code | jax-002 | 1,669,333 | 476,301 | +1,193,032 | +250.48% | skill 1/1; base 1/1 |
| claude-code | jax-003 | 745,879 | 887,128 | -141,249 | -15.92% | skill 1/1; base 1/1 |
| claude-code | jax-004 | 2,637,995 | 2,282,069 | +355,926 | +15.60% | skill 1/1; base 1/1 |
| claude-code | jax-005 | 596,440 | 460,588 | +135,852 | +29.50% | skill 1/1; base 1/1 |
| claude-code | jax-006 | 29,117 | 29,469 | -352 | -1.19% | skill 1/1; base 1/1 |
| codex | All cases | 3,927,148 | 2,136,742 | +1,790,406 | +83.79% | skill 6/6; base 6/6 |
| codex | jax-001 | 793,152 | 614,925 | +178,227 | +28.98% | skill 1/1; base 1/1 |
| codex | jax-002 | 304,987 | 323,916 | -18,929 | -5.84% | skill 1/1; base 1/1 |
| codex | jax-003 | 1,028,052 | 595,600 | +432,452 | +72.61% | skill 1/1; base 1/1 |
| codex | jax-004 | 1,727,473 | 556,340 | +1,171,133 | +210.51% | skill 1/1; base 1/1 |
| codex | jax-005 | 60,043 | 32,627 | +27,416 | +84.03% | skill 1/1; base 1/1 |
| codex | jax-006 | 13,441 | 13,334 | +107 | +0.80% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 10,698,028 | 7,481,642 | +3,216,386 | +42.99% | skill 12/12; base 12/12 |

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

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cuequivariance-jax/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (276 chars, recommend 50-150) (`skills/cuequivariance-jax/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/cuequivariance-jax/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/cuequivariance-jax/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/cuequivariance-jax/SKILL.md`)
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

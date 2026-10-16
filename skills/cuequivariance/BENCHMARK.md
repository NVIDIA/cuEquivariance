# Skill Benchmark: cuequivariance

> **Overall verdict: NEUTRAL — One or more dimensions remain below PASS**

Live evaluation did not show a material gain or regression. Collect more evidence or improve the skill before making a publication decision.

## Evaluation Metadata

- Skill: `cuequivariance`
- Evaluation date: 2026-10-16
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
| Overall | 86.3% — baseline ran, but no comparable score was available; uplift unavailable | 82.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 75.0% (-25.0 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 96.7% (-3.3 points) |
| Discoverability | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 83.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 95.4% → 95.8% (+0.4 points) | 88.8% → 80.0% (-8.8 points) |
| Efficiency | 71.9% — baseline ran, but no comparable score was available; uplift unavailable | 71.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,494,094 | 6,764,067 | -2,269,973 | -33.56% | skill 6/6; base 6/6 |
| claude-code | core-001 | 935,725 | 1,050,205 | -114,480 | -10.90% | skill 1/1; base 1/1 |
| claude-code | core-002 | 558,016 | 1,212,941 | -654,925 | -53.99% | skill 1/1; base 1/1 |
| claude-code | core-003 | 924,549 | 1,870,785 | -946,236 | -50.58% | skill 1/1; base 1/1 |
| claude-code | core-004 | 717,396 | 1,190,142 | -472,746 | -39.72% | skill 1/1; base 1/1 |
| claude-code | core-005 | 1,327,938 | 1,409,548 | -81,610 | -5.79% | skill 1/1; base 1/1 |
| claude-code | core-006 | 30,470 | 30,446 | +24 | +0.08% | skill 1/1; base 1/1 |
| codex | All cases | 2,407,936 | 3,244,738 | -836,802 | -25.79% | skill 6/6; base 6/6 |
| codex | core-001 | 472,531 | 660,932 | -188,401 | -28.51% | skill 1/1; base 1/1 |
| codex | core-002 | 626,920 | 924,625 | -297,705 | -32.20% | skill 1/1; base 1/1 |
| codex | core-003 | 561,064 | 742,408 | -181,344 | -24.43% | skill 1/1; base 1/1 |
| codex | core-004 | 644,768 | 817,679 | -172,911 | -21.15% | skill 1/1; base 1/1 |
| codex | core-005 | 88,355 | 85,281 | +3,074 | +3.60% | skill 1/1; base 1/1 |
| codex | core-006 | 14,298 | 13,813 | +485 | +3.51% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,902,030 | 10,008,805 | -3,106,775 | -31.04% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **NEUTRAL** | 2 agent(s); 6 task(s) |

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

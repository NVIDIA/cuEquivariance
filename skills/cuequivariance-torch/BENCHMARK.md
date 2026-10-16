# Skill Benchmark: cuequivariance-torch

> **Overall verdict: NEUTRAL — One or more dimensions remain below PASS**

Live evaluation did not show a material gain or regression. Collect more evidence or improve the skill before making a publication decision.

## Evaluation Metadata

- Skill: `cuequivariance-torch`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:c59a90389aabc4ce9d6756217b3f98021106bda2f5a6df42bb36c78c5a0135db` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 81.8% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 50.0% → 50.0% (±0.0 points) | Not available |
| Correctness | 100.0% → 100.0% (±0.0 points) | Not available |
| Discoverability | 87.2% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 90.1% → 100.0% (+9.9 points) | Not available |
| Efficiency | 71.9% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 7,423,526 | 6,490,723 | +932,803 | +14.37% | skill 6/6; base 6/6 |
| claude-code | torch-001 | 1,379,936 | 1,308,078 | +71,858 | +5.49% | skill 1/1; base 1/1 |
| claude-code | torch-002 | 1,182,955 | 603,673 | +579,282 | +95.96% | skill 1/1; base 1/1 |
| claude-code | torch-003 | 1,161,987 | 928,849 | +233,138 | +25.10% | skill 1/1; base 1/1 |
| claude-code | torch-004 | 1,286,524 | 1,044,323 | +242,201 | +23.19% | skill 1/1; base 1/1 |
| claude-code | torch-005 | 2,382,097 | 2,576,008 | -193,911 | -7.53% | skill 1/1; base 1/1 |
| claude-code | torch-006 | 30,027 | 29,792 | +235 | +0.79% | skill 1/1; base 1/1 |
| codex | All cases | 5,264,185 | 4,068,564 | N/A | N/A | skill 5/6; base 6/6 |
| codex | torch-001 | N/A | 1,321,564 | N/A | N/A | skill 0/1; base 1/1 |
| codex | torch-002 | 1,651,497 | 768,903 | +882,594 | +114.79% | skill 1/1; base 1/1 |
| codex | torch-003 | 1,103,907 | 689,022 | +414,885 | +60.21% | skill 1/1; base 1/1 |
| codex | torch-004 | 520,123 | 425,385 | +94,738 | +22.27% | skill 1/1; base 1/1 |
| codex | torch-005 | 1,974,911 | 850,079 | +1,124,832 | +132.32% | skill 1/1; base 1/1 |
| codex | torch-006 | 13,747 | 13,611 | +136 | +1.00% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 12,687,711 | 10,559,287 | N/A | N/A | skill 11/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **NEUTRAL** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cuequivariance-torch/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (363 chars, recommend 50-150) (`skills/cuequivariance-torch/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/cuequivariance-torch/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No mention of error handling or validation (`skills/cuequivariance-torch/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/cuequivariance-torch/SKILL.md`)
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

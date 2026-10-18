# Skill Benchmark: cuequivariance-torch

> **Overall verdict: NEUTRAL — One or more dimensions remain below PASS**

Live evaluation did not show a material gain or regression. Collect more evidence or improve the skill before making a publication decision.

## Evaluation Metadata

- Skill: `cuequivariance-torch`
- Evaluation date: 2026-10-18
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
| Overall | 84.0% — baseline ran, but no comparable score was available; uplift unavailable | 74.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 50.0% → 58.3% (+8.3 points) | 50.0% → 41.7% (-8.3 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 96.7% → 100.0% (+3.3 points) |
| Discoverability | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 83.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 92.2% → 92.9% (+0.7 points) | 82.4% → 91.7% (+9.3 points) |
| Efficiency | 79.6% — baseline ran, but no comparable score was available; uplift unavailable | 58.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,992,274 | 6,906,967 | -1,914,693 | -27.72% | skill 6/6; base 6/6 |
| claude-code | torch-001 | 795,386 | 1,870,442 | -1,075,056 | -57.48% | skill 1/1; base 1/1 |
| claude-code | torch-002 | 475,055 | 947,637 | -472,582 | -49.87% | skill 1/1; base 1/1 |
| claude-code | torch-003 | 896,363 | 1,413,656 | -517,293 | -36.59% | skill 1/1; base 1/1 |
| claude-code | torch-004 | 912,314 | 1,249,324 | -337,010 | -26.98% | skill 1/1; base 1/1 |
| claude-code | torch-005 | 1,883,055 | 1,396,040 | +487,015 | +34.89% | skill 1/1; base 1/1 |
| claude-code | torch-006 | 30,101 | 29,868 | +233 | +0.78% | skill 1/1; base 1/1 |
| codex | All cases | 5,514,111 | 7,798,241 | -2,284,130 | -29.29% | skill 6/6; base 6/6 |
| codex | torch-001 | 1,261,467 | 1,871,775 | -610,308 | -32.61% | skill 1/1; base 1/1 |
| codex | torch-002 | 1,070,595 | 970,046 | +100,549 | +10.37% | skill 1/1; base 1/1 |
| codex | torch-003 | 1,172,773 | 1,222,710 | -49,937 | -4.08% | skill 1/1; base 1/1 |
| codex | torch-004 | 556,283 | 2,789,080 | -2,232,797 | -80.05% | skill 1/1; base 1/1 |
| codex | torch-005 | 1,439,357 | 931,065 | +508,292 | +54.59% | skill 1/1; base 1/1 |
| codex | torch-006 | 13,636 | 13,565 | +71 | +0.52% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 10,506,385 | 14,705,208 | -4,198,823 | -28.55% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **NEUTRAL** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- Semantic Version Validation: No semantic version label present; resource will use commit-hash history (opting back out of an existing label is allowed)
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 2 files for PII
- Code Integrity & Hygiene: Checking 1 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 2 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No Python scripts found in scripts/
- Context Deduplication: Collected 1 file(s)
- Inter-Skill Deduplication: Parsed skill 'cuequivariance-torch': 151 char description
- AGENT_EVAL: Tier 3 evaluation complete: verdict NEUTRAL; best agent claude-code

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

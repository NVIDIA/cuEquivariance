# Skill Benchmark: cuequivariance-jax

> **Overall verdict: NEUTRAL — One or more dimensions remain below PASS**

Live evaluation did not show a material gain or regression. Collect more evidence or improve the skill before making a publication decision.

## Evaluation Metadata

- Skill: `cuequivariance-jax`
- Evaluation date: 2026-10-19
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
| Overall | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 100.0% (+16.7 points) | 100.0% → 66.7% (-33.3 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 96.7% (-3.3 points) |
| Discoverability | 87.0% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 97.9% → 96.7% (-1.2 points) | 97.9% → 92.9% (-5.0 points) |
| Efficiency | 81.6% — baseline ran, but no comparable score was available; uplift unavailable | 65.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,437,471 | 5,353,635 | +83,836 | +1.57% | skill 6/6; base 6/6 |
| claude-code | jax-001 | 794,408 | 1,156,143 | -361,735 | -31.29% | skill 1/1; base 1/1 |
| claude-code | jax-002 | 1,006,269 | 471,710 | +534,559 | +113.32% | skill 1/1; base 1/1 |
| claude-code | jax-003 | 797,985 | 611,625 | +186,360 | +30.47% | skill 1/1; base 1/1 |
| claude-code | jax-004 | 1,596,338 | 1,666,259 | -69,921 | -4.20% | skill 1/1; base 1/1 |
| claude-code | jax-005 | 1,212,898 | 1,418,401 | -205,503 | -14.49% | skill 1/1; base 1/1 |
| claude-code | jax-006 | 29,573 | 29,497 | +76 | +0.26% | skill 1/1; base 1/1 |
| codex | All cases | 1,725,111 | 3,333,306 | -1,608,195 | -48.25% | skill 6/6; base 6/6 |
| codex | jax-001 | 309,970 | 837,967 | -527,997 | -63.01% | skill 1/1; base 1/1 |
| codex | jax-002 | 263,600 | 398,498 | -134,898 | -33.85% | skill 1/1; base 1/1 |
| codex | jax-003 | 263,329 | 573,627 | -310,298 | -54.09% | skill 1/1; base 1/1 |
| codex | jax-004 | 777,827 | 1,456,163 | -678,336 | -46.58% | skill 1/1; base 1/1 |
| codex | jax-005 | 96,998 | 53,508 | +43,490 | +81.28% | skill 1/1; base 1/1 |
| codex | jax-006 | 13,387 | 13,543 | -156 | -1.15% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,162,582 | 8,686,941 | -1,524,359 | -17.55% | skill 12/12; base 12/12 |

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
- Inter-Skill Deduplication: Parsed skill 'cuequivariance-jax': 133 char description
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

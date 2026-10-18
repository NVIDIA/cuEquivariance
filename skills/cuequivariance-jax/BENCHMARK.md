# Skill Benchmark: cuequivariance-jax

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuequivariance-jax`
- Evaluation date: 2026-10-18
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
| Overall | 86.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 50.0% → 75.0% (+25.0 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 89.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 97.9% → 87.9% (-10.0 points) | 82.9% → 92.9% (+10.0 points) |
| Efficiency | 78.9% — baseline ran, but no comparable score was available; uplift unavailable | 68.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,947,603 | 4,390,017 | +557,586 | +12.70% | skill 6/6; base 6/6 |
| claude-code | jax-001 | 1,583,310 | 908,185 | +675,125 | +74.34% | skill 1/1; base 1/1 |
| claude-code | jax-002 | 692,110 | 531,175 | +160,935 | +30.30% | skill 1/1; base 1/1 |
| claude-code | jax-003 | 1,202,926 | 819,390 | +383,536 | +46.81% | skill 1/1; base 1/1 |
| claude-code | jax-004 | 829,453 | 1,416,254 | -586,801 | -41.43% | skill 1/1; base 1/1 |
| claude-code | jax-005 | 610,198 | 685,455 | -75,257 | -10.98% | skill 1/1; base 1/1 |
| claude-code | jax-006 | 29,606 | 29,558 | +48 | +0.16% | skill 1/1; base 1/1 |
| codex | All cases | 1,936,562 | 2,876,893 | -940,331 | -32.69% | skill 6/6; base 6/6 |
| codex | jax-001 | 310,054 | 759,473 | -449,419 | -59.18% | skill 1/1; base 1/1 |
| codex | jax-002 | 273,170 | 469,747 | -196,577 | -41.85% | skill 1/1; base 1/1 |
| codex | jax-003 | 673,888 | 324,300 | +349,588 | +107.80% | skill 1/1; base 1/1 |
| codex | jax-004 | 611,120 | 1,265,573 | -654,453 | -51.71% | skill 1/1; base 1/1 |
| codex | jax-005 | 54,698 | 44,481 | +10,217 | +22.97% | skill 1/1; base 1/1 |
| codex | jax-006 | 13,632 | 13,319 | +313 | +2.35% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,884,165 | 7,266,910 | -382,745 | -5.27% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

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
- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent codex

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

## Description: <br>
Use when building or debugging cuEquivariance PyTorch tensor products or equivariant GNNs, especially CPU references, layouts, indexing, and gradients. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and researchers building equivariant geometric neural networks in PyTorch use this skill to write, debug, and validate cuequivariance_torch code, including CPU reference calculations, backend selection, irrep layouts and weights, gather/scatter indexing, gradient checks, and trainable message-passing layers. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuEquivariance documentation](https://docs.nvidia.com/cuda/cuequivariance/) <br>
- [cuEquivariance GitHub repository](https://github.com/NVIDIA/cuEquivariance) <br>
- [pip interpreter selection (--python option)](https://pip.pypa.io/en/stable/topics/python-option/) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (5 positive, 1 negative), 1 attempt per task, each run in an isolated sandbox; with-skill runs compared against no-skill baselines. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? <br>
- Efficiency: Did it avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.0% — baseline ran, but no comparable score was available; uplift unavailable | 74.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 50.0% → 58.3% (+8.3 points) | 50.0% → 41.7% (-8.3 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 96.7% → 100.0% (+3.3 points) |
| Discoverability | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 83.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 92.2% → 92.9% (+0.7 points) | 82.4% → 91.7% (+9.3 points) |
| Efficiency | 79.6% — baseline ran, but no comparable score was available; uplift unavailable | 58.1% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.13.0 (source: VERSION file, git commit 531da35 2026-10-18) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>

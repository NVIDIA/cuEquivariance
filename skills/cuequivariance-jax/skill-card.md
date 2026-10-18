## Description: <br>
Use when writing or debugging cuEquivariance JAX polynomials, irrep dictionaries, Flax NNX layers, or CPU/CUDA execution and layouts. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill when writing or debugging JAX code with cuequivariance to execute equivariant polynomials on CPU or GPU, work with dict[Irrep, Array] or RepArray data and layouts, and build Flax NNX equivariant layers for geometric neural networks such as NequIP- and MACE-style GNNs. <br>

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
- [JAX installation](https://docs.jax.dev/en/latest/installation.html) <br>
- [pip interpreter selection](https://pip.pypa.io/en/stable/topics/python-option/) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (5 positive, 1 negative), 1 attempt per task, each run in an isolated sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` plus 50% `token_efficiency`. <br>

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
| Overall | 86.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 50.0% → 75.0% (+25.0 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 89.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 97.9% → 87.9% (-10.0 points) | 82.9% → 92.9% (+10.0 points) |
| Efficiency | 78.9% — baseline ran, but no comparable score was available; uplift unavailable | 68.9% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.13.0 (source: VERSION, CHANGELOG, released 2026-10-15) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>

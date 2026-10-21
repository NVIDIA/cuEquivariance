# Agent skill maintenance

Each immediate child directory containing a `SKILL.md` is one independent skill.
NVSkills CI and nv-carps discover these paths:

```text
skills/
  cuequivariance/
    SKILL.md
    evals/{evals.json,config.yml}
    scripts/requirements.txt
  cuequivariance-jax/
    SKILL.md
    evals/{evals.json,config.yml}
    scripts/requirements.txt
  cuequivariance-torch/
    SKILL.md
    evals/{evals.json,config.yml}
    scripts/requirements.txt
```

Edit the regular files here. The Python package resources at
`<package>/<package>/SKILL.md` are relative symlinks back to these files, keeping
`python -m <package> skill` compatible without maintaining duplicate instructions.
Do not replace the canonical files with symlinks, or nest one skill inside another.
Each skill can be distributed independently; its source-location tables explain
how to find code in a checkout or an installed package.

## Evaluation coverage

Each regression dataset (`evals/evals.json`) contains five positive cases and one
unrelated negative case. The JAX and PyTorch skills also have a separate
`evals/challenges.json` capability dataset; the original six cases are unchanged.
The core cases were adapted from the BioNeMo agent toolkit's cuEquivariance
evaluation ideas; they use this repository's APIs and contain no references to
BioNeMo-only skill resources. The JAX and PyTorch cases exercise their respective
frontends rather than reusing the core skill's expected routing.

| Skill | Positive cases |
| --- | --- |
| Core | Complex Z3 irreps and CG covariance; full SO3 tensor product; split/dense channelwise agreement; layout conversion and rotation; older-API fallback without installing packages |
| JAX | Broadcasting, JIT and finite-difference gradients; dictionary layout conversion; NNX linear normalization and rotation; split polynomial execution; CPU spherical harmonics and backend choice |
| PyTorch | Spherical harmonics and gradcheck; linear layouts and rotation; external per-example weights; gather/scatter with repeated receivers and gradients; CPU backend fallback |

`evals/config.yml` requests an isolated skill workspace and a rebuilt Harbor
environment. In SkillEvaluator 1.5.6, Docker task generation installs
`scripts/requirements.txt`. The `k8s-sandbox` path instead stages skill files into
an existing sandbox; `base_image_mode: rebuild` alone does **not** install these
dependencies there. Verify the actual interpreter and package versions in each
arm before treating a run as an evaluation of the pinned environment. Provision
dependencies equally for both arms and establish a new baseline when changing
the environment. JAX uses `harbor.pre_agent_setup`; Torch uses the native Harbor
environment healthcheck described below.

The JAX and PyTorch setup commands publish `/workspace/.venv` only after the
pinned packages pass a CPU smoke check. Failed environments remain at their
unique `.cueq-env.*` paths for diagnosis; the next attempt creates a fresh one.
An existing `.venv`, including a broken symlink, is preserved and must pass the
check without being replaced. Setup uses binary wheels and skips bytecode
compilation. Torch installs its pinned CPU build with `--no-deps` from
[PyTorch's alternate CPU index](https://download-r2.pytorch.org/whl/cpu/torch/).
Pip selects the compatible wheel for the host interpreter and architecture.
This avoids the original index host that failed certificate verification in
Astra while retaining HTTPS verification and the configured `PIP_CERT` bundle.
A second install resolves Torch's dependencies and cuEquivariance from the normal index,
avoiding dependency-metadata incompatibilities with older host pip versions.
Both arms use the same setup and package pins.

SkillEvaluator 1.5.6 hardcodes 120 seconds for generated `pre_agent_setup`
healthchecks. Torch selects `harbor.task_source: native_harbor` so its repo-owned
`evals/harbor/*/task.toml` files can give installation and the smoke check
600 seconds. This uses the unmodified evaluator's supported native-task path.
The agent and verifier budgets remain 300 and 180 seconds, respectively, and
grading remains `aces_default`. Both arms still run the same six prompts from
`evals/evals.json`. Setup failures remain failures, and Harbor collects
`/workspace/cueq-setup.log` as a task artifact for diagnosis.

`evals/setup_cpu.sh` is the canonical Torch bootstrap script. Its contents and
the prompts are embedded in the native tasks so setup also runs in the baseline,
where the skill itself is absent. After changing the script or dataset, refresh
and check the generated files:

```bash
python .github/scripts/prepare_skill_eval_tasks.py
python .github/scripts/prepare_skill_eval_tasks.py --check
```

The setup regression tests check synchronization and that capturing logs does
not hide a failing exit status. A successful local setup does not establish that
the shared Astra sandbox is healthy; inspect the live preflight result before
interpreting a missing Torch score as a skill-quality failure.

The requirements pin the cuEquivariance release or source revision whose APIs
were checked, and pin the JAX/Flax or CPU PyTorch versions.
They do not require sibling skills or a repository checkout inside the sandbox.
When updating the API baseline, update the pins and run the numerical checks
before changing the assertions. The CUDA backend constraints are checked as
guidance; these datasets do not benchmark GPU performance or validate CUDA kernels.

The tests under `.github/tests/test_*_skill_evals.py` verify the mathematical and
API expectations behind the prompts. `test_jax_skill_examples.py` executes each
published CPU example independently and checks layout conversion with duplicate
irreps and split polynomials with additional batch axes, parity, and JIT.
`test_torch_skill_examples.py` executes each published CPU recipe independently,
including tensor-product backward, and checks the GNN's determinism,
equivariance, and gradients. These are maintenance checks,
not live agent scores. To run them against
a development checkout with the relevant frontend dependencies installed:

```bash
pytest -q .github/tests/test_core_skill_evals.py
JAX_PLATFORMS=cpu pytest -q .github/tests/test_jax_skill_evals.py .github/tests/test_jax_skill_examples.py .github/tests/test_jax_skill_challenges.py .github/tests/test_skill_challenge_packaging.py
pytest -q .github/tests/test_torch_skill_evals.py .github/tests/test_torch_skill_examples.py .github/tests/test_torch_skill_challenges.py
```

To verify the pinned evaluation environment in a separate Python 3.12 environment:

```bash
python -m pip install -r skills/cuequivariance/scripts/requirements.txt \
  -r skills/cuequivariance-jax/scripts/requirements.txt \
  -r skills/cuequivariance-torch/scripts/requirements.txt pytest
JAX_PLATFORMS=cpu pytest -q .github/tests
```

The request workflow delegates live evaluation to NVSkills CI. With the same
SkillEvaluator version as the service (1.5.6 when this layout was introduced),
the dataset/config preflight can also be run locally:

```bash
skillevaluator tier3 validate skills/cuequivariance
skillevaluator tier3 validate skills/cuequivariance-jax
skillevaluator tier3 validate skills/cuequivariance-torch
```

### Challenge-v1 capability pilot

The challenge datasets contain two positive integration tasks and one nearby
negative task per frontend. Their IDs deliberately differ from the regression
IDs so results from the two datasets cannot be paired accidentally.

| Case suffix | Required outcome |
| --- | --- |
| `challenge-layout-v1` | Migrate original O3 `ir_mul` buffers with separated duplicate blocks and both parities into dictionaries, reconstruct the original order, and retain gradients, batch axes and transformation behavior. JAX also requires JIT; Torch includes noncontiguous tensors. |
| `challenge-graph-v1` | Compose radial weights, normalized harmonics, a channelwise product, destination accumulation and a projection. Match an independent NumPy edge-loop reference before and after a training update, retain gradients through every parameter and both floating inputs, and handle relabeling, edge permutations and isolated destinations. |
| `challenge-routing-v1` | Repair an ordinary JAX/Torch regression-loss broadcasting bug without invoking cuEquivariance or its skill. |

Prompts specify the public interface, mathematics, constraints and observable
outcomes. They leave the frontend APIs and implementation strategy to the agent.
Every task writes an importable `/workspace/solution.py`; execution claims remain
subject to the ordinary trajectory checks.

`.github/evals/cueq_challenges.py` independently executes the submitted artifact
in a bounded subprocess using the prepared CPU interpreter. The positive
oracles use original block slicing and core NumPy descriptors rather than the
submitted frontend forward. Checks include multiple data seeds, independent
forward values, finite differences, rotations, O3 inversion where applicable,
graph resizing and permutations, and a real parameter update.
It emits a binary `numerical_success` custom metric: all required numerical
checks must pass. A missing file, exception, detached gradient or timeout fails.
The nearby negative additionally checks the loss values and prediction gradients;
the ACES routing check detects skill activation beyond imports in the submission.

The maintainer references (`.github/evals/reference_*_challenges.py`) are outside
all skill directories and are never included in the pilot. Regression tests run
the verifier against those references and deliberate defects: overwritten
duplicate irreps, wrong layouts, parity merging, incorrectly shared edge data,
detached vectors, mean instead of sum aggregation, and the original MSE
broadcasting bug. These tests validate the benchmark and verifier, not agent lift.

Prepare a fresh pilot with Python, PyYAML and SkillEvaluator 1.5.6 available:

```bash
python .github/scripts/prepare_skill_challenges.py --export-harbor
```

This creates a unique directory under `.venv/` and prints a manifest location
and two evaluation commands. `--backend jax` or `--backend torch` selects one
frontend. `--output PATH` selects a new output directory; existing directories
are refused. Without `--export-harbor`, preparation needs only Python and PyYAML.

The staged skills retain the exact canonical `SKILL.md`, requirements and
environment setup. Only the selected dataset and grading mode change:
`aces_plus_custom` adds artifact verification while retaining the default ACES
metrics and quality gate. **`numerical_success` does not replace ACES's overall
score or change its gate.** Read both the numerical result and the normal
trajectory assessment; neither should be inferred from the other.
The Torch pilot renders native tasks from its challenge IDs and prompts and
uses the same 600-second setup budget. Its manifest also records the bootstrap
script and native task file digests.

The export validates both conditions, identical prompts/verifiers, and the
absence of eval metadata, grader code and reference solutions in the agent's
runtime bundle. This requires the builder's private evaluator-source support;
older builders are rejected. Do not use `--copy-repo`: the maintenance checkout
contains the reference answers.

The manifest records skill, dataset, configuration and verifier digests. Its
commands pin the models from the 2026-10-20 comparison:
`claude-code=aws/anthropic/bedrock-claude-opus-4-8` and
`codex=openai/openai/gpt-5.5`. They request baseline plus skill, three attempts
per case and no early stop on success: 36 executions per frontend, 72 for both.
Confirm those models are still available in the authenticated environment;
changing a model requires a new baseline with that same model.
The preparation command only stages and validates; it does not launch agents.
Before running the printed commands in the authenticated CI/runtime environment:

```bash
skillevaluator tier3 doctor --env-mode k8s-sandbox --agents claude-code,codex
```

Keep all attempts, infrastructure failures and trial artifacts. Fix the skill
text only after this first paired pilot, then measure both revisions on the
same frozen challenge dataset and environment. Reserve additional numerical
variants for later validation. Do not compare challenge lift directly with the
earlier six-case regression lift, or interpret local reference passes as evidence
of agent improvement. The ordinary NVSkills workflow continues to discover
`evals/evals.json`; promotion of the new benchmark follows the pilot.

### Interpreting Tier 3 comparisons

Use the exported JSON and per-trial evidence to compare revisions. The pipeline
gate, static quality score, execution coverage, and skill uplift answer different
questions; a green pipeline does not establish positive uplift.

- Check `tier3.coverage` first. Compare the same eligible agents and models across
  revisions. An agent excluded for `agent_process_exit` has no valid quality
  score; a displayed zero is not a measured skill regression. Exit 137 alone
  does not identify whether memory pressure, timeout, or another kill caused it.
- Use the precision recorded in `tier3.quality`. With default grading in 1.5.6, gate
  lift averages the with-skill minus baseline differences for the shared metrics
  `security`, `accuracy`, `goal_accuracy`, and `behavior_check`. Skill execution
  and tool productivity are N/A for baseline; token efficiency is a separate
  diagnostic and is outside that gate calculation.
- Pair trials by case ID. Inspect security evidence and numerical/workflow
  checks before using aggregate recommendations. Record total and productive
  tool calls plus uncached prompt and completion tokens to locate setup and
  exploration overhead.
- Hold the dataset, dependency versions, environment, agent models, evaluator,
  thresholds, and attempt policy fixed for a skill-only comparison. Capture
  package versions from the actual run: requirements files alone are insufficient.
  Preserve all attempts and infrastructure failures when repeating a comparison.
- One attempt per case is exploratory evidence. Repeat both arms before claiming
  a small improvement is reliable; do not select only the best rerun. Keep new
  evaluation cases in a separately identified benchmark until both revisions
  have been measured on it.

## Packaging

The release workflow materializes symlinks **before** running Hatchling, including
the existing LICENSE, README, VERSION and attribution links. Keep this step:
building an sdist directly from the linked development tree can leave links whose
targets are outside that distribution. Release wheels and sdists must carry a
regular `SKILL.md` resource.

The ordinary CPU test workflow runs the packaging regression check for each
package. To check all three locally with their runtime dependencies and Hatchling
installed:

```bash
python .github/scripts/check_skill_packaging.py
```

The script builds from a temporary materialized copy, checks the source CLI,
inspects wheel and sdist resources, rebuilds a wheel from each sdist, and verifies
the CLI imports and prints the resource from each wheel. It leaves the checkout's
symlinks intact. Use `--package cuequivariance_jax` (or either other Python package
name) to check one package.

## Service onboarding and publication

The source workflow is `.github/workflows/request-nvskills-ci.yml`, using the
public reusable workflow in `NVIDIA/skills`. Activation also requires:

1. An enabled `NVIDIA/cuEquivariance` entry in
   `NVIDIA/nvskills-ci/config/onboarded-repositories.json`, with the `external`
   profile and `main` as an allowed PR base.
2. The NVSkills GitHub App installed with access to this repository, and the
   repository secret `NVSKILLS_CI_DISPATCH_TOKEN` provisioned by the CI owners.
   Protect the request workflow with the repository's normal maintainer review.
3. The request workflow on the default branch so `issue_comment` events run.
   A maintainer can then comment `/nvskills-ci` on a same-repository PR that
   changes these skills. Fork PRs require a separate central policy opt-in.
4. The nv-carps standalone-skill catalog fix: Tier 2 should use the `skills/`
   catalog rather than traversing unrelated Python package symlinks. Team/plugin
   catalogs keep their existing handling. Cross-tree duplicate comparison is
   outside this standalone catalog's scope.
5. Successful service validation and genuine generated signatures, skill cards,
   and benchmarks. Only then merge the three-path registration in
   `NVIDIA/skills/components.d/cuequivariance.yml` and let catalog sync publish it.

The component registry must point at each of the three individual skill
directories, with matching catalog directory names. Do not manufacture generated
validation artifacts to satisfy catalog onboarding checks.

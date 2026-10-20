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

Each dataset contains five positive cases and one unrelated negative case.
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
arm before treating a run as an evaluation of the pinned environment. A shared
custom evaluation image or common `harbor.pre_agent_setup` is needed when the
sandbox does not already provide them. Provision it equally for both arms and
establish a new baseline when changing the environment.

The JAX and PyTorch setup commands publish `/workspace/.venv` only after the
pinned packages pass a CPU smoke check. Failed environments remain at their
unique `.cueq-env.*` paths for diagnosis; the next attempt creates a fresh one.
An existing `.venv`, including a broken symlink, is preserved and must pass the
check without being replaced. Setup uses binary wheels, skips bytecode
compilation, and fetches only the PyTorch wheel from the CPU index. A second
install resolves its dependencies and cuEquivariance from the normal index,
avoiding dependency-metadata incompatibilities with older host pip versions.
Both arms use the same setup and package pins.

These are repository-side setup improvements, not a change to SkillEvaluator's
healthcheck timeout. A successful local setup does not establish that the shared
Astra sandbox will finish within its budget. Inspect the live preflight result
before interpreting a missing Torch score as a skill-quality failure.

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
JAX_PLATFORMS=cpu pytest -q .github/tests/test_jax_skill_evals.py .github/tests/test_jax_skill_examples.py
pytest -q .github/tests/test_torch_skill_evals.py .github/tests/test_torch_skill_examples.py
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

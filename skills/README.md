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
environment. SkillEvaluator installs `scripts/requirements.txt` when generating
that environment. Those requirements pin cuEquivariance to the immutable source
revision whose APIs were checked, and pin the JAX/Flax or CPU PyTorch versions.
They do not require sibling skills or a repository checkout inside the sandbox.
When updating the API baseline, update the pins and run the numerical checks
before changing the assertions. The CUDA backend constraints are checked as
guidance; these datasets do not benchmark GPU performance or validate CUDA kernels.

The tests under `.github/tests/test_*_skill_evals.py` verify the mathematical and
API expectations behind the prompts. They are maintenance checks, not live agent
scores. To run them against a development checkout with the relevant frontend
dependencies installed:

```bash
pytest -q .github/tests/test_core_skill_evals.py
JAX_PLATFORMS=cpu pytest -q .github/tests/test_jax_skill_evals.py
pytest -q .github/tests/test_torch_skill_evals.py
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

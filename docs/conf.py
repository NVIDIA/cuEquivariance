# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime
import sphinx_rtd_theme


current_year = datetime.datetime.now().year

project = "NVIDIA cuEquivariance"
copyright = f"2024-{current_year}, NVIDIA Corporation & Affiliates"
author = "NVIDIA Corporation & Affiliates"

with open("../VERSION") as version_file:
    version = version_file.read().strip()
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["README.md", "_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The theme to use for HTML
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "style_nav_header_background": "#000000",
    "logo_only": False,
}
html_context = {
    "display_github": True,
    "github_user": "NVIDIA",
    "github_repo": "cuEquivariance",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_show_sphinx = False

# Logo and favicon
html_static_path = ["_static"]
html_logo = "_static/cuequivariance-240x240.png"
html_favicon = "_static/cuequivariance-32x32.png"

# -- Other options -----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

doctest_global_setup = """
import numpy as np
import torch
import jax
import jax.numpy as jnp
"""

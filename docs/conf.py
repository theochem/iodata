# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import runpy
import sys

from intersphinx_registry import get_intersphinx_mapping
from packaging.version import Version
from sphinx.ext.apidoc import main as main_api_doc

sys.path.append(os.path.dirname(__file__))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "IOData"
copyright = "2019, The IOData Development Team"  # noqa: A001
author = "The IOData Development Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Built-in Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Third-party extensions
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = get_intersphinx_mapping(packages={"python", "numpy", "scipy"})


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/table.css"]
# Embedded SVG as recommended in Furo template.
# See https://pradyunsg.me/furo/customisation/footer/#using-embedded-svgs
with open("github.svg") as fh:
    GITHUB_ICON_SVG = fh.read().strip()
html_theme_options = {
    "source_repository": "https://github.com/theochem/iodata",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/theochem/iodata",
            "html": GITHUB_ICON_SVG,
            "class": "",
        },
    ],
}

# -- Configuration for autodoc extensions ---------------------------------

autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "members": None,
    "inherited-members": True,
    "ignore-module-all": True,
}
napoleon_use_rtype = False
add_module_names = False


def autodoc_skip_member(_app, _what, name, _obj, skip, _options):
    """Decide which parts to skip when building the API doc."""
    if name == "__init__":
        return False
    return skip


def setup(app):
    """Set up sphinx."""
    app.connect("autodoc-skip-member", autodoc_skip_member)


# -- Configuration of mathjax extension -----------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "ket": [r"{\left\vert { #1 } \right\rangle}", 1],
            "bra": [r"{\left\langle { #1} \right\vert}", 1],
            "braket": [r"{\left\langle {#1} \mid { #2} \right\rangle}", 2],
            "ketbra": [
                r"{\left\vert { #1 } \right\rangle\left\langle { #2} \right\vert}",
                2,
            ],
            "ev": [r"{\left\langle {#2} \vert {#1} \vert {#2} \right\rangle}", 2],
            "mel": [r"{\left\langle{ #1 }\right\vert{ #2 }\left\vert{#3}\right\rangle}", 3],
        }
    },
}

# -- Utility functions -------------------------------------------------------


def _get_version_info():
    """Get the version as defined in pyproject.toml"""
    from setuptools_scm import Configuration
    from setuptools_scm._get_version_impl import _get_version

    config = Configuration.from_file("../pyproject.toml", "./")
    verinfo = Version(_get_version(config, force_write_version_files=False))
    return f"{verinfo.major}.{verinfo.minor}", str(verinfo)


def _pre_build():
    """Things to be executed before Sphinx builds the documentation"""
    runpy.run_path("gen_formats.py", run_name="__main__")
    runpy.run_path("gen_formats_tab.py", run_name="__main__")
    runpy.run_path("gen_inputs.py", run_name="__main__")
    os.environ["SPHINX_APIDOC_OPTIONS"] = ",".join(
        key for key, value in autodoc_default_options.items() if value is True
    )
    main_api_doc(["--append-syspath", "--output-dir=pyapi/", "../iodata/", "--separate"])


version, release = _get_version_info()
_pre_build()
html_title = f"{project} {version}"

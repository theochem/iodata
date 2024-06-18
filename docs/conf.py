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
GITHUB_ICON_SVG = """\
<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0\
-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63\
-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87\
.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .2\
7 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95\
.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8\
-8-8z"></path>
</svg>
"""
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

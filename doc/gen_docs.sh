#!/usr/bin/env bash

SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,inherited-members sphinx-apidoc \
    -a -o pyapi/ ../iodata/ --separate

./gen_formats.py > formats.rst
./gen_inputs.py > inputs.rst
./gen_formats_tab.py > formats_tab.inc

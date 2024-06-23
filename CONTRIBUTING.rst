..
    : IODATA is an input and output module for quantum chemistry.
    :
    : Copyright (C) 2011-2019 The IODATA Development Team
    :
    : This file is part of IODATA.
    :
    : IODATA is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : IODATA is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --


We'd love you to contribute.
Here are some practical tips to help you get started.

For IOData, you can follow the general
`QC-Devs Contributing Guide <https://github.com/theochem/.github/blob/main/CONTRIBUTING.md>`_.

When following that guide, you only need to take into account these specifics for IOData:

- The repository URL is: git@github.com:theochem/iodata.git
- To run all tests locally, you can use the following commands:

  .. code-block:: bash

     # Run tests excluding those marked as slow.
     pytest -m "not slow"
     # Build the documentation.
     (cd docs; make html)
     # Finally, if the above steps all pass, run the slow tests.
     pytest -m slow

  The first command already performs the majority of the tests,
  but only takes a few seconds to complete.
  The next two are also useful, but take more time.
  They are only worth trying if the first one works.

The sections below describe how to contribute to features that are specific to IOData.
These are not covered in the general QC-Devs Contributing Guide.


Adding a new file format
------------------------

Each file format is implemented in a module of the ``iodata.formats`` package.
These modules all use the same API. Please consult existing formats for some guidance,
e.g. the :py:mod:`iodata.formats.xyz` is a simple but complete example.
From the following list, ``PATTERNS`` and one of the functions must be implemented:

* ``PATTERNS = [ ... ]``:
  a list of glob patterns used to recognize file formats from the file names.
  This is used to select the correct module from ``iodata.formats`` in functions in ``iodata.api``.
* ``load_one``: load a single IOData object.
* ``dump_one``: dump a single IOData object.
* ``load_many``: load multiple IOData objects (iterator) from a single file.
* ``dump_many``: dump multiple IOData objects (iterator) to a single file.


``load_one`` function: reading a single IOData object from a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To support reading a new file format,
the module must include a ``load_one`` function with the following signature:

.. code-block:: python

    @document_load_one("format", ['list', 'of', 'guaranteed', 'attributes'],
                       ['list', 'of', 'attributes', 'which', 'may', 'be', 'read'],
                       notes)
    def load_one(lit: LineIterator) -> dict:
        """Do not edit this docstring. It will be overwritten."""
        # Actual code to read the file


The ``LineIterator`` instance provides a convenient interface for reading files
and can be found in ``iodata.utils``.
As a rule of thumb, always use ``next(lit)`` to read a new line from the file.
You can use this iterator in several ways:

.. code-block:: python

    # When you need to read one line.
    line = next(lit)

    # When sections appear in a file in fixed order, you can use helper functions.
    data1 = _load_helper_section1(lit)
    data2 = _load_helper_section2(lit)

    # When you intend to read everything in a file (not for trajectories).
    for line in lit:
        # Do something with the line.
        ...

    # When you just need to read a section.
    for line in lit:
        # Do something with the line.
        if done_with_section:
            break

    # When you need a fixed numbers of lines, say 10.
    for i in range(10):
        line = next(lit)

    # More complex example, in which you detect several sections
    # and call other functions to parse those sections.
    # The code is not sensitive to the order of the sections.
    while True:
        line = next(lit)
        if end_pattern in line:
            break
        elif line == 'section1':
            data1 = _load_helper_section1(lit)
        elif line == 'section2':
            data2 = _load_helper_section2(lit)

    # Same as above, but reading until the end of the file.
    # You cannot use a for loop when multiple lines must be read in one iteration.
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        if end_pattern in line:
            break
        elif line == 'section1':
            data1 = _load_helper_section1(lit)
        elif line == 'section2':
            data2 = _load_helper_section2(lit)


In some cases, you may need to move a line back in the file because it was read too early.
For example, in the Molden format, this is sometimes unavoidable.
If necessary, you can *push back* the line for later reading with ``lit.back(line)``.

.. code-block:: python

    # When you just need to read a section.
    for line in lit:
        # Do something with line.
        if done_with_section:
            # Only now it becomes clear that you've read one line too far.
            lit.back(line)
            break

When you encounter a file format error while reading the file, raise a ``LoadError`` exception:

.. code-block:: python

    from ..utils import LoadError

    @document_load_one(...)
    def load_one(lit: LineIterator) -> dict:
        ...
        if something_wrong:
            raise LoadError("Describe the problem that made it impossible to load the file.", lit)

The error that appears in the terminal will automatically include the file name and line number.
If your code has already read the full file and encounters an error when processing the data,
you can use ``raise LoadError("Describe problem in a sentence.", lit.filename)`` instead.
This way, no line number is included in the error message.

Sometimes, it is possible to correct errors while reading a file.
In this case, you should warn the user that the file contains (fixable) errors:

.. code-block:: python

    from warnings import warn

    from ..utils import LoadWarning

    @document_load_one(...)
    def load_one(lit: LineIterator) -> dict:
        ...
        if something_fixed:
            warn(LoadWarning("Describe the issue that was fixed while loading.", lit), stacklevel=2)

Always use ``stacklevel=2`` when raising warnings.


``dump_one`` functions: writing a single IOData object to a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``dump_one`` functions are conceptually simpler:
they just take an open file object and an ``IOData`` instance as arguments,
and should write the data to the open file.

.. code-block:: python

    @document_dump_one("format", ['guaranteed', 'attributes'], ['optional', 'attribtues'], notes)
    def dump_one(fh: TextIO, data: IOData):
        """Do not edit this docstring. It will be overwritten."""
        # Code to write data to fh.


``load_many`` function: reading multiple IOData objects from a single file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function works essentially in the same way as ``load_one``, but can load multiple molecules.
For example:

.. code-block:: python

    @document_load_many("XYZ", ['atcoords', 'atnums', 'title'])
    def load_many(lit: LineIterator) -> Iterator[dict]:
        """Do not edit this docstring. It will be overwritten."""
        # XYZ Trajectory files are a simple concatenation of individual XYZ files,
        # making it trivial to load many frames.
        while True:
            try:
                yield load_one(lit)
            except StopIteration:
                return

The XYZ trajectory format is simply a concatenation of individual XYZ files,
so you can use the ``load_one`` function to read a single frame.
Some file formats require more complicated approaches.
In any case, the ``yield`` keyword must be used for every frame read from a file.


``dump_many`` function: writing multiple IOData objects to a single file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Also ``dump_many`` is very similar to ``dump_one``,
but just takes an iterator over multiple IOData instances as an argument.
It is expected to write them all to a single open file object.
For example:

.. code-block:: python

    @document_dump_many("XYZ", ['atcoords', 'atnums'], ['title'])
    def dump_many(f: TextIO, datas: Iterator[IOData]):
        """Do not edit this docstring. It will be overwritten."""
        # Similar to load_many, this is relatively easy.
        for data in datas:
            dump_one(f, data)

Again, we take advantage of the simple structure of the XYZ trajectory format,
i.e. the simple concatenation of individual XYZ files.
For other formats, this might be more complicated.


Notes on attrs
--------------

IOData uses the `attrs`_ library, not to be confused with the `attr`_ library,
for classes that represent data loaded from files:
``IOData``, ``MolecularBasis``, ``Shell``, ``MolecularOrbitals`` and ``Cube``.
This allows for basic attribute validation, which eliminates potentially silly bugs.
(See ``iodata/attrutils.py`` and the use of ``validate_shape`` in all of these classes.)

The following ``attrs`` functions may be useful when working with these classes:

- The data can be converted to plain Python data types using the ``attrs.asdict`` function.
  Make sure you add the ``retain_collection_types=True`` option, to avoid the following problem:
  https://github.com/python-attrs/attrs/issues/646
  For example.

  .. code-block:: python

      from iodata import load_one
      import attrs
      iodata = load_one("example.xyz")
      fields = attrs.asdict(iodata, retain_collection_types=True)

  A similar ``astuple`` function works as you would expect.

- A `shallow copy`_ with a few modified attributes can be created using ``attrs.evolve``:

  .. code-block:: python

      from iodata import load_one
      import attrs
      iodata1 = load_one("example.xyz")
      iodata2 = attrs.evolve(iodata1, title="another title")

  The use of ``evolve`` becomes mandatory when you want to change two or more attributes
  whose shape must be consistent.
  For example, the following will fail:

  .. code-block:: python

      from iodata import IOData
      iodata = IOData(atnums=[7, 7], atcoords=[[0, 0, 0], [2, 0, 0]])
      # The next line will fail because the size of atnums and atcoords becomes inconsistent.
      iodata.atnums = [8, 8, 8]
      iodata.atcoords = [[0, 0, 0], [2, 0, 1], [4, 0, 0]]

  The following code, which has the same intent, does work:

  .. code-block:: python

      from iodata import IOData
      import attrs
      iodata1 = IOData(atnums=[7, 7], atcoords=[[0, 0, 0], [2, 0, 0]])
      iodata2 = attrs.evolve(
          iodata1,
          atnums=[8, 8, 8],
          atcoords=[[0, 0, 0], [2, 0, 1], [4, 0, 0]],
      )

  For brevity, lists (of lists) have been used in these examples.
  These are always converted to arrays by the constructor or when assigned to attributes.


.. _Bash: https://en.wikipedia.org/wiki/Bash_(Unix_shell)
.. _Python: https://en.wikipedia.org/wiki/Python_(programming_language)
.. _GitHub Fork feature: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo
.. _Pansini guide: https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53#rules-for-a-great-git-commit-message-style
.. _direnv: https://direnv.net/
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _type hinting: https://docs.python.org/3/library/typing.html
.. _pytest: https://docs.pytest.org/en/stable/
.. _numpy.testing: https://numpy.org/doc/stable/reference/routines.testing.html#module-numpy.testing
.. _codecov: https://codecov.io/
.. _semantic line breaks: https://sembr.org/
.. _NumPy's docstring format: https://numpydoc.readthedocs.io/en/latest/format.html
.. _atomic units: https://en.wikipedia.org/wiki/Atomic_units
.. _attrs: https://www.attrs.org/en/stable/
.. _attr: https://github.com/denis-ryzhkov/attr
.. _shallow copy: https://docs.python.org/3/library/copy.html?highlight=shallow

We'd love you to contribute.
Here are some practical tips to help you get started.

This document assumes that you are familiar with `Bash`_ and `Python`_.
If you are not familiar with git or GitHub,
we recommend that you take a look at the `GitHub Guides <https://guides.github.com/>`_
or this `Git Book <https://git-scm.com/book/en/v2>`_.


Development setup
-----------------

The following instructions will create a local clone of the IOData Git repository
that is suitable for development.
It is assumed that you have Python 3.9 or later installed on your system.

The commands below are provided as a guideline to get a working development setup.
Feel free to adapt them to your personal preferences.

- For Linux or macOS:

  .. code-block:: bash

    git clone git@github.com:theochem/iodata.git
    cd iodata
    python -m venv venv
    source venv/bin/activate
    pip install -e .[dev]
    pre-commit install

  The manual activation of the virtual environment with ``source venv/bin/activate``
  has to be repeated every time you open a new terminal window, which is impractical.
  We recommend using `direnv`_ to automate the activation of the virtual environment.
  After installing and setting up direnv, you can configure it for IOData as follows:

  .. code-block:: bash

    echo 'source venv/bin/activate' > .envrc
    direnv allow

  Now, the virtual environment is always activated when you change to the ``iodata`` directory.

- For Windows: ...


Git and GitHub configuration
----------------------------

- If you don't already have an SSH key pair, create one using the following terminal command:

  .. code-block:: bash

    ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f ${HOME}/.ssh/id_rsa_github_com

  An empty passphrase is convenient and should be fine.
  This will generate a private and a public key in ``${HOME}/.ssh``.

- Upload your *public* SSH key to `<https://github.com/settings/keys>`_.
  This is a single long line in ``${HOME}/.ssh/id_rsa_github_com.pub``,
  which you can copy and paste into your browser.

- Configure SSH to use this key pair for authentication when pushing branches to GitHub.
  Add the following to your ``.ssh/config`` file:

  .. code-block::

    # Do not allow SSH to try all keys, as SSH servers only accept a few attempts.
    # Keys must be specified for each host.
    IdentitiesOnly yes

    Host github.com
        Hostname github.com
        ForwardX11 no
        IdentityFile %d/.ssh/id_rsa_github_com

  (Make sure you have the correct path to the private key file.)

- Configure Git to use the name and email address associated with your GitHub account:

  .. code-block:: bash

    git config --global user.name "Your Name"
    git config --global user.email "youremail@yourdomain.com"

- Create a fork of the project, using the `GitHub Fork feature`_.
  Add your fork as a second remote to your local repository,
  for which we will use the short name ``mine`` below, but any short name will do:

  .. code-block:: bash

    git remote add mine git@github.com:<your-github-account>/iodata.git



Github work flow
----------------

1. If you want to make changes,
   please open a GitHub issue first, or search for an existing issue,
   so we can discuss it before you code:

   - Use the issue to explain what you want to accomplish and why.
     This often saves a lot of time and effort in the long run.
   - Also use the issue also to plan your changes.
     Try to solve only one problem at a time,
     rather than fixing multiple problems and adding different features at once.
     Small changes are easier to manage, also for the reviewer in the last step below.
   - Finally, mention in the corresponding issue when you are working on it.
     Avoid duplicate efforts by assigning yourself to the issue.

   Once you have determined what changes need to be made to the source code,
   you can proceed with the following steps.
   (It is assumed that you have created the development setup above.)


2. Create a new branch, with a name that indicates the purpose of your change:

   .. code-block:: bash

     git checkout -b new-feature


3. Make changes to the source,
   for which the following section discusses conventions, recommendations and guidelines.


4. Verify that all tests pass and that the documentation still builds without warnings or errors:

   .. code-block:: bash

     pytest
     (cd docs; make html)


5. Commit your changes using ``git commit``.

   You will notice that ``pre-commit`` checks for and possibly fixes minor problems.
   If it finds something (even if is automatically fixed), it will abort the commit.
   This gives you a chance to fix those problems or check the automatic fixes first.
   When you are happy with all the cleanup, run ``git commit`` again.

   When prompted, write a meaningful commit message.
   The first line is a short summary, written in the imperative mood.
   Optionally, this can be followed by  blank line and a longer description.
   We follow the `Pansini Guide`_ style for commit messages.
   This makes it easier to generate changelogs and to understand the history of the project.


6. Push your branch to your forked repository on GitHub:

   .. code-block:: bash

       git push mine -u new-feature

   A link should appear in the terminal that will take you to the next step.


7. Make a pull request from your ``new-feature`` branch in your forked repository
   to the ``main`` branch in the original repository.


8. Wait for the GitHub Actions to complete.
   These should pass.
   Typically, someone should review your pull request within a few days or weeks.
   Ideally, the review will result in minor corrections.
   We'll do our best to avoid major issues in step 1.


General code guidelines
-----------------------

Code contributions are the most common type of contribution.
We welcome all types of contributions,
including bug fixes, new features, and documentation updates.
All code contributions should follow the guidelines below.


1. Code should be well-written
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general, we follow the `PEP 8`_ style guide for Python code.
Most style conventions are taken care off by pre-commit.
If you have it installed as described above, the basics are covered.

We strive to write compact and elegant Python code
that is not fully addressed by the linters configured in pre-commit.
This is one of the points we look for when reviewing a pull request.


2. Code should be well-documented
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- All functions and classes (except tests) should have a docstring
  explaining what they do and how to use them.
- We encourage the use of mathematical equations in docstrings, using LaTeX notation.
- We do not encourage the use of example code in docstrings,
  as it can become outdated and difficult to maintain.
  Instead, we encourage the use stand-alone examples that can be executed and tested separately.
- We use `type hinting` to document the types of function (and method) arguments and return values.
- We use `NumPy's docstring format`_, except that types are documented with type hints.
- We recommend using `semantic line breaks`_ when writing comments or documentation.
  This improves the readability of ``git diff`` when reviewing changes.
- Your code will be read by other developers, so it should be easy to understand.
  If a part of the code seems complex, it should have comments explaining what it does.
  (When in doubt, add a comment!)
  Good comments emphasize the intent of the code, rather than literally describing it.


3. Code should be tested
^^^^^^^^^^^^^^^^^^^^^^^^

- All code should be tested.
  We use `pytest`_ for testing.

  - When you add new code, you should also add tests for it.
  - If you fix a bug, you should also add a test that fails without and passes with your fix.
  - Use ``np.testing.assert_allclose`` and  ``np.testing.assert_equal``
    to compare floating-point and integer NumPy arrays, respectively.
    ``np.testing.assert_allclose`` can also be used for  comparing floating point scalars.
    In all other cases (not involving floating point numbers),
    the simple ``assert a == b`` works just as well, and is more readable.
    See `numpy.testing`_ for more details.

- We use `codecov`_ in most of our packages to check the code coverage of our tests.
  Please make sure that your code is well-tested and that the coverage does not decrease.


4. Code should be consistent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Code needs to be consistent with the rest of the codebase.
This makes it easier to review and maintain.
This includes:

- Variable names, function names, and class names should be consistent
  with the rest of the codebase.
- Most QC-Devs repositories use `atomic units`_ internally.
  We ask that you try to preserve this (for consistency), but still document units.
- Even more, variable names should be consistent across QC-Devs packages.
  We are in the process of making a glossary, but for now,
  please take a look at the existing codes on `GitHub <https://github.com/theochem>`_ and try to match them.

We value your contributions and appreciate your efforts to improve QC-Devs packages.
By following these guidelines, you can ensure smoother collaboration and enhance the overall quality of the project.


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

When you encounter a file format error while reading the file, call ``lit.error(msg)``,
where ``msg`` is a short message describing the problem.
The error that appears in the terminal will automatically include the file name and line number.


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

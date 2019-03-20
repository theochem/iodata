We'd love you to contribute. Here are some practical hints to help out.


General recommendations
=======================

- Please, be careful with tools like autopep8, black or yapf. They may result in
  a massive number of changes, making pull requests harder to review. Also, when
  using them, use a maximum line length of 100. To avoid confusion, only clean
  up the code you are working.


Adding new file formats
=======================

Each file format is implemented in a module of the package ``iodata.formats``.
These modules all follow the same API. The following must always be present

.. code-block :: python

    # A list of glob patterns used to recognize file formats from the file names.
    # This is used to use the correct module in ``iodata.formats`` when
    # using the ``load_one`` and ``dump_one`` functions.
    patterns = [ ... ]


`load` functions: reading from files
------------------------------------

In order to read from a new file format, the module must contain a ``load``
function with the following signature:

.. code-block :: python

    def load(lit: LineIterator) -> Dict:
        """Load data from <please-fill-in-some-info>.

        Parameters
        ----------
        lit
            The line iterator to read the data from.

        Returns
        -------
        out : dict
            Output dictionary contain keys ...
            Output dictionary may contain keys ...

        """
        # Actual code to read the file


The ``LineIterator`` instance provides a convenient interface for reading files
and can be found in ``iodata.utils``. As a rule of thumb, always use
``next(lit)`` to read a new line from the file. You can use this iterator in
a few ways:

.. code-block:: python

    # When you need to read one line.
    line = next(lit)

    # When section appear in a file in fixed order, you can use helper functions.
    data1 = _load_helper_section1(lit)
    data2 = _load_helper_section2(lit)

    # When you intend to read everything in a file (not for trajectories).
    for line in lit:
        # do something with line.

    # When you just need to read a section.
    for line in lit:
        # do something with line
        if done_with_section:
            break

    # When you need a fixed numbers of lines, say 10.
    for i in range(10):
        line = next(lit)

    # More complex example, in which you detect several sections and call other
    # functions to parse those sections. The code is not sensitive to the
    # order of the sections.
    while True:
        line = next(lit)
        if end_pattern in line:
            break
        elif line == 'section1':
            data1 = _load_helper_section1(lit)
        elif line == 'section2':
            data2 = _load_helper_section2(lit)

    # Same as above, but reading till end of file. You cannot use a for loop
    # when multiple lines must be read in one iteration.
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


In some cases, one may have to push back a line because it was read to early.
For example. in the Molden format, that is sometimes unavoidable. Then you
can *push back* the line for later reading with ``lit.back(line)``.

.. code-block:: python

    # When you just need to read a section
    for line in lit:
        # do something with line
        if done_with_section:
            # only now it becomes clear that you've read one line to far
            lit.back(line)
            break

When you encounter a file-format error while reading the file, call
``lit.error(msg)``, where ``msg`` is a short message describing the problem.
The error appearing on screen will automatically also contain the


`dump` functions: writing to files
----------------------------------

TODO


Github work flow
================

1. Before diving into technicalities: if you intend to make major changes,
   beyond fixing bugs and small functionality improvements, please open a Github
   issue first, so we can discuss before coding. Please explain what you intend
   to accomplish and why. That often saves a lot of time and trouble in the long
   run.

   Use the issue to plan your changes. Try to solve only one problem at a time,
   instead of fixing several issues and adding different features in a single
   shot. Small changes are easier to handle, also for the reviewer in the last
   step below.

   Mention in the corresponding issue when you are working on it. "Claim" the
   issue to avoid duplicate efforts.

2. TODO: Roberto needs more testing.
   Install Roberto, which is the driver for our CI setup. It can also replicate
   the continuous integration on your local machine, which makes it easier to
   prepare a passable pull request. See TODO FIX URL.

3. Make a fork of the project, using the Github "fork" feature.

4. Clone the original repository on your local machine and enter the directory

   .. code-block:: bash

    git clone git@github.com:theochem/iodata.git
    cd iodata

5. Add your fork as a second remote to your local repository, for which we will
   use the short name `mine` below, but any short name is fine:

   .. code-block:: bash

    git remote add mine git@github.com:<your-github-account>/iodata.git

6. Make a new branch, with a name that hints at the purpose of your
   modification:

   .. code-block:: bash

    git checkout -b new-feature

7. Make changes to the source. Please, make it easy for others to understand
   your code. Also, add tests that verify your code works as intended.
   Rules of thumb:

   - Write transparent code, e.g. self-explaining variable names.
   - Add comments to passages that are not easy to understand at first glance.
   - Write docstrings explaining the API.
   - Add unit tests when feasible.

8. Commit your changes with a meaningful commit message. The first line is a
   short summary, written in the imperative mood. Optionally, this can be
   followed by an empty line and a longer description.

   If you feel the summary line is too short to describe what you did, it
   may be better to split your changes into multiple commits.

9. TODO: MAKE THIS WORK!
   Run Roberto and fix all problems it reports. Either one of the following
   should work

   .. code-block:: bash

    rob                 # Normal case
    python3 -m roberto  # Only if your PATH is not set correctly

   Style issues, failing tests and packaging issues should all be detected at
   this stage.

10. Push your branch to your forked repository on Github:

    .. code-block:: bash

        git push mine -u new-feature

    A link should be printed on screen, which will take the next step for you.

11. Make a pull request from your branch `new-feature` in your forked repository
    to the `master` branch in the original repository.

12. Wait for the tests on Travis-CI to complete. These should pass. Also
    coverage analysis will be shown, but this is merely indicative. Normally,
    someone should review your pull request in a few days. Ideally, the review
    results in minor corrections at worst. We'll do our best to avoid larger
    problems in step 1.
